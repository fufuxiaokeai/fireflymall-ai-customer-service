"""
验证内容：
1. ONNX 加载与推理（动态序列长度）
2. 用定稿阈值复现测试集召回（期望 ≈0.725，1% 业务误报口径）
3. 业务负例误报（期望 0）
4. 0.9 硬拦线以上样本明细（期望全部是注入）
5. CPU 推理延迟（ms/条）

用法：python tests/verify_onnx.py
依赖：onnxruntime transformers（本地首次运行会下载 tokenizer）
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ONNX_PATH = ROOT / 'model' / 'roberta_inj.onnx'
THR_LABEL, THR_HARD = 0.018, 0.9   # 定稿参数（2026-08-19 第三轮训练）
MAX_LEN = 128

os.environ['HF_HOME'] = str(ROOT / 'model' / 'hfl' / 'chinese-roberta-wwm-ext')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

try:
    import onnxruntime as ort
except ImportError:
    print('[错误] 缺少 onnxruntime: pip install onnxruntime')
    sys.exit(1)
try:
    import truststore
    truststore.inject_into_ssl()
    from transformers import AutoTokenizer
except ImportError:
    print('[错误] 缺少 transformers: pip install transformers')
    sys.exit(1)


def main():
    if not ONNX_PATH.exists():
        print(f'[错误] 缺少 {ONNX_PATH} —— 请先在 Kaggle 导出 ONNX 并放入 model/ 目录')
        return

    print('加载 tokenizer（hfl/chinese-roberta-wwm-ext，首次需下载）...')
    # local_files_only：tokenizer 已缓存在 HF_HOME（model/hfl/...），跳过联网（hf-mirror 超时坑）
    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext', local_files_only=True)
    sess = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
    print('ONNX 加载 OK ✓')

    def predict(texts, batch_size=128):
        # 分批推理：全量一次会 OOM（10004 条 attention 矩阵需 ~7.3GB 连续内存）
        probs = []
        for i in range(0, len(texts), batch_size):
            chunk = list(texts[i:i + batch_size])
            enc = tokenizer(chunk, max_length=MAX_LEN, truncation=True,
                            padding='max_length', return_tensors='np')
            logits = sess.run(['logits'], {
                'input_ids': enc['input_ids'],
                'attention_mask': enc['attention_mask'],
                'token_type_ids': enc['token_type_ids'],
            })[0]
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs.append(e[:, 1] / e.sum(axis=1))
        return np.concatenate(probs)

    test = pd.read_csv(ROOT / 'data/injection_dataset/test_external.csv')
    biz = pd.read_csv(ROOT / 'data/injection_dataset/business_negatives.csv')

    # ---- 1. 推理延迟（单条 ×10 取均值）----
    t0 = time.time()
    for _ in range(10):
        predict(test.text.tolist()[:1], batch_size=1)
    latency = (time.time() - t0) / 10 * 1000
    print(f'\n[1] CPU 推理延迟: {latency:.1f} ms/条（MAX_LEN={MAX_LEN}）')

    # ---- 2. 全量推理 ----
    t0 = time.time()
    test_probs = predict(test.text.tolist())
    biz_probs = predict(biz.text.tolist())
    print(f'[2] 全量推理 {len(test_probs) + len(biz_probs)} 条耗时 {time.time() - t0:.1f}s')

    # ---- 3. 定稿阈值复现 ----
    y_test = test.label.astype(int).values
    recall = float(((test_probs >= THR_LABEL) & (y_test == 1)).sum() / y_test.sum())
    fp_biz = float((biz_probs >= THR_LABEL).mean())
    hard = test_probs >= THR_HARD
    hard_recall = float((hard & (y_test == 1)).sum() / y_test.sum())
    hard_fp = float((biz_probs >= THR_HARD).mean())
    print(f'\n[3] 定稿阈值复现（thr={THR_LABEL}）:')
    print(f'    测试集注入召回: {recall:.3f}（Kaggle 实测期望 ≈0.725）')
    print(f'    业务负例误报: {fp_biz:.3f}（期望 0）')
    print(f'    硬拦线 {THR_HARD}: 拦截 {hard.sum()} 条, 注入召回 {hard_recall:.3f}, 业务误报 {hard_fp:.3f}')

    # ---- 4. 硬拦线样本明细 ----
    if hard.sum():
        print(f'\n[4] 硬拦线以上样本（{hard.sum()} 条，期望全部是注入）:')
        for _, r in test[hard].head(10).iterrows():
            print(f'    [{r["label"]}] {r["text"][:60]!r}')
    else:
        print('\n[4] 硬拦线以上 0 条（阈值 0.9 过严？可下调观察）')

    # ---- 5. 业务负例最高分（摸误报边际）----
    top = np.argsort(biz_probs)[-5:][::-1]
    print('\n[5] 业务负例最高分 Top5（边际风险样本）:')
    for i in top:
        print(f'    {biz_probs[i]:.4f}  {biz.text.iloc[i][:40]!r}')


if __name__ == '__main__':
    main()
