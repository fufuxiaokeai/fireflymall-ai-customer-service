"""本地环境口径扫描（2026-08-19）：以本地 ONNX + tokenizers 0.22 实测定输入层阈值"""
import os
import sys

os.environ['HF_HOME'] = r'model/hfl/chinese-roberta-wwm-ext'
import numpy as np
import pandas as pd
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP = r'model/hfl/chinese-roberta-wwm-ext/hub/models--hfl--chinese-roberta-wwm-ext/snapshots/5c58d0b8ec1d9014354d691c538661bf00bfdb44/tokenizer.json'

tok = Tokenizer.from_file(SNAP)
tok.enable_truncation(max_length=128)
tok.enable_padding(pad_id=0, pad_token='[PAD]')
sess = ort.InferenceSession(os.path.join(ROOT, 'model/roberta_inj.onnx'), providers=['CPUExecutionProvider'])


def predict(texts, bs=256):
    out = []
    for i in range(0, len(texts), bs):
        encs = tok.encode_batch(list(texts[i:i + bs]))
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        logits = sess.run(['logits'], {
            'input_ids': ids, 'attention_mask': mask, 'token_type_ids': np.zeros_like(ids)})[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        out.append(e[:, 1] / e.sum(axis=1))
    return np.concatenate(out)


def main():
    val = pd.read_csv(os.path.join(ROOT, 'data/injection_dataset/val.csv'))
    test = pd.read_csv(os.path.join(ROOT, 'data/injection_dataset/test_external.csv'))
    biz = pd.read_csv(os.path.join(ROOT, 'data/injection_dataset/business_negatives.csv'))
    y_val, y_test = val.label.astype(int).values, test.label.astype(int).values

    print('val 推理...', flush=True)
    pv = predict(val.text.tolist())
    print('test 推理...', flush=True)
    pt = predict(test.text.tolist())
    print('biz 推理...', flush=True)
    pb = predict(biz.text.tolist())

    biz_sorted = np.sort(pb)[::-1]
    print('\n业务负例分数 Top5:', [round(x, 4) for x in biz_sorted[:5]], flush=True)
    print('\n口径扫描（本地实测）:', flush=True)
    for n_fp, note in [(0, '0 条误拦'), (1, '1 条(1.4%)'), (2, '2 条(2.9%)')]:
        thr = biz_sorted[n_fp] + 0.0001 if n_fp < len(biz_sorted) else 0.0
        print(f'  {note}: thr>{thr:.4f}  val召回={(pv[y_val==1]>=thr).mean():.3f}  '
              f'测试召回={(pt[y_test==1]>=thr).mean():.3f}  硬拦率={(pt>=0.9).mean():.3f}', flush=True)

    hard = pt >= 0.9
    print(f'\n硬拦(>=0.9): {hard.sum()} 条, 注入占比 {(hard & (y_test==1)).sum()/max(hard.sum(),1):.3f}, '
          f'业务负例>=0.9: {(pb>=0.9).sum()}', flush=True)

    # 模板层（商城真实攻击形态）在本地环境的召回
    templ = val[val.source.isin(['template', 'noise']) & (val.label.astype(int) == 1)]
    if len(templ):
        pt2 = predict(templ.text.tolist())
        print(f'模板层注入召回(thr=0.018): {(pt2>=0.018).mean():.3f} (n={len(templ)})', flush=True)


if __name__ == '__main__':
    main()
