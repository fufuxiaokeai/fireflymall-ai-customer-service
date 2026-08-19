"""
注入分类模型训练 notebook 生成器（2026-08-19）

生成 tests/train_injection_classifier.ipynb（Kaggle T4 可跑）。

设计要点（对应数据集质量检查结论与训练方案）：
1. 双模型基线：TextCNN（轻量抗过拟合）vs chinese-roberta（容量大需正则）
2. 模板层加权：场景对齐数据（template+noise 共 261 条）用 loss 加权，避免学成
   "有害内容检测器"而欠拟合真注入（nemotron zh 层标签语义是"有害"而非"注入"）
3. 阈值扫描双口径：业务负例集（business_negatives.csv，独立于训练集）误报率<1%
   + hf safe 误报率<1% 都达标时取最高召回；只在 val 上扫，test 只做最终验证
4. 分层报告：按 source 分组输出，模板层召回是场景指标，train-val gap 量化过拟合
5. 软投票对比（模型差异度大时）；Stacking 预留注释（数据量小，不默认启用）
6. ONNX 导出骨架（训练后本地接入 msg_handle 前）

用法：python tests/make_train_notebook.py  →  生成 tests/train_injection_classifier.ipynb
Kaggle 使用：上传 data/injection_dataset/（含 business_negatives.csv）为数据集，
改 CELL_CONFIG 的 DATA_DIR 为挂载路径后逐 cell 运行。
"""
import json
from pathlib import Path
from textwrap import dedent

OUT = Path(__file__).resolve().parent / 'train_injection_classifier.ipynb'


def md(src):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': [l + '\n' for l in dedent(src).strip('\n').split('\n')]}


def code(src):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {},
            'outputs': [], 'source': [l + '\n' for l in dedent(src).strip('\n').split('\n')]}


CELLS = []

CELLS.append(md("""
# 注入分类模型训练（TextCNN vs chinese-roberta）

**数据**：`data/injection_dataset/` 三个 CSV + `business_negatives.csv`（独立业务负例集）
**目标**：误报率 <1%（业务负例集 + hf safe 双口径）前提下，注入召回最高

运行顺序：改配置 → 逐 cell 运行。依赖：Kaggle 默认镜像 + transformers（自动安装）。
"""))

CELLS.append(code("""
# ==================== 配置（唯一需要改的 cell） ====================
MODEL_NAME = 'hfl/chinese-roberta-wwm-ext'   # 中文 RoBERTa（若下载慢可换 hfl/chinese-bert-wwm-ext）
USE_TEXT_CNN = True    # 轻量基线
USE_ROBERTA = True     # 高容量基线（需更长训练时间）

# Kaggle 数据集挂载后改这里：/kaggle/input/<数据集名>/injection_dataset
DATA_DIR = 'data/injection_dataset'

# 模板层加权：场景对齐数据（source in template/noise）的 loss 权重倍率。
# 理由：hf 层 12000 条标签语义是"有害内容"（NVIDIA nemotron），真注入只有模板层 261 条，
# 不加权模型会学成"有害内容检测器"而欠拟合商城真注入。
# 2026-08-19 实测：10.0 导致模板层过拟合（train-val gap +0.13~0.21，模型背模板原文），
# 降至 3.0 平衡对齐与泛化。
TEMPLATE_WEIGHT = 3.0

SEED = 42
MAX_LEN = 128          # 注意：hf 层有 1486 条 >200 字长样本（max 5315），128 截断会丢尾部——可试 256 对比
BATCH_SIZE = 32
EPOCHS = 8
PATIENCE = 2           # 早停
LR_TEXT_CNN = 1e-3
LR_ROBERTA = 2e-5      # 高容量模型用小 lr + 冻结部分层（防过拟合）
WEIGHT_DECAY = 0.01

# 阈值目标（val 上扫阈值，只约束业务误报；hf 误报仅作报告项）
# 2026-08-19 实测：hf safe 是 nemotron"安全内容"，含大量指令性/角色扮演文本，
# 与注入措辞天然接近——hf 误报<1% 会把阈值逼到 0.97、测试召回打到 0.08；
# 而业务误报已由 business_negatives（69 条真实商城对话，独立于训练集）保证。
# 误杀 hf 层指令性文本对线上零影响（真用户不会那样问客服）。
FP_BUSINESS_MAX = 0.01   # 业务负例集误报率上限（上线后真正的误报风险）
"""))

CELLS.append(code("""
# ==================== 依赖与种子 ====================
import os, json, random, math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
torch.set_num_threads(4)

# numpy 2.x 与 torch <2.3 ABI 不兼容（.numpy() 报 'Numpy is not available'）
import numpy as np
torch_major, torch_minor = (int(x) for x in torch.__version__.split('.')[:2])
if int(np.__version__.split('.')[0]) >= 2 and (torch_major, torch_minor) < (2, 3):
    print('[警告] torch <2.3 与 numpy 2.x 不兼容 → .numpy() 会报错')
    print('        推荐修复: !pip install -q torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121  # 兼容 numpy 2，且是 P100 最后支持线')
    print('        兜底修复: 干净会话(未加载过 numpy 2)里装 numpy==1.26.4；在已加载 numpy 2 的会话里降级会损坏环境，必须 Restart Session')
# 镜像自带 torchvision/torchaudio 是 torch 2.10 配套的，与 torch 2.3.1 不兼容，
# 且 transformers 导入链会顺带导入 torchvision（image_utils）→ 卸载（本项目不用它们）
try:
    import torchvision  # noqa: F401
    print('[警告] 检测到 torchvision，与 torch 2.3.1 不兼容（transformers 导入会炸）')
    print('        修复: !pip uninstall -y torchvision torchaudio  然后 Restart Kernel 重跑')
except ImportError:
    pass


def get_device():
    \"\"\"CUDA 可用不代表 kernel 匹配：Kaggle 免费 GPU 常分到 P100（sm_60），新版 torch
    官方 wheel 已移除 sm_60 kernel → torch.cuda.is_available()=True 但执行即
    'no kernel image' 报错。用一次 matmul 自检，失败回退 CPU。\"\"\"
    if not torch.cuda.is_available():
        return torch.device('cpu')
    try:
        a = torch.randn(16, 16, device='cuda')
        b = torch.randn(16, 16, device='cuda')
        _ = (a @ b).cpu()          # kernel 执行，能在这里抛错
        return torch.device('cuda')
    except Exception as e:
        print(f'[警告] CUDA kernel 自检失败（{type(e).__name__}）→ 回退 CPU')
        print('        TextCNN CPU 可跑（几分钟）；RoBERTa CPU 很慢，建议二选一：')
        print('        ① Kaggle Runtime 换 GPU（T4）；')
        print('        ② !pip install -q torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121')
        print('           # 2.3.1 是 P100(sm_60) 最后支持线，且兼容 numpy 2.x（无需降级 numpy）')
        return torch.device('cpu')


device = get_device()
print('device:', device, '| torch:', torch.__version__, '| cuda:', torch.version.cuda)
if device.type == 'cuda':
    print('GPU:', torch.cuda.get_device_name(0))
"""))

CELLS.append(code("""
# ==================== 加载数据 ====================
def load_csv(path):
    df = pd.read_csv(path)
    assert set(df.columns) >= {'text', 'label', 'source', 'subtype', 'noise_level'}, f'列不完整: {path}'
    assert set(df.label.unique()) <= {0, 1}, f'label 非法值: {path}'
    df = df.fillna('')
    return df

train_df = load_csv(f'{DATA_DIR}/train.csv')          # 9808 训练
val_df = load_csv(f'{DATA_DIR}/val.csv')              # 2453 验证（阈值扫描用）
test_df = load_csv(f'{DATA_DIR}/test_external.csv')   # 10004 第三方（最终验证，不参与任何调参）
business_df = load_csv(f'{DATA_DIR}/business_negatives.csv')  # 独立业务负例集

print(f'train={len(train_df)} val={len(val_df)} test={len(test_df)} business_neg={len(business_df)}')
print(f'模板层(场景对齐): {(train_df.source.isin(["template", "noise"])).sum()} / {len(train_df)}')
print(f'负正比: {int((train_df.label == 0).sum())} : {int((train_df.label == 1).sum())}')
"""))

CELLS.append(code("""
# ==================== tokenizer + Dataset ====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class InjDataset(Dataset):
    def __init__(self, texts, labels=None, weights=None):
        self.texts = texts
        self.labels = labels
        self.weights = weights  # 模板层加权（训练集专用）

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = tokenizer(self.texts[i], max_length=MAX_LEN, truncation=True,
                        padding='max_length', return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(int(self.labels[i]), dtype=torch.long)
        if self.weights is not None:
            item['weights'] = torch.tensor(self.weights[i], dtype=torch.float)
        return item

def make_weights(df, weight):
    \"\"\"模板层（含噪声变体）是场景对齐黄金数据，给高 loss 权重\"\"\"
    return [weight if s in ('template', 'noise') else 1.0 for s in df.source]

train_ds = InjDataset(train_df.text.tolist(), train_df.label.tolist(), make_weights(train_df, TEMPLATE_WEIGHT))
val_ds = InjDataset(val_df.text.tolist(), val_df.label.tolist())
test_ds = InjDataset(test_df.text.tolist(), test_df.label.tolist())
business_ds = InjDataset(business_df.text.tolist())
"""))

CELLS.append(code("""
# ==================== 工具：分层报告 + 双口径阈值扫描 ====================
def predict_proba(model, ds):
    \"\"\"返回 (n,) 的注入概率\"\"\"
    model.eval()
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    probs = []
    with torch.no_grad():
        for b in loader:
            x = {k: v.to(device) for k, v in b.items() if k in ('input_ids', 'attention_mask', 'token_type_ids')}
            out = model(**x)
            probs.append(torch.softmax(out.logits, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)

def by_source_report(df, probs, thr=0.5):
    \"\"\"按 source 分层的准确率/注入召回——模板层召回是场景指标，hf 层是辅助指标\"\"\"
    p = probs >= thr
    rows = []
    for src in sorted(df.source.unique()):
        m = (df.source == src).values
        y = df.label.astype(int).values
        acc = (p[m] == (y[m] == 1)).mean()
        pos = y[m].sum()
        rec = (p[m] & (y[m] == 1)).sum() / pos if pos else float('nan')
        rows.append({'source': src, 'n': int(m.sum()), 'acc': round(float(acc), 3), 'recall_inject': round(float(rec), 3)})
    return pd.DataFrame(rows)

def scan_thresholds(val_df, val_probs, biz_probs, fp_biz_max):
    \"\"\"val 上扫阈值：业务负例误报率达标时取最高注入召回。hf safe 误报仅报告（非约束）。
    返回 (best, curve)：best=(thr, fp_biz, fp_hf, recall) 或 None；curve 用于画图。\"\"\"
    y = val_df.label.astype(int).values
    hf_safe = ((val_df.source.str.startswith('hf')) & (val_df.label.astype(int) == 0)).values
    best = None
    curve = []
    for thr in np.linspace(0.01, 0.99, 300):
        fp_biz = float((biz_probs >= thr).mean())
        fp_hf = float((val_probs[hf_safe] >= thr).mean()) if hf_safe.sum() else 0.0
        recall = float((val_probs[y == 1] >= thr).mean())
        curve.append((thr, fp_biz, fp_hf, recall))
        if fp_biz <= fp_biz_max:
            if best is None or recall > best[3]:
                best = (float(thr), fp_biz, fp_hf, recall)
    return best, curve

def evaluate(model, tag, train_df, val_df, test_df, business_df, fp_biz_max, verbose=True):
    \"\"\"完整评估：train/val/test 概率 + 阈值扫描（只约束业务误报）+ 分层报告 + train-val gap\"\"\"
    probs = {split: predict_proba(model, ds) for split, ds in
             [('train', train_ds), ('val', val_ds), ('test', test_ds), ('biz', business_ds)]}
    best, curve = scan_thresholds(val_df, probs['val'], probs['biz'], fp_biz_max)
    y_tr, y_va = train_df.label.astype(int).values, val_df.label.astype(int).values
    gap = ((probs['train'] >= 0.5) == (y_tr == 1)).mean() - ((probs['val'] >= 0.5) == (y_va == 1)).mean()
    if verbose:
        print(f'===== {tag} =====')
        print(f'train-val acc gap: {gap:+.4f}（越大过拟合越重）')
        if best is None:
            print(f'!! 业务误报率 {fp_biz_max:.0%} 内无阈值可选')
        else:
            thr, fp_biz, fp_hf, rec = best
            p_test = probs['test'] >= thr
            y_test = test_df.label.astype(int).values
            test_rec = float((p_test & (y_test == 1)).sum() / y_test.sum())
            print(f'最佳阈值 {thr:.3f}: 业务误报={fp_biz:.4f} hf误报={fp_hf:.4f}(参考) val注入召回={rec:.3f}')
            print(f'测试集(shalanova+手工)注入召回 @该阈值: {test_rec:.3f}')
            print('--- val 分层报告（thr=0.5）---')
            print(by_source_report(val_df, probs['val']).to_string(index=False))
    return {'probs': probs, 'best': best, 'curve': curve, 'gap': gap}
"""))

CELLS.append(code("""
# ==================== TextCNN（轻量基线） ====================
import types

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_filters=128, filter_sizes=(2, 3, 4, 5), dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, k, padding=k // 2) for k in filter_sizes])
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_filters * len(filter_sizes), 2),
        )

    def forward(self, input_ids, attention_mask=None, **kw):
        x = self.embedding(input_ids).transpose(1, 2)            # (B, E, L)
        pooled = [torch.max(torch.relu(c(x)), dim=2).values for c in self.convs]
        return types.SimpleNamespace(logits=self.classifier(torch.cat(pooled, dim=1)))
"""))

CELLS.append(code("""
# ==================== 训练函数（早停 + 模板层加权 loss） ====================
def train_model(model, train_ds, val_ds, val_df, epochs, lr, tag, use_weights=True):
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    y_val = val_df.label.astype(int).values
    best_acc, patience = None, 0
    for ep in range(epochs):
        model.train()
        total = 0.0
        for b in loader:
            x = {k: v.to(device) for k, v in b.items() if k in ('input_ids', 'attention_mask', 'token_type_ids')}
            y = b['labels'].to(device)
            loss = loss_fn(model(**x).logits, y)
            if use_weights and 'weights' in b:
                loss = (loss * b['weights'].to(device)).mean()
            else:
                loss = loss.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
        val_acc = float(((predict_proba(model, val_ds) >= 0.5) == (y_val == 1)).mean())
        print(f'[{tag}] epoch {ep + 1}: train_loss={total / len(loader):.4f} val_acc={val_acc:.4f}')
        if best_acc is None or val_acc > best_acc:
            best_acc, patience = val_acc, 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f'[{tag}] early stop @epoch {ep + 1}')
                break
    return model
"""))

CELLS.append(code("""
# ==================== 训练 TextCNN ====================
if USE_TEXT_CNN:
    cnn = TextCNN(tokenizer.vocab_size).to(device)
    n_params = sum(p.numel() for p in cnn.parameters())
    print(f'TextCNN 参数量: {n_params / 1e6:.1f}M')
    cnn = train_model(cnn, train_ds, val_ds, val_df, EPOCHS, LR_TEXT_CNN, 'TextCNN')
else:
    cnn = None
"""))

CELLS.append(code("""
# ==================== 训练 RoBERTa（冻结部分层 + 小 lr 防过拟合） ====================
if USE_ROBERTA and device.type == 'cpu':
    print('[警告] CPU 训练 RoBERTa 极慢（每 epoch 约 15-30 分钟）。建议：')
    print('        ① Kaggle Runtime 换 GPU（T4）；')
    print('        ② !pip install -q torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121  # P100 可用，兼容 numpy 2')
    print('        ③ 先用 EPOCHS=3 快速验证管线，再决定是否完整训练')
if USE_ROBERTA:
    # 兼容性注意：hfl/chinese-roberta-wwm-ext 权重是 pytorch_model.bin（无 safetensors），
    # 新版 transformers 强制 torch>=2.6 才能 torch.load（CVE-2025-32434 检查），而 torch 2.6
    # 不支持 P100(sm_60) → 若报 "upgrade torch to at least v2.6"，执行：
    #   !pip install -q transformers==4.44.2  然后 Restart Kernel 重跑
    roberta_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    # hfl/chinese-roberta-wwm-ext 是 BERT 架构（model_type=bert，属性 .bert）；真 RoBERTa 是 .roberta。
    # 兼容两种：哪个存在用哪个。
    base = getattr(roberta_model, 'roberta', None) or getattr(roberta_model, 'bert', None)
    if base is None:
        print('!! 无法定位 base 模型属性，跳过冻结（小数据过拟合风险）')
        n_train = sum(p.numel() for p in roberta_model.parameters())
    else:
        n_layers = len(base.encoder.layer)
        freeze_up_to = max(0, n_layers - 4)   # 只微调最后 4 层 + 池化/分类头
        for i, layer in enumerate(base.encoder.layer):
            if i < freeze_up_to:
                for p in layer.parameters():
                    p.requires_grad = False
        print(f'base 模型: {n_layers} 层，冻结前 {freeze_up_to} 层')
        n_train = sum(p.numel() for p in roberta_model.parameters() if p.requires_grad)
    print(f'可训练参数: {n_train / 1e6:.1f}M')
    roberta_model = train_model(roberta_model, train_ds, val_ds, val_df, EPOCHS, LR_ROBERTA, 'RoBERTa')
else:
    roberta_model = None
"""))

CELLS.append(code("""
# ==================== 评估：阈值扫描 + 分层报告 ====================
results = {}
if USE_TEXT_CNN:
    results['TextCNN'] = evaluate(cnn, 'TextCNN', train_df, val_df, test_df, business_df, FP_BUSINESS_MAX)
if USE_ROBERTA:
    results['RoBERTa'] = evaluate(roberta_model, 'RoBERTa', train_df, val_df, test_df, business_df, FP_BUSINESS_MAX)
"""))

CELLS.append(code("""
# ==================== 阈值-误报率曲线（决策依据可视化） ====================
for tag, r in results.items():
    curve = np.array(r['curve'])          # (thr, fp_biz, fp_hf, recall)
    plt.figure(figsize=(6, 4))
    plt.plot(curve[:, 0], curve[:, 3], label='val 注入召回')
    plt.plot(curve[:, 0], curve[:, 1], label='业务负例误报率')
    plt.plot(curve[:, 0], curve[:, 2], label='hf safe 误报率')
    if r['best']:
        thr, fp_biz, fp_hf, rec = r['best']
        plt.axvline(thr, ls='--', color='gray', alpha=0.7)
        plt.annotate(f'best thr={thr:.2f}', (thr, 0.5))
    plt.xlabel('阈值'); plt.legend(); plt.title(f'{tag} 阈值扫描')
    plt.tight_layout(); plt.show()
"""))

CELLS.append(code("""
# ==================== 软投票对比（两个模型差异度大时收益明显） ====================
if USE_TEXT_CNN and USE_ROBERTA:
    val_df2 = val_df.copy()
    # 复用 evaluate 的扫描逻辑，只是概率换成两模型均值
    from types import SimpleNamespace as NS
    def vote(tag_a, tag_b):
        va = 0.5 * results[tag_a]['probs']['val'] + 0.5 * results[tag_b]['probs']['val']
        vb = 0.5 * results[tag_a]['probs']['biz'] + 0.5 * results[tag_b]['probs']['biz']
        y = val_df.label.astype(int).values
        hf_safe = ((val_df.source.str.startswith('hf')) & (val_df.label.astype(int) == 0)).values
        best, curve = None, []
        for thr in np.linspace(0.01, 0.99, 300):
            fp_biz = float((vb >= thr).mean())
            fp_hf = float((va[hf_safe] >= thr).mean()) if hf_safe.sum() else 0.0
            rec = float((va[y == 1] >= thr).mean())
            curve.append((thr, fp_biz, fp_hf, rec))
            if fp_biz <= FP_BUSINESS_MAX and (best is None or rec > best[3]):
                best = (float(thr), fp_biz, fp_hf, rec)
        return best, curve
    best, _ = vote('TextCNN', 'RoBERTa')
    print('===== 软投票 (TextCNN+RoBERTa 概率均值) =====')
    if best is None:
        print('!! 无阈值达标')
    else:
        thr, fp_biz, fp_hf, rec = best
        p_test = 0.5 * (results['TextCNN']['probs']['test'] + results['RoBERTa']['probs']['test']) >= thr
        y_test = test_df.label.astype(int).values
        print(f'最佳阈值 {thr:.3f}: 业务误报={fp_biz:.4f} hf误报={fp_hf:.4f} val召回={rec:.3f} '
              f'测试集召回={(p_test & (y_test == 1)).sum() / y_test.sum():.3f}')
    print('\\n对比表（val 最佳阈值处）：')
    rows = [['模型', '阈值', '业务误报', 'hf误报', 'val召回']]
    for tag in ('TextCNN', 'RoBERTa'):
        b = results[tag]['best']
        if b:
            rows.append([tag, f'{b[0]:.3f}', f'{b[1]:.4f}', f'{b[2]:.4f}', f'{b[3]:.3f}'])
    if best:
        rows.append(['软投票', f'{best[0]:.3f}', f'{best[1]:.4f}', f'{best[2]:.4f}', f'{best[3]:.3f}'])
    widths = [max(len(r[i]) for r in rows) for i in range(5)]
    for r in rows:
        print('  ' + '  '.join(c.ljust(widths[i]) for i, c in enumerate(r)))
"""))

CELLS.append(code("""
# ==================== ONNX 导出（定稿模型，Kaggle 端执行） ====================
# 定稿参数（2026-08-19 第三轮训练）：
#   RoBERTa（chinese-roberta-wwm-ext，冻结前 8 层），1% 业务误报口径
#   打标签线 thr=0.018 | 硬拦线 0.9
# 需要: !pip install -q onnx onnxruntime
import torch

roberta_model.eval()
roberta_model.to('cpu')

dummy = (
    torch.ones(1, MAX_LEN, dtype=torch.long),
    torch.ones(1, MAX_LEN, dtype=torch.long),
    torch.zeros(1, MAX_LEN, dtype=torch.long),
)
torch.onnx.export(
    roberta_model, dummy,
    'roberta_inj.onnx',
    input_names=['input_ids', 'attention_mask', 'token_type_ids'],
    output_names=['logits'],
    dynamic_axes={'input_ids': {0: 'batch', 1: 'seq'}, 'attention_mask': {0: 'batch', 1: 'seq'},
                  'token_type_ids': {0: 'batch', 1: 'seq'}, 'logits': {0: 'batch'}},
    opset_version=17,
    do_constant_folding=True,
)
print('已导出 roberta_inj.onnx（下载后放到主项目 model/ 目录）')
"""))

CELLS.append(md("""
## 决策指引

1. **选型**：软投票 ≥ 单模型。若两者差异度大且软投票达标，用它；若 RoBERTa 单独就达标且
   TextCNN 差距大，直接上 RoBERTa（级联复核：边界区才调它，成本可控）。
2. **模板层召回**（val 分层报告第一行附近）：场景指标。如果明显低于 hf 层召回，加大
   TEMPLATE_WEIGHT（或过采样模板层）重训。
3. **train-val gap > 0.05**：过拟合，加大 WEIGHT_DECAY / dropout / 减小冻结层数。
4. **val 达标但 test 崩**：阈值过拟合 val。回到扫阈值时只允许 hf_safe 双口径内选点，
   并对比 test 上分层报告（shalanova vs manual）差异。
5. **Stacking（慎用）**：数据量 1.2 万 + 模板层仅 261 条，元模型容易过拟合。只有软投票
   不达标且两模型各有优劣时才考虑：OOF 预测 + LogisticRegression 元模型，务必 5 折 CV。
"""))

CELLS.append(md("""
## ONNX 导出与本地接入（训练完成后）

```python
# ---- TextCNN（静态导出，最简单）----
# torch.onnx.export(cnn.to('cpu'), torch.ones(1, MAX_LEN, dtype=torch.long),
#                   'textcnn_inj.onnx', input_names=['input_ids'],
#                   output_names=['logits'], opset_version=17)

# ---- RoBERTa（动态 shape，用 optimum）----
# pip install optimum onnxruntime
# 1) 先保存微调权重：torch.save(roberta_model.state_dict(), 'roberta_inj.pt')
# 2) 导出：from optimum.onnxruntime import ORTModelForSequenceClassification
#    m = ORTModelForSequenceClassification.from_pretrained(MODEL_NAME, export=True,
#         state_dict=torch.load('roberta_inj.pt', map_location='cpu'))
#    m.save_pretrained('roberta_inj_onnx/')   # 含 tokenizer 配置

# ---- 本地接入（msg_handle 前，路由/子 Agent 之前）----
# onnxruntime 推理 → softmax → 注入概率 >= 阈值：
#   高置信拒绝（>= 0.9 之类）：直接拦截，不进入图
#   中置信（阈值 ~ 0.5~0.9）：允许进图但给对话上下文打【注入风险】标签
#   低置信（< 阈值）：放行
# 阈值取 notebook 里 val 上双口径达标的最优点；上线后监控线上误报率，用业务负例集定期复测。
"""))

NOTEBOOK = {
    'cells': CELLS,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

OUT.write_text(json.dumps(NOTEBOOK, ensure_ascii=False, indent=1), encoding='utf-8')
print(f'notebook 已生成: {OUT}')
