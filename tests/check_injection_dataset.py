"""
注入数据集质量检查脚本（2026-08-19）

对 data/injection_dataset/ 三个 CSV 做训练前质量检查：
1. 分布检查：label×source×noise_level×subtype 交叉统计、8 注入子类型齐全性、负正比
2. 长度检查：按来源分组 min/max/中位数；异常行（<2 字 / >200 字）
3. 泄漏检查（全集互查，不做抽样——吸取 BM25 gold 采样偏置教训）：
   - train∪val vs test_external 精确文本交集 + 两侧 label 一致性
   - 手工 case 与构造模板的文本重叠量化（回归锚点泄漏，设计使然但要量化）
   - shalanova 与训练集文本重叠
   - 训练集内部 noise 行与 template 行精确重复（噪声失效检测）
   - 训练集内部同一文本跨 label（标注冲突）
4. 格式检查：空/空白/纯标点行、hf 层纯英文行、label∈{0,1}、CSV 列一致
5. 抽样目测：固定 seed=42，hf 正 10 / hf 负 10 / noise high 10 / shalanova 10

输出：stdout 汇总 + tests/data/check_report.txt
用法：python tests/check_injection_dataset.py
"""
import csv
import random
import re
import statistics
from collections import Counter
from pathlib import Path

# 复用构建脚本的模板定义（同目录导入；truststore 注入在模块顶层，无副作用）
from build_injection_dataset import MANUAL_CASES, INJECT_TEMPLATES, NORMAL_TEMPLATES

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data' / 'injection_dataset'
REPORT_PATH = Path(__file__).resolve().parent / 'data' / 'check_report.txt'
SEED = 42

LEN_SHORT, LEN_LONG = 2, 200
SAMPLE_SIZE = 10


def read_csv(path: Path) -> list[dict]:
    with open(path, 'r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def fmt_counter(c: Counter) -> str:
    return dict(sorted(c.items(), key=lambda kv: -kv[1])).__repr__()


def check_distribution(train, val, out):
    out.append('=' * 60)
    out.append(f'[1] 分布检查  train={len(train)} val={len(val)} (合计 {len(train) + len(val)})')
    all_rows = train + val
    labels = Counter(int(r['label']) for r in all_rows)
    out.append(f'  label: {fmt_counter(labels)}  (负正比 {labels[0] / labels[1]:.2f})')
    out.append(f'  source: {fmt_counter(Counter(r["source"] for r in all_rows))}')
    out.append(f'  noise_level: {fmt_counter(Counter(r["noise_level"] for r in all_rows))}')

    # 8 注入子类型齐全性（训练集 label=1 且来自构造层）
    expected = set(INJECT_TEMPLATES)
    actual = {r['subtype'] for r in all_rows if int(r['label']) == 1 and r['source'] in ('template', 'noise')}
    out.append(f'  注入子类型: 期望 {len(expected)} 个 -> 实际 {len(actual)}')
    missing = expected - actual
    if missing:
        out.append(f'    [警告] 缺失子类型: {sorted(missing)}')
    else:
        out.append(f'    全部齐全: {sorted(expected)}')
    subtype_counts = Counter(r['subtype'] for r in all_rows if int(r['label']) == 1)
    out.append(f'  注入样本 subtype 分布: {fmt_counter(subtype_counts)}')

    # 训练/验证的 label 与 source 交叉
    cross = Counter((r['source'], int(r['label'])) for r in all_rows)
    out.append(f'  source×label: {fmt_counter(cross)}')


def check_length(all_rows, out):
    out.append('=' * 60)
    out.append('[2] 长度检查（按来源分组，字符数）')
    by_src = {}
    for r in all_rows:
        by_src.setdefault(r['source'], []).append(len(r['text']))
    for src, lens in sorted(by_src.items(), key=lambda kv: -len(kv[1])):
        out.append(f'  {src:<12} n={len(lens):>5}  min={min(lens):>4}  median={statistics.median(lens):>6.1f}  max={max(lens):>4}')
    anomalies = [(r['source'], r['text']) for r in all_rows
                 if len(r['text']) < LEN_SHORT or len(r['text']) > LEN_LONG]
    out.append(f'  异常行（<{LEN_SHORT} 或 >{LEN_LONG} 字）: {len(anomalies)} 条')
    for src, t in anomalies[:10]:
        out.append(f'    [{src}] {t[:50]!r}')


def check_leakage(train, val, test, out):
    out.append('=' * 60)
    out.append('[3] 泄漏检查（全集互查，不抽样）')
    train_rows = train + val
    train_texts = {r['text'] for r in train_rows}
    train_label = {r['text']: int(r['label']) for r in train_rows}

    # 3a. 训练 vs 测试集
    ovl = [r for r in test if r['text'] in train_texts]
    conflict = [(r['text'], train_label[r['text']], int(r['label'])) for r in ovl
                if train_label[r['text']] != int(r['label'])]
    out.append(f'  train∪val ∩ test_external 精确重叠: {len(ovl)} / {len(test)} 条')
    for r in ovl[:10]:
        out.append(f'    [{r["source"]} label={r["label"]}] {r["text"][:60]!r}')
    if conflict:
        out.append(f'    [严重] 重叠且两侧 label 冲突 {len(conflict)} 条: {conflict[:5]}')
    else:
        out.append('    重叠行两侧 label 一致 ✓')

    # 3b. 手工 case vs 构造模板（回归锚点泄漏，设计使然）
    template_texts = set()
    for texts in INJECT_TEMPLATES.values():
        template_texts.update(texts)
    template_texts.update(NORMAL_TEMPLATES)
    manual_ovl = [c for c in MANUAL_CASES if c['text'] in template_texts]
    out.append(f'  手工 case 与构造模板精确重叠: {len(manual_ovl)} / {len(MANUAL_CASES)} 条 '
               f'（回归锚点泄漏，属设计使然；公平性由 shalanova 第三方集保证）')
    for c in manual_ovl:
        out.append(f'    [{c["subtype"]}] {c["text"][:60]!r}')

    # 3c. shalanova vs 训练集
    sha_ovl = [r for r in test if r['source'] == 'shalanova_zh' and r['text'] in train_texts]
    out.append(f'  shalanova 与训练集精确重叠: {len(sha_ovl)} 条')

    # 3d. noise 行 = template 行（噪声失效）
    template_rows = {r['text'] for r in train_rows if r['source'] == 'template'}
    noise_fail = [r for r in train_rows if r['source'] == 'noise' and r['text'] in template_rows]
    out.append(f'  噪声失效（noise 行与 template 行同文）: {len(noise_fail)} / '
               f'{sum(1 for r in train_rows if r["source"] == "noise")} 条')
    for r in noise_fail[:5]:
        out.append(f'    [level={r["noise_level"]}] {r["text"][:60]!r}')

    # 3e. 训练集内部标注冲突（同文跨 label）
    label_sets = {}
    for r in train_rows:
        label_sets.setdefault(r['text'], set()).add(int(r['label']))
    conflicts = [(t, sorted(ls)) for t, ls in label_sets.items() if len(ls) > 1]
    out.append(f'  训练集内部同文跨 label 冲突: {len(conflicts)} 条')
    for t, ls in conflicts[:5]:
        out.append(f'    {ls} {t[:60]!r}')


def check_format(train, val, test, out):
    out.append('=' * 60)
    out.append('[4] 格式检查')
    all_rows = train + val + test
    empty = [r for r in all_rows if not r['text'].strip()]
    punct_only = [r for r in all_rows if re.fullmatch(r'[^\w]+', r['text'].strip())]
    out.append(f'  空文本: {len(empty)} 条；纯标点/符号: {len(punct_only)} 条')
    for r in punct_only[:5]:
        out.append(f'    [{r["source"]}] {r["text"][:40]!r}')

    # hf/shalanova 层纯英文（ASCII）行——中文层混入英文算异常（构造层的 base64 是有意为之）
    ascii_rows = [r for r in all_rows if r['source'] in ('hf_zh_real', 'hf_zh_aug', 'shalanova_zh')
                  and re.fullmatch(r'[\x00-\x7f\s]+', r['text'])]
    out.append(f'  hf/shalanova 层纯英文行（应≈0）: {len(ascii_rows)} 条')
    for r in ascii_rows[:5]:
        out.append(f'    [{r["source"]}] {r["text"][:60]!r}')

    bad_label = [r for r in all_rows if r['label'] not in ('0', '1')]
    out.append(f'  label 非法值: {len(bad_label)} 条')
    missing_cols = [r for r in all_rows if set(r.keys()) != {'text', 'label', 'source', 'subtype', 'noise_level'}]
    out.append(f'  CSV 列不完整: {len(missing_cols)} 条')


def sample_eyeball(train, val, test, out):
    out.append('=' * 60)
    out.append(f'[5] 抽样目测（固定 seed={SEED}，每组最多 {SAMPLE_SIZE} 条，供人工看）')
    rng = random.Random(SEED)
    train_rows = train + val
    groups = [
        ('hf 注入样本', [r for r in train_rows if r['source'] in ('hf_zh_real', 'hf_zh_aug') and int(r['label']) == 1]),
        ('hf 正常样本', [r for r in train_rows if r['source'] in ('hf_zh_real', 'hf_zh_aug') and int(r['label']) == 0]),
        ('噪声 high 注入', [r for r in train_rows if r['source'] == 'noise' and r['noise_level'] == '2' and int(r['label']) == 1]),
        ('shalanova 第三方', [r for r in test if r['source'] == 'shalanova_zh']),
    ]
    for name, rows in groups:
        picked = rng.sample(rows, min(SAMPLE_SIZE, len(rows)))
        out.append(f'  --- {name}（抽样 {len(picked)} 条）---')
        for r in picked:
            out.append(f'    [{r["label"]}] {r["text"][:80]!r}')


def main():
    out = []
    for name in ('train.csv', 'val.csv', 'test_external.csv'):
        p = DATA_DIR / name
        if not p.exists():
            print(f'[错误] 缺少数据集文件: {p}，请先运行 python tests/build_injection_dataset.py')
            return
    train = read_csv(DATA_DIR / 'train.csv')
    val = read_csv(DATA_DIR / 'val.csv')
    test = read_csv(DATA_DIR / 'test_external.csv')

    check_distribution(train, val, out)
    check_length(train + val, out)
    check_leakage(train, val, test, out)
    check_format(train, val, test, out)
    sample_eyeball(train, val, test, out)

    report = '\n'.join(out)
    print(report)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report + '\n', encoding='utf-8')
    print(f'\n报告已写入: {REPORT_PATH}')


if __name__ == '__main__':
    main()
