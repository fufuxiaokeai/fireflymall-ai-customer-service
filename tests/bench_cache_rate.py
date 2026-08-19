# -*- coding: utf-8 -*-
"""
K-V 缓存命中率基准：记忆中间件三类模型调用的缓存行为
====================================================

背景（对应中间件成本讨论）：
  中间件有三个 LLM 调用（数学调参 / 片段切分 / 画像总结），每次都是
  [静态系统提示词][本次输入] 的全新会话。有人提出"携带上次对话历史"
  来提升 K-V 缓存命中率。本脚本用真实 API 用量数据回答两个问题：
    1. 当前写法（A 组）的缓存命中率到底是多少？
    2. 携带历史（B 组）是否真的更便宜？缓存命中率上升 ≠ 成本下降。

实验分组（每组 6 次连续调用，第 1 次冷启动，第 2~6 次看稳态）：
  A-math        基线：数学调参，主题每次变化（当前生产写法）
  A-math-same   控制：数学调参，主题固定不变
  A-split       基线：切分，增量消息窗口（当前生产写法）
  A-summary     基线：总结，增量新片段（当前生产写法）
  B-math        方案：携带历史 + 忽略指令（增长前缀）
  B-split       方案：携带历史 + 忽略指令（增长前缀）
  B-summary     方案：携带历史 + 忽略指令（增长前缀）
  D-chat        正对照：主对话风格——增长前缀自然复用（无忽略指令）

只测缓存命中率与计费，不依赖 RabbitMQ / Redis / SQLite / 业务场景。
提示词通过 AST 从源码原样提取（提示词变更后本脚本自动跟随，无需手改）。

命中率依据：DeepSeek 官方 API 每响应返回 usage.prompt_cache_hit_tokens /
prompt_cache_miss_tokens（计费字段：命中按折扣价、未命中按全价）。
单次命中率 = hit/(hit+miss)；汇总命中率 = 第 2~6 次累计 hit/(hit+miss)。

定价（deepseek-v4-flash 人民币，2026-08-17 调价前，每百万 tokens）：
  输入命中 ¥0.02 / 输入未命中 ¥1.0 / 输出 ¥2.0
  （8-17 起峰谷价：空闲 ¥0.05/1.5/4.5，高峰 9-12、14-18 点 ¥0.10/3.0/9.0；
   命中与未命中比值不变，本脚本结论不受影响）

环境：需在项目根目录（或 tests/）运行，.env 中 DEEPSEEK_API_KEY 已配置（load_dotenv 向上查找）。
用法：python tests/bench_cache_rate.py
原始数据落盘：tests/data/bench_cache_rate_result.json（不重跑也可重计价）
"""
import asyncio
import ast
import json
import os
import random
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # Windows 控制台中文

from dotenv import load_dotenv
load_dotenv()  # 从 .env 加载 DEEPSEEK_API_KEY

if not os.environ.get('DEEPSEEK_API_KEY'):
    raise SystemExit('[错误] 未找到 DEEPSEEK_API_KEY，请确认项目根目录 .env 已配置')

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

# ---------- 价格（¥/1M tokens，现行价；8-17 起峰谷价见 docstring） ----------
P_HIT = 0.02
P_MISS = 1.0
P_OUT = 2.0

# ---------- 1. 从源码原样提取三个生产提示词（不导入中间件模块，避免拉入 RabbitMQ/Redis/DB） ----------
def _extract_self_attr_prompt(path: str, attr: str, marker: str) -> str:
    src = open(path, encoding='utf-8').read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                    and target.value.id == 'self' and target.attr == attr
                    and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
                    and marker in node.value.value):
                return node.value.value
    raise SystemExit(f'[错误] 未从 {path} 提取到 self.{attr}（marker={marker}）')

MATH_PROMPT = _extract_self_attr_prompt('Tools/middleware/memory/time_memory.py', 'math_agent_prompt', '数学专家')
SPLIT_PROMPT = _extract_self_attr_prompt('Tools/middleware/memory/memory_rag.py', 'prompt', '切分区域')
SUMMARY_PROMPT = _extract_self_attr_prompt('Tools/middleware/memory/memory_rag.py', 'prompt', '总结专家')
# D-chat 用主对话提示词占位（导入 agent.main_agent 会拉起 RabbitMQ，故不导入；只影响 D 组绝对量，不影响命中率结论）
CHAT_PROMPT = '你是智能客服助手。请基于对话历史和记忆片段，礼貌、简洁地回答用户的问题。'

IGNORE_PREFIX = ('【缓存前缀】以下历史对话仅供缓存复用，与本次任务无关，'
                 '请完全忽略其内容，不要依据它们作答，只根据【本次任务】部分作答。')

# ---------- 2. 确定性合成内容（与真实中文对话 token 分布近似，可复现） ----------
_rng = random.Random(20260816)
_SENTENCES = [
    '你好，我想咨询一下这款产品的具体情况和价格。',
    '好的，请稍等，我帮您查一下最新的库存和优惠信息。',
    '这个型号有没有其他颜色可以选择？发货大概需要多久？',
    '目前有现货，下单后一般 2 到 3 天内可以发出，偏远地区可能会稍慢一些。',
    '如果收到货后发现质量问题，应该如何申请退换货？',
    '您可以在订单页面直接提交售后申请，我们会在一到两个工作日内审核。',
    '请问维修的时间大概要多久？费用大概在什么范围？',
    '常规问题一般一周内可以完成维修，具体费用需要检测后才能确定。',
    '发票是电子发票还是纸质发票？开票信息我需要提供哪些？',
    '默认开具电子发票，请在订单备注里提供抬头和税号，我们会尽快开具。',
    '我想修改一下收货地址，订单已经付款了，还能改吗？',
    '如果订单还未发货，可以帮您修改地址，已发货的订单需要联系快递处理。',
    '客服的上班时间是几点到几点？周末有人值班吗？',
    '我们的人工客服工作时间为每天 9 点到 21 点，周末也正常值班。',
    '你们有没有线下的实体门店？具体在什么位置？',
    '我们目前以线上销售为主，部分城市设有线下体验店，具体地址可以查询官网。',
]

def _make_window(k: int) -> str:
    """生成第 k 个消息窗口（增量切分的"本次输入"），格式与中间件 asplit_text 一致"""
    rng = random.Random(1000 + k)
    n = rng.randint(12, 16)
    lines = []
    for i in range(n):
        role = 'user' if i % 2 == 0 else 'ai'
        lines.append(f'对应的消息索引{i}，{role}:{rng.choice(_SENTENCES)}')
    return '\n'.join(lines)

WINDOWS = [_make_window(k) for k in range(1, 7)]
THEMES = ['商品咨询-手机', '物流-延迟', '退换货-质量', '发票-开票', '售后-维修', '价格-优惠']

# ---------- 3. 调用与用量采集 ----------
_model = init_chat_model(
    'deepseek-v4-flash',
    temperature=1.3,
    max_tokens=200,  # 仅本测试限制输出长度；不影响输入 token 流与缓存判定
    extra_body={'thinking': {'type': 'disabled'}},  # 与 config.yaml model.summary.kwargs 一致
)

async def _call(messages, k):
    t0 = time.perf_counter()
    resp = await _model.ainvoke(messages)
    dt = time.perf_counter() - t0
    meta = getattr(resp, 'response_metadata', {}) or {}
    usage = meta.get('token_usage') or meta.get('usage') or {}
    hit = int(usage.get('prompt_cache_hit_tokens', 0) or 0)
    miss = int(usage.get('prompt_cache_miss_tokens', 0) or 0)
    out = int(usage.get('completion_tokens', 0) or 0)
    if not usage:
        print(f'  [WARN] 第{k}次调用未取到 usage，response_metadata 可用键: {list(meta.keys())}')
    cost = (hit * P_HIT + miss * P_MISS + out * P_OUT) / 1e6
    sample = ' '.join(str(resp.content)[:60].split())
    return dict(k=k, hit=hit, miss=miss, out=out, cost=cost, dt=dt, sample=sample)

async def run_group(name, build_messages, n=6):
    print(f'\n===== {name} =====')
    rows = []
    for k in range(1, n + 1):
        row = await _call(build_messages(k), k)
        rows.append(row)
        rate = row['hit'] / (row['hit'] + row['miss'] + 1e-9) * 100
        print(f'  第{k}次 | 未命中 {row["miss"]:>6} | 命中 {row["hit"]:>6} | '
              f'命中率 {rate:6.1f}% | 计费 ¥{row["cost"]:.6f} | {row["dt"]:.1f}s | {row["sample"][:36]}')
    return rows

# ---------- 4. 各组请求构造 ----------
builders = {
    'A-math (基线:主题变化)': lambda k: [SystemMessage(MATH_PROMPT),
        HumanMessage(f'当前用户user_001的对话主题：{THEMES[k - 1]}')],
    'A-math-same (控制:主题不变)': lambda k: [SystemMessage(MATH_PROMPT),
        HumanMessage(f'当前用户user_001的对话主题：{THEMES[0]}')],
    'A-split (基线:增量窗口)': lambda k: [SystemMessage(SPLIT_PROMPT),
        HumanMessage('请将以下文本按照不同主题进行切分区域：\n' + WINDOWS[k - 1])],
    'A-summary (基线:新片段)': lambda k: [SystemMessage(SUMMARY_PROMPT),
        HumanMessage('目前的新增对话片段如下：\n' + WINDOWS[k - 1])],
    'B-math (方案:携带历史+忽略指令)': lambda k: [SystemMessage(MATH_PROMPT),
        HumanMessage(IGNORE_PREFIX + '\n\n' + '\n\n'.join(WINDOWS[:k - 1]) +
                     '\n\n[本次任务]\n' + f'当前用户user_001的对话主题：{THEMES[k - 1]}')],
    'B-split (方案:携带历史+忽略指令)': lambda k: [SystemMessage(SPLIT_PROMPT),
        HumanMessage(IGNORE_PREFIX + '\n\n' + '\n\n'.join(WINDOWS[:k - 1]) +
                     '\n\n[本次任务]\n请将以下文本按照不同主题进行切分区域：\n' + WINDOWS[k - 1])],
    'B-summary (方案:携带历史+忽略指令)': lambda k: [SystemMessage(SUMMARY_PROMPT),
        HumanMessage(IGNORE_PREFIX + '\n\n' + '\n\n'.join(WINDOWS[:k - 1]) +
                     '\n\n[本次任务]\n目前的新增对话片段如下：\n' + WINDOWS[k - 1])],
    'D-chat (正对照:增长前缀)': lambda k: [SystemMessage(CHAT_PROMPT),
        HumanMessage('\n\n'.join(WINDOWS[:k]) + '\n\n请用一句话回复以上对话中的最新问题。')],
}

async def main():
    print(f'模型: deepseek-v4-flash | 每组 6 次连续调用 | 命中 ¥{P_HIT}/M | 未命中 ¥{P_MISS}/M | 输出 ¥{P_OUT}/M')
    results = {}
    for name, build in builders.items():
        results[name] = await run_group(name, build)

    # 原始数据落盘：缓存跨运行保留，重跑冷启动不冷；落盘便于以后用新价格重计价
    with open(Path(__file__).resolve().parent / 'data' / 'bench_cache_rate_result.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ---------- 5. 汇总 ----------
    def input_cost(rows):
        return (sum(r['hit'] for r in rows) * P_HIT + sum(r['miss'] for r in rows) * P_MISS) / 1e6

    print('\n\n==================== 汇总（按组，¥） ====================')
    print(f'{"组别":<28}{"命中率(2~6次)":>14}{"总输入tok":>10}{"输入成本¥":>12}{"总计费¥":>12}')
    for name, rows in results.items():
        total_hit = sum(r['hit'] for r in rows)
        total_miss = sum(r['miss'] for r in rows)
        warm = rows[1:]
        warm_hit = sum(r['hit'] for r in warm)
        warm_rate = warm_hit / (warm_hit + sum(r['miss'] for r in warm) + 1e-9)
        total_cost = sum(r['cost'] for r in rows)
        print(f'{name:<28}{warm_rate * 100:>13.1f}%{total_hit + total_miss:>10}'
              f'{input_cost(rows):>12.6f}{total_cost:>12.6f}')

    print('\n==================== 关键对比（输入成本，纯缓存账） ====================')
    for pair in [('A-math (基线:主题变化)', 'B-math (方案:携带历史+忽略指令)'),
                 ('A-split (基线:增量窗口)', 'B-split (方案:携带历史+忽略指令)'),
                 ('A-summary (基线:新片段)', 'B-summary (方案:携带历史+忽略指令)')]:
        a, b = pair
        ratio = input_cost(results[b]) / input_cost(results[a])
        print(f'  {a}  ¥{input_cost(results[a]):.6f}')
        print(f'  {b}  ¥{input_cost(results[b]):.6f}   -> B/A = {ratio:.2f}x')

if __name__ == '__main__':
    asyncio.run(main())
