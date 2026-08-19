# -*- coding: utf-8 -*-
"""
端到端记忆对比基准：BalancedMultiDimensionMemory(BMDM) vs 官方 SummarizationMiddleware vs 无记忆基线
================================================================================

动机
----
BMDM 与官方记忆机制的核心差异在「长对话超过上下文预算后，还能否召回早期事实」，
因此采用「针-草垛（needle-in-haystack）」式多轮事实召回评测：

1. 在对话开头注入若干条互不相关的「事实针」（姓名、偏好、订单号、约束、爱好…）。
2. 中间插入大量无关「闲聊草垛」，把对话推到记忆触发阈值以上（触发总结/分片）。
3. 之后逐条提问，要求召回这些事实（含"很久前"与"较近期"混合）。
4. 用关键词命中判断是否答对，并统计每轮的输入/输出 token 与耗时。

对比三组配置（共用同一个底层模型）：
- baseline   : 无任何记忆中间件（朴素 create_agent）
- summarize  : 官方 langchain SummarizationMiddleware
- bmdm       : 项目自研 BalancedMultiDimensionMemory（走项目现有 graph，与线上完全一致）

运行前提
--------
必须在本项目根目录、且以下依赖可用时运行（与启动 main.py 相同的环境）：
- .env 里的 DEEPSEEK_API_KEY / DASHSCOPE_API_KEY 已配置
- Redis(本地 6379)、RabbitMQ(192.168.152.135)、sqlite-vec 可用（bmdm 路径需要）
- python bench_memory_recall.py

注意：
- bmdm 的记忆触发阈值来自 config.yaml 的 model.summary（fraction 0.8，max_input_tokens=12800，
  即约 10240 token 才触发分片/检索）。为让记忆真正生效，FILLER_TURNS 需足够大把对话推到阈值。
  想快速验证可临时把 config.yaml 的 pattern 改成 'messages' + trigger_threshold 改小。
- summarize 的触发阈值在下方 SUMMARIZE_KWARGS 里按消息数配置，与 bmdm 是两套阈值，
  这里只求「各自的默认合理配置」，不做参数对齐（对齐需另行做参数扫描）。
"""

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 仓库根，保证 load_config/agent 等可导入

# ---- 与 run.py 一致的环境准备（须在任何 langchain/hf 导入前） ----
from load_config.config import config, ROOT_BASE_DIR_PATH

hf_conf = config['huggingface']
if hf_conf['mirror']:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
if hf_conf['download_dir']:
    os.environ['HF_HOME'] = str(ROOT_BASE_DIR_PATH / hf_conf['download_dir'])

from langchain.agents import create_agent
from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage

# BMDM：走项目现有 graph（已完整接好 store / checkpointer / 中间件 / context）
from agent.main_agent import graph
from SPO.state import UserContext

# ======================================================================
# 探针数据：fact 为「早先注入的事实针」，questions 为「之后要召回的事实」。
# 每道题 expected 是多个可命中关键词（命中其一即算对）。
# ======================================================================
FACTS = [
    ("我的名字叫林晓雨", ["林晓雨"]),
    ("我最喜欢的咖啡是埃塞俄比亚耶加雪菲", ["耶加雪菲", "埃塞俄比亚"]),
    ("我的订单号是 ORD-88421", ["ORD-88421", "88421"]),
    ("我对花生严重过敏", ["花生", "过敏"]),
    ("我周末喜欢去爬山", ["爬山"]),
    ("我的收货地址是杭州市西湖区文三路 8 号", ["杭州", "文三路"]),
]

RECALL_QUESTIONS = [
    # (问题, 期望关键词, 属于第几条 fact)
    ("我之前告诉过你我叫什么名字？", ["林晓雨"], 0),
    ("我还记得我喜欢的咖啡是哪一种？", ["耶加雪菲", "埃塞俄比亚"], 1),
    ("帮我查一下我之前给你的订单号。", ["ORD-88421", "88421"], 2),
    ("我有什么饮食禁忌来着？", ["花生", "过敏"], 3),
    ("我上次说周末一般干什么？", ["爬山"], 4),
    ("我的收货地址你还记得吗？", ["杭州", "文三路"], 5),
]

FILLER_TURNS = 40  # 草垛轮数：调大以真正触发 bmdm 的记忆分片/检索

# 官方 SummarizationMiddleware 配置（按消息数触发，避免依赖模型 profile）
SUMMARIZE_KWARGS = dict(
    trigger=("messages", 30),   # 超过 30 条消息即总结
    keep=("messages", 10),      # 总结后保留最近 10 条
)

BASELINE_SYSTEM_PROMPT = "你是一个记忆力很好的助手，请忠实记住用户说过的信息，并在被问到时不编造、如实回答。"


def build_model():
    return init_chat_model(model=config['model']['name'], **config['model']['params'])


def _last_ai_text(messages):
    for m in reversed(messages):
        if isinstance(m, AIMessage) and isinstance(m.content, str) and m.content:
            return m.content
    return ""


def _score(answer: str, keywords: list[str]) -> bool:
    return any(k in answer for k in keywords)


# ---- 三种被评测对象的「一问一答」接口，统一签名：answer = await run(turn_idx) ----

async def run_baseline(agent, session: list):
    """朴素 create_agent，历史直接压进 messages。"""
    res = await agent.ainvoke({"messages": list(session)})
    return _last_ai_text(res["messages"])


async def run_agent_with_history(agent, session: list):
    res = await agent.ainvoke({"messages": list(session)})
    return _last_ai_text(res["messages"])


async def run_bmdm(msg: str, thread_id: str):
    """走项目现有 graph：内部已有记忆中间件、checkpointer 与 store。"""
    res = await graph.ainvoke(
        {"msg": msg},
        config={"configurable": {"thread_id": thread_id}},
        context=UserContext(user_id=thread_id),
    )
    return res.get("out_msg", "")


def _usage_tokens(usage_metadata) -> tuple[int, int]:
    if not usage_metadata:
        return (0, 0)
    return (
        usage_metadata.get("input_tokens") or usage_metadata.get("prompt_tokens") or 0,
        usage_metadata.get("output_tokens") or usage_metadata.get("completion_tokens") or 0,
    )


def build_dialogue() -> list[tuple[str, str]]:
    """构造完整对话脚本：事实注入 -> 草垛闲聊 -> 召回提问。"""
    turns = []
    for fact, _ in FACTS:
        turns.append(("user", fact))
        turns.append(("assistant", "好的，我已经记住了。"))
    for i in range(FILLER_TURNS):
        turns.append(("user", f"顺便聊聊，你知道今天天气怎么样吗？这是第 {i} 次问你无关的事。"))
        turns.append(("assistant", "抱歉，我无法获取实时天气，不过很高兴和你聊天。"))
    for q, _, _ in RECALL_QUESTIONS:
        turns.append(("user", q))
    return turns


async def run_config(name: str) -> dict:
    turns = build_dialogue()
    thread_id = f"bench-{name}"
    model = build_model()

    if name == "bmdm":
        answers = {}
        in_tok = out_tok = 0
        t0 = time.perf_counter()
        for role, text in turns:
            if role != "user":
                continue
            ans = await run_bmdm(text, thread_id)
            answers[text] = ans
        dt = time.perf_counter() - t0
        return {"answers": answers, "in_tok": in_tok, "out_tok": out_tok, "secs": dt}

    # baseline / summarize 共用 create_agent 骨架
    middleware = []
    if name == "summarize":
        middleware = [SummarizationMiddleware(model=model, **SUMMARIZE_KWARGS)]
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt=BASELINE_SYSTEM_PROMPT,
        middleware=middleware,
    )

    session: list = []
    answers = {}
    in_tok = out_tok = 0
    t0 = time.perf_counter()
    for role, text in turns:
        if role == "user":
            session.append(HumanMessage(content=text))
        else:
            session.append(AIMessage(content=text))
        if role != "user":
            continue
        res = await agent.ainvoke({"messages": list(session)})
        ai_text = _last_ai_text(res["messages"])
        answers[text] = ai_text
        session.append(AIMessage(content=ai_text))
        um = res["messages"][-1].usage_metadata
        i, o = _usage_tokens(um)
        in_tok += i
        out_tok += o
    dt = time.perf_counter() - t0
    return {"answers": answers, "in_tok": in_tok, "out_tok": out_tok, "secs": dt}


async def main():
    print("=" * 70)
    print("记忆召回对比：baseline(无记忆) vs summarize(官方) vs bmdm(自研)")
    print(f"事实针 {len(FACTS)} 条 / 草垛 {FILLER_TURNS} 轮 / 召回题 {len(RECALL_QUESTIONS)} 道")
    print("=" * 70)

    results = {}
    for name in ["baseline", "summarize", "bmdm"]:
        print(f"\n>>> 运行 {name} ...")
        results[name] = await run_config(name)

    print("\n" + "=" * 70)
    print("召回命中结果（每道题答对打 √）")
    print("=" * 70)
    header = f"{'问题':<26s}" + "".join(f"{n:>12s}" for n in results)
    print(header)
    for q, keywords, _ in RECALL_QUESTIONS:
        row = f"{q[:24]:<26s}"
        for name in results:
            ans = results[name]["answers"].get(q, "")
            ok = _score(ans, keywords)
            row += f"{('√' if ok else '✗'):>12s}"
        print(row)

    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    print(f"{'指标':<24s}" + "".join(f"{n:>12s}" for n in results))
    for metric in ["in_tok", "out_tok", "secs"]:
        row = f"{metric:<24s}"
        for name in results:
            v = results[name][metric]
            row += f"{v:>12.1f}" if isinstance(v, float) else f"{v:>12d}"
        print(row)
    # 召回率
    row = f"{'recall':<24s}"
    for name in results:
        hit = sum(_score(results[name]["answers"].get(q, ""), kw) for q, kw, _ in RECALL_QUESTIONS)
        row += f"{hit}/{len(RECALL_QUESTIONS):>9s}"
    print(row)


if __name__ == "__main__":
    asyncio.run(main())
