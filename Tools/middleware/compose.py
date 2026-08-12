"""
中间件洋葱链组合器 —— 手写版 create_agent 接线（脱离 create_agent 使用）

create_agent（langchain/agents/factory.py）会把 AgentMiddleware 拆成三类接线：
1. before/after 钩子 → 变成真正的图节点，串行执行，返回 dict 并入 state
   （before 按列表顺序执行，after 按逆序执行，与 factory.py:1600-1661 的连线顺序一致）
2. wrap_model_call / awrap_model_call → 组合成"洋葱链"包住真正的模型执行，
   列表第一个中间件 = 最外层；各层返回的 Command 按 内层→外层 顺序累积，
   节点应用时普通字段后写覆盖（即最外层中间件胜出，与官方 docstring 一致）
3. wrap_tool_call / awrap_tool_call → 传给 ToolNode 的 wrapper

本模块在手写 StateGraph 中手工复现这三类接线，供 chat_node / 工具节点调用。
注意：只支持异步（awrap_*）实现；若中间件只实现了同步版（wrap_*），
请先仿照 langchain/agents/middleware/_execution.py 转成异步再接入。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence

from langchain.agents.middleware import (
    AgentMiddleware,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage
from langgraph.types import Command

from langgraph.prebuilt.tool_node import ToolCallRequest

ModelHandler = Callable[[ModelRequest], Awaitable[ModelResponse]]
"""最内层真正执行模型的函数（对应 create_agent 的 _execute_model_async）"""

ModelWrapper = Callable[
    [ModelRequest, ModelHandler],
    Awaitable[ModelResponse | AIMessage | ExtendedModelResponse],
]
"""单个中间件的 awrap_model_call"""

ToolExecute = Callable[[ToolCallRequest], Awaitable[Any]]
ToolWrapper = Callable[[ToolCallRequest, ToolExecute], Awaitable[Any]]
"""单个中间件的 awrap_tool_call"""


@dataclass
class ComposedModelResult:
    """洋葱链执行结果：真正的模型响应 + 各层中间件累积的 Command（内层→外层顺序）"""

    model_response: ModelResponse
    commands: list[Command] = field(default_factory=list)


def _has_override(middleware: AgentMiddleware, hook_name: str) -> bool:
    """中间件是否覆写了指定钩子"""
    return getattr(middleware.__class__, hook_name) is not getattr(AgentMiddleware, hook_name)


def _normalize(result: ModelResponse | AIMessage | ExtendedModelResponse) -> ComposedModelResult:
    """把任意合法的中间件返回类型归一化为 ComposedModelResult"""
    if isinstance(result, ExtendedModelResponse):
        commands = [result.command] if result.command is not None else []
        return ComposedModelResult(model_response=result.model_response, commands=commands)
    if isinstance(result, ModelResponse):
        return ComposedModelResult(model_response=result)
    return ComposedModelResult(model_response=ModelResponse(result=[result]))


async def _run_hooks(
    middlewares: Sequence[AgentMiddleware],
    *,
    async_hook: str,
    sync_hook: str,
    state: dict[str, Any],
    runtime: Any,
    reverse: bool = False,
) -> dict[str, Any]:
    """串行执行钩子并合并返回的 state 更新（before 按列表顺序 / after 按逆序）"""
    updates: dict[str, Any] = {}
    ordered = reversed(middlewares) if reverse else middlewares
    for m in ordered:
        if _has_override(m, async_hook):
            update = await getattr(m, async_hook)(state, runtime)
        elif _has_override(m, sync_hook):
            update = getattr(m, sync_hook)(state, runtime)
        else:
            continue
        if update:
            updates.update(update)
    return updates


async def run_before_agent(
    middlewares: Sequence[AgentMiddleware], state: dict[str, Any], runtime: Any
) -> dict[str, Any]:
    """所有中间件的 before_agent 钩子（列表顺序），返回合并的 state 更新"""
    return await _run_hooks(
        middlewares, async_hook='abefore_agent', sync_hook='before_agent',
        state=state, runtime=runtime,
    )


async def run_after_agent(
    middlewares: Sequence[AgentMiddleware], state: dict[str, Any], runtime: Any
) -> dict[str, Any]:
    """所有中间件的 after_agent 钩子（逆序），返回合并的 state 更新"""
    return await _run_hooks(
        middlewares, async_hook='aafter_agent', sync_hook='after_agent',
        state=state, runtime=runtime, reverse=True,
    )


async def run_before_model(
    middlewares: Sequence[AgentMiddleware], state: dict[str, Any], runtime: Any
) -> dict[str, Any]:
    """所有中间件的 before_model 钩子（列表顺序），返回合并的 state 更新"""
    return await _run_hooks(
        middlewares, async_hook='abefore_model', sync_hook='before_model',
        state=state, runtime=runtime,
    )


async def run_after_model(
    middlewares: Sequence[AgentMiddleware], state: dict[str, Any], runtime: Any
) -> dict[str, Any]:
    """所有中间件的 after_model 钩子（逆序），返回合并的 state 更新"""
    return await _run_hooks(
        middlewares, async_hook='aafter_model', sync_hook='after_model',
        state=state, runtime=runtime, reverse=True,
    )


def chain_model_call_wrappers(
    middlewares: Sequence[AgentMiddleware],
) -> Callable[[ModelRequest, ModelHandler], Awaitable[ComposedModelResult]] | None:
    """洋葱链组合 awrap_model_call：列表第一个 = 最外层，最后一个 = 最内层。

    返回 None 表示没有中间件覆写了模型调用钩子（节点直接执行 handler 即可）。
    """
    wrappers: list[ModelWrapper] = []
    for m in middlewares:
        if _has_override(m, 'awrap_model_call'):
            wrappers.append(m.awrap_model_call)  # type: ignore[arg-type]
        elif _has_override(m, 'wrap_model_call'):
            raise NotImplementedError(
                f"{m.name} 只实现了同步 wrap_model_call；本组合器只支持异步实现，"
                "请参照 langchain/agents/middleware/_execution.py 转为异步"
            )

    if not wrappers:
        return None
    if len(wrappers) == 1:
        single = wrappers[0]

        async def single_chain(request: ModelRequest, handler: ModelHandler) -> ComposedModelResult:
            return _normalize(await single(request, handler))

        return single_chain

    def compose_two(outer: ModelWrapper, inner: ModelWrapper) -> ModelWrapper:
        async def composed(request: ModelRequest, handler: ModelHandler) -> Any:
            commands: list[Command] = []

            async def inner_handler(req: ModelRequest) -> ModelResponse:
                # 每次调用内层前清空，避免重试路径残留上一轮收集的 Command
                commands.clear()
                inner_result = await inner(req, handler)
                if isinstance(inner_result, ComposedModelResult):
                    # 内层本身也是组合层（多中间件折叠时）：先累积其 Command，再透传模型响应
                    commands.extend(inner_result.commands)
                    return inner_result.model_response
                if isinstance(inner_result, ExtendedModelResponse):
                    if inner_result.command is not None:
                        commands.append(inner_result.command)
                    return inner_result.model_response
                if isinstance(inner_result, ModelResponse):
                    return inner_result
                return ModelResponse(result=[inner_result])  # AIMessage 归一化

            outer_result = await outer(request, inner_handler)
            normalized = _normalize(outer_result)
            # 内层 Command 在前（先应用），外层 Command 在后（后写覆盖普通字段）
            return ComposedModelResult(
                model_response=normalized.model_response,
                commands=commands + normalized.commands,
            )

        return composed

    # 右折叠：wrappers[0] 最外层 → wrappers[-1] 最内层
    chain: ModelWrapper = wrappers[-1]
    for wrapper in reversed(wrappers[:-1]):
        chain = compose_two(wrapper, chain)
    return chain  # type: ignore[return-value]


def chain_tool_call_wrappers(
    middlewares: Sequence[AgentMiddleware],
) -> ToolWrapper | None:
    """洋葱链组合 awrap_tool_call：列表第一个 = 最外层（与 factory.py:633-694 一致）。

    返回 None 表示没有中间件覆写了工具调用钩子（ToolNode 不用传 wrapper）。
    """
    wrappers: list[ToolWrapper] = []
    for m in middlewares:
        if _has_override(m, 'awrap_tool_call'):
            wrappers.append(m.awrap_tool_call)  # type: ignore[arg-type]
        elif _has_override(m, 'wrap_tool_call'):
            raise NotImplementedError(
                f"{m.name} 只实现了同步 wrap_tool_call；本组合器只支持异步实现"
            )

    if not wrappers:
        return None
    if len(wrappers) == 1:
        return wrappers[0]

    def compose_two(outer: ToolWrapper, inner: ToolWrapper) -> ToolWrapper:
        async def composed(request: ToolCallRequest, execute: ToolExecute) -> Any:
            async def call_inner(req: ToolCallRequest) -> Any:
                return await inner(req, execute)

            return await outer(request, call_inner)

        return composed

    result: ToolWrapper = wrappers[-1]
    for wrapper in reversed(wrappers[:-1]):
        result = compose_two(wrapper, result)
    return result
