from typing import List, Optional, Callable

from langchain.agents.middleware import InterruptOnConfig
from langchain.agents.middleware.human_in_the_loop import DecisionType
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

_TOOL_REGISTRY: dict[str, list[BaseTool | Callable[[Callable | Runnable], BaseTool]]] = {}
_STRATEGY_INTERVENE: dict[str, bool | InterruptOnConfig] = {}


def register_tool(agent_name: str):
    def register(func):
        _TOOL_REGISTRY.setdefault(agent_name, []).append(func)
        return func
    return register


def interrupt(
        allow: bool = False,
        decisions: Optional[List[DecisionType]] = None,
        description: str = ""
):
    """
    装饰器工厂：标记工具函数在人机协同中的中断策略。

    Args:
        allow: 是否开启中断。False 表示自动批准（默认）。
        decisions: 允许的决策列表，如 ["approve", "reject"]。
                   如果不传且 allow=True，则默认开放全部三种决策。
        description: 中断时可选的描述信息。
    """

    def decorator(func):
        func_name = getattr(func, 'name', None) or getattr(func, '__name__', None)

        if not allow:
            # 明确标记为不中断
            _STRATEGY_INTERVENE[func_name] = False
        else:
            if decisions is not None:
                # 精细模式：只开放部分决策
                _STRATEGY_INTERVENE[func_name] = InterruptOnConfig(
                    allowed_decisions=decisions,
                    description=description
                )
            else:
                # 简单模式：开放全部决策
                _STRATEGY_INTERVENE[func_name] = True

        return func  # 保留原函数，不做包装
    return decorator


def get_tools(agent_name: str):
    return _TOOL_REGISTRY[agent_name].copy()


def get_strategy_interfere():
    return _STRATEGY_INTERVENE.copy()
