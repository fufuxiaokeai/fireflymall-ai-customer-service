"""
工具调用提示中间件

在每次工具执行前，向流式通道（stream_mode='custom'）发出一条用户可见的提示事件：
    AI客服调用了工具{工具名}

挂载方式：
- 主 Agent：加入 main_middlewares（已通过 chain_tool_call_wrappers 接入 ToolNode）
- 子 Agent：create_agent(..., middleware=[ToolCallNoticeMiddleware()])

无需改动任何工具函数本体。
"""
from typing import Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware

from langgraph.prebuilt.tool_node import ToolCallRequest

ToolExecute = Callable[[ToolCallRequest], Awaitable[Any]]


class ToolCallNoticeMiddleware(AgentMiddleware):
    """工具执行前发自定义事件的中间件（仅覆写 awrap_tool_call，不影响模型调用链）"""

    name = "ToolCallNotice"

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        execute: ToolExecute,
    ) -> Any:
        if runtime := request.runtime:
            runtime.stream_writer(f"AI客服调用了工具{request.tool_call.get('name', '')}")
        return await execute(request)
