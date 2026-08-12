"""
文件提示中间件：把本轮用户上传的文件引用注入模型输入。

设计要点：
- 只注入本轮：不写回 state、不进 checkpointer（因此不能用 before_model 钩子——
  其返回值会并入 state 落库），旧轮次的文件不会进入后续模型输入。
- 提示词内置被动分析约束：仅当用户明确要求分析文件内容时才调用文件分析工具。
"""
from typing import Callable, Awaitable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage, SystemMessage

from Tools.log_settings import LogSetting

logger = LogSetting.create(__name__)


class FileNoticeMiddleware(AgentMiddleware):
    name = 'FileNoticeMiddleware'

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        # 取最新一条 HumanMessage 的 metadata['files']
        files = None
        for msg in reversed(request.messages):
            if isinstance(msg, HumanMessage):
                files = msg.additional_kwargs.get('files')
                break
        if not files:
            return await handler(request)

        lines = [f"- {f.get('name', '文件')}（路径：{f['path']}）" for f in files]
        notice = SystemMessage(content=(
            "用户本轮上传了以下文件（路径仅服务端可用，不要向用户展示路径）：\n"
            + "\n".join(lines)
            + "\n仅当用户明确要求分析文件内容时，才调用文件分析工具并传入对应路径；"
              "否则只需在答复中确认已收到文件。若文件内容不合规，如实告知用户。"
        ))
        # 追加到消息队列末尾：不破坏 system+历史的前缀缓存
        new_request = request.override(messages=[*request.messages, notice])
        logger.info(f"已注入文件提示：{files}")
        return await handler(new_request)
