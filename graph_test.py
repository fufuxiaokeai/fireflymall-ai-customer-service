from load_config.config import config, ROOT_BASE_DIR_PATH
import os

hf_conf = config['huggingface']

if hf_conf['mirror']:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

if hf_conf['download_dir']:
    os.environ['HF_HOME'] = str(ROOT_BASE_DIR_PATH / hf_conf['download_dir'])

from agent.main_agent import UserContext, graph
from langchain_core.messages import AIMessageChunk

import asyncio


async def graph_test(msg, graph_config, context):
    # v2 路线：graph.astream 默认 version='v1'，stream_mode='messages' 走经典 v1 handler
    # （on_llm_new_token 的 AIMessageChunk 流）。多模式输出为 (mode, data) 元组：
    #   messages -> (chunk, metadata)；custom -> 工具调用提示事件
    #
    # 思考文本过滤（标识+判断）：
    # - chat_node 的模型流式增量（AIMessageChunk）先缓冲
    # - 紧随其后的整条重放消息（chat_node 的包装消息）带 additional_kwargs['stream_visible']，
    #   用于结算缓冲：True → 输出（最终回答）；False → 丢弃（路由决策/工具调用的思考文本）
    # 实测 chunk 间隔中位数 0ms（批量突发），本就不存在人眼可感知的打字机，消息级输出无损失。
    print(f"👤 用户: {msg}")
    print("=" * 60)
    buf: list[str] = []
    async for mode, data in graph.astream(
        {'msg': msg},
        config=graph_config,
        context=context,
        stream_mode=['messages', 'custom'],
    ):
        if mode == 'messages':
            chunk, metadata = data
            if metadata.get('langgraph_node') != 'chat_node':
                continue
            if isinstance(chunk, AIMessageChunk):
                # 模型流式增量：先缓冲（是否展示取决于紧随其后的重放消息标识）
                if isinstance(chunk.content, str) and chunk.content:
                    buf.append(chunk.content)
            else:
                # chat_node 重放的整条包装消息：按标识结算上一缓冲
                visible = chunk.additional_kwargs.get('stream_visible')
                if visible is not None:
                    if visible:
                        print(''.join(buf), end='', flush=True)
                        print()
                    buf = []
        elif mode == 'custom':
            print()
            print(data)
    print()
    print("=" * 60)


if __name__ == '__main__':
    config = {"configurable": {"thread_id": "1"}}
    context = UserContext(user_id="1")
    while True:
        a = input('请输入文本: ')
        asyncio.run(graph_test(a, config, context))
