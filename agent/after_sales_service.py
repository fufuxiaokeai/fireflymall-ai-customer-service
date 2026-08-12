from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from SPO.state import AfterSalesInputState, UserContext
from Tools import after_sales_tool  # noqa: F401  导入以触发 @register_tool 注册，供 get_tools 使用
from Tools import get_tools
from Tools.log_settings import LogSetting
from Tools.middleware.tool_notice import ToolCallNoticeMiddleware
from langgraph._internal._constants import CONFIG_KEY_RUNTIME, CONFIG_KEY_STREAM, CONF
from langgraph.pregel.protocol import StreamProtocol
from load_config.config import config, read_sub_agent_prompt, ROOT_BASE_DIR_PATH

after_sales_system_prompt = read_sub_agent_prompt(
    ROOT_BASE_DIR_PATH / 'agent/prompt/after_sales_service_prompt.md'
)

logger = LogSetting.create(__name__)

after_sales_model = init_chat_model(
    model=config['model']['name'],
    **config['model']['params']
)

after_sales_agent = create_agent(
    model=after_sales_model,
    tools=get_tools('after_sales_agent'),
    system_prompt=after_sales_system_prompt,
    middleware=[ToolCallNoticeMiddleware()],
)


async def after_sales(
        state: AfterSalesInputState,
        runtime: Runtime
):
    """
    用于与售后服务专门人员对话。
    参数：
        state: 专门的售后服务输入消息
    """
    if runtime.execution_info.node_attempt >= 3:
        return {
            'assistant': {
                'source': 'after_sales_service',
                'error': '本节点由于内部服务出现了故障，请联系技术人员进行维修'
            }
        }
    desk_msg = state.get('problem').strip()
    if not desk_msg:
        return {
            'assistant': {
                'source': 'after_sales_service',
                'error': '不允许未指定问题进行提问'
            }
        }
    logger.info(f"对话内容为：{desk_msg}")
    thread_id = state.get('thread_id')
    if not thread_id:
        return {
            'assistant': {
                'source': 'after_sales_service',
                'error': '不允许未指定thread_id字段'
            }
        }
    context = UserContext(user_id=thread_id)
    file_url = state.get('file_url')
    if file_url:
        file_info = f"。同时用户发来一个证据文件，文件在：{file_url}"
        desk_msg += file_info
    # 标记 __pregel_stream（值须为合法 StreamProtocol，DuplexStream 会读 .modes）并显式带上根
    # runtime：继承根 run 的 stream_writer，否则子 Agent 工具发出的自定义事件
    after_sales_config = {CONF: {
        "thread_id": thread_id,
        CONFIG_KEY_STREAM: StreamProtocol(lambda _: None, {"custom"}),
        CONFIG_KEY_RUNTIME: runtime,
    }}
    after_sales_result = await after_sales_agent.ainvoke(
        {"messages": [HumanMessage(content=desk_msg)]},  # type: ignore
        config=after_sales_config,
        context=context
    )
    logger.info(f"对话结果为：{after_sales_result}")
    return {
        'assistant': {
            'source': 'after_sales_service',
            'route_answer': after_sales_result["messages"][-1].content
        }
    }