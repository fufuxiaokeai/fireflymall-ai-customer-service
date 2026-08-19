from langchain.agents import create_agent, AgentState
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from SPO.state import BaseState, UserContext
from Tools import during_sale_tool  # noqa: F401  导入以触发 @register_tool 注册，供 get_tools 使用
from Tools import get_tools
from Tools.log_settings import LogSetting
from Tools.middleware.tool_notice import ToolCallNoticeMiddleware
from langgraph._internal._constants import CONFIG_KEY_RUNTIME, CONFIG_KEY_STREAM, CONF
from langgraph.pregel.protocol import StreamProtocol
from load_config.config import config, read_sub_agent_prompt, ROOT_BASE_DIR_PATH

during_sale_system_prompt = read_sub_agent_prompt(
    ROOT_BASE_DIR_PATH / 'agent/prompt/during_sale_service_prompt.md'
)

logger = LogSetting.create(__name__)

during_sale_model = init_chat_model(
    model=config['model']['name'],
    **config['model']['params']
)

during_sale_agent = create_agent(
    model=during_sale_model,
    tools=get_tools('during_sale_agent'),
    system_prompt=during_sale_system_prompt,
    middleware=[ToolCallNoticeMiddleware()],
    context_schema=UserContext,
    state_schema=AgentState,
)


async def during_sale(
        state: BaseState,
        runtime: Runtime
):
    """
    用于与售中交易服务专门人员对话。
    参数：
        state：基本的消息输入状态
    """
    if runtime.execution_info.node_attempt >= 3:
        return {
            'assistant': {
                'source': 'during_sale_service',
                'error': '本节点由于内部服务出现了故障，请联系技术人员进行维修'
            }
        }

    thread_id = state.get('thread_id')
    if not thread_id:
        return {
            'assistant': {
                'source': 'during_sale_service',
                'error': '不允许未指定thread_id字段'
            }
        }

    # 判断是否清除记忆
    if state.get("is_begin", False):
        during_sale_checkpoint = getattr(during_sale_agent, 'checkpoint', None)
        if not during_sale_checkpoint:
            return {
                'assistant': {
                    'source': 'during_sale_service',
                    'error': "该节点暂时不支持清除记忆功能"
                }
            }
        try:
            if hasattr(during_sale_checkpoint, 'adelete_thread'):
                await during_sale_checkpoint.adelete_thread(thread_id)
                return {
                    'assistant': {
                        'source': 'during_sale_service',
                        'route_answer': "记忆清除完毕，可以随时开始新一轮任务"
                    }
                }
            elif hasattr(during_sale_checkpoint, 'delete_thread'):
                during_sale_checkpoint.delete_thread(thread_id)
                return {
                    'assistant': {
                        'source': 'during_sale_service',
                        'route_answer': "记忆清除完毕，可以随时开始新一轮任务"
                    }
                }
            else:
                logger.error(f"不支持清除记忆功能，thread_id为{thread_id}")
                return {
                    'assistant': {
                        'source': 'during_sale_service',
                        'error': "该节点暂时不支持清除记忆功能"
                    }
                }
        except Exception as e:
            logger.error(f"删除失败，原因为：{e}")
            return {
                'assistant': {
                    'source': 'during_sale_service',
                    'error': f"删除失败，原因为：{e}"
                }
            }

    desk_msg = state.get('problem').strip()
    if not desk_msg:
        return {
            'assistant': {
                'source': 'during_sale_service',
                'error': '问题表述不能为空'
            }
        }
    logger.info(f"对话内容为：{desk_msg}")

    context = UserContext(user_id=thread_id)
    # 标记 __pregel_stream（值须为合法 StreamProtocol，DuplexStream 会读 .modes）并显式带上根
    # runtime：继承根 run 的 stream_writer，否则子 Agent 工具发出的自定义事件
    during_sale_config = {CONF: {
        "thread_id": thread_id,
        CONFIG_KEY_STREAM: StreamProtocol(lambda _: None, {"custom"}),
        CONFIG_KEY_RUNTIME: runtime,
    }}
    during_sale_result = await during_sale_agent.ainvoke(
        {"messages": [HumanMessage(content=desk_msg)]},  # type: ignore
        config=during_sale_config,
        context=context
    )
    logger.info(f"对话结果为：{during_sale_result}")
    return {
        'assistant': {
            'source': 'during_sale_service',
            'route_answer': during_sale_result["messages"][-1].content
        }
    }
