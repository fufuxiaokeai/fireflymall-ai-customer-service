from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from SPO.state import FrontDeskSalespersonInputState, UserContext
from SPO.user import DeskSalespersonUserPersona
from Tools import product_process  # noqa: F401  导入以触发 @register_tool 注册，供 get_tools 使用
from Tools import get_tools
from Tools.log_settings import LogSetting
from Tools.middleware.tool_notice import ToolCallNoticeMiddleware
from langgraph._internal._constants import CONFIG_KEY_RUNTIME, CONFIG_KEY_STREAM, CONF
from langgraph.pregel.protocol import StreamProtocol
from load_config.config import config, read_sub_agent_prompt, ROOT_BASE_DIR_PATH

desk_salesperson_prompt = read_sub_agent_prompt(
    ROOT_BASE_DIR_PATH / 'agent/prompt/front_desk_salesperson_prompt.md'
)

logger = LogSetting.create(__name__)

desk_salesperson_tools = get_tools('product_process_agent')

desk_salesperson_model = init_chat_model(
    model=config['model']['name'],
    **config['model']['params']
)

desk_salesperson_agent = create_agent(
    model=desk_salesperson_model,
    tools=desk_salesperson_tools,
    system_prompt=desk_salesperson_prompt,
    middleware=[ToolCallNoticeMiddleware()],
)


def _user_persona_to_str(user_persona: DeskSalespersonUserPersona):
    if not user_persona:
        return ""
    field_labels = {
        'preference': '用户喜好',
        'hate': '用户的憎恶点',
        'dislike_goods_id': '用户不喜欢的商品ID列表',
        'dislike_brand': '用户不喜欢的品牌列表',
        'dislike_style': '用户不喜欢的风格',
        'like_goods_id': '用户喜欢的商品ID列表',
        'like_brand': '用户喜欢的品牌列表',
        'like_style': '用户喜欢的风格',
    }

    lines = []
    for key, label in field_labels.items():
        value = user_persona.get(key, None)
        if not value:
            continue
        if isinstance(value, list):
            value = '、'.join(value)
        lines.append(f'{label}: {value}')
    return '\n'.join(lines)


async def desk_salesperson(
        state: FrontDeskSalespersonInputState,
        runtime: Runtime
) -> dict:
    """
    用于与售前导购专门人员对话。
    参数：
        state: 专属于售前的输入状态
    """
    if runtime.execution_info.node_attempt >= 3:
        return {
            'assistant': {
                'source': 'desk_salesperson_service',
                'error': '本节点由于内部服务出现了故障，请联系技术人员进行维修'
            }
        }
    user_persona = state.get('user_person')
    thread_id = state.get('thread_id')
    if not thread_id:
        return {
            'assistant': {
                'source': 'desk_salesperson_service',
                'error': '不允许未指定thread_id字段'
            }
        }
    desk_msg = state.get('problem').strip()
    if not desk_msg:
        return {
            'assistant': {
                'source': 'desk_salesperson_service',
                'error': '不允许未指定问题进行提问'
            }
        }
    context = UserContext(user_id=thread_id)
    user_persona_str = _user_persona_to_str(user_persona)
    if file_url := state.get('file_url'):
        desk_msg += f"\n文件路径为：{','.join(file_url)}"
    if user_persona_str:
        desk_msg += f"\n注意点：\n{user_persona_str}"
    # 标记 __pregel_stream（值须为合法 StreamProtocol，DuplexStream 会读 .modes）并显式带上根
    # runtime：继承根 run 的 stream_writer，否则子 Agent 工具发出的自定义事件
    desk_salesperson_config = {CONF: {
        "thread_id": thread_id,
        CONFIG_KEY_STREAM: StreamProtocol(lambda _: None, {"custom"}),
        CONFIG_KEY_RUNTIME: runtime,
    }}
    desk_salesperson_result = await desk_salesperson_agent.ainvoke(
        {"messages": [HumanMessage(content=desk_msg)]},
        config=desk_salesperson_config,
        context=context
    )
    logger.info(f"对话结果为：{desk_salesperson_result}")
    return {
        'assistant': {
            'source': 'desk_salesperson_service',
            'route_answer': desk_salesperson_result["messages"][-1].content
        }
    }


if __name__ == '__main__':
    # while True:
    #     a = input("请输入对话内容：")
    #     result = desk_salesperson(a)
    #     print(result)
    pass
