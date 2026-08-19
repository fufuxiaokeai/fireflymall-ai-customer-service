from dataclasses import dataclass
from typing import TypedDict, Optional, Literal, NotRequired, List

from langchain.agents import AgentState
from langgraph.graph import MessagesState

from SPO.user import DeskSalespersonUserPersona


@dataclass
class UserContext:
    """运行时上下文：主 Agent 节点与记忆中间件都依赖 user_id"""
    user_id: str
    # 注入检测结果（图外检测传入，不进 checkpoint；chat_node 据此拼当轮旁白）
    injection: Optional[dict] = None   # {'level': 'suspect'|'warn', 'score': float}


class RouteResult(TypedDict):
    """路由翻译的结果"""
    # 消息来源：desk_salesperson_service: 售前; during_sale_service: 售中; after_sales_service: 售后
    source: Literal['desk_salesperson_service', 'during_sale_service', 'after_sales_service', 'artificial']
    # 路由消息结果-只用于没有发生任何有关节点处理等情况，例如像子Agent的工具发生问题的话，也会进行填写.
    route_answer: Optional[str]
    # 错误信息
    error: Optional[str]


class MainState(MessagesState):
    # 自定义系统提示词（首轮由 msg_handle 播种，之后由记忆中间件接管重写）
    system_prompt: Optional[str]
    # 子Agent返回的消息
    assistant: NotRequired[Optional[RouteResult]]
    # 记忆中间件的消息处理游标（未处理消息的起点，仅触发轮推进）
    new_msg_idx: NotRequired[int]
    # 当前用户画像（每轮由 msg_handle 从 store 刷新；中间件拼提示词时置于核心与片段之间）
    user_profile: NotRequired[Optional[str]]
    # 人工客服接管标记（按线程隔离存于图状态，由 artificial_node 置 True、msg_handle(human_reply) 置 False）
    manual_intervention: NotRequired[bool]


class BaseState(TypedDict):
    # 问题表述
    problem: str
    # 线程ID
    thread_id: str
    # 记忆清除
    is_begin: NotRequired[bool]


class FrontDeskSalespersonInputState(BaseState):
    file_url: Optional[List[str]]
    user_person: Optional[DeskSalespersonUserPersona]


class ArtificialInputState(BaseState):
    # 商户ID，模仿时将会取 mer-114514
    merchant_id: str
    # 是否由于内部异常导致需要人工客服接管
    is_internal_error: NotRequired[bool]


class NodeErrorState(MainState):
    type: str


class AfterSalesInputState(BaseState):
    file_url: Optional[str]


class InputState(TypedDict):
    msg: str
    system_prompt: Optional[str]
    # 人工客服回复入口：True 时 msg_handle 将 msg 作为人工客服消息注入并结束接管（供 FastAPI 调用）
    human_reply: NotRequired[bool]
    # 本轮用户上传的文件引用列表 [{'name': 展示名, 'path': 绝对路径}]，
    file_urls: NotRequired[list[dict]]


class OutputState(TypedDict):
    out_msg: str
    # 是否处于人工客服接管中（FastAPI 据此判断是否将消息直接路由给人工客服）
    manual_intervention: NotRequired[bool]


class RouteClassification(TypedDict):
    """路由前往"""
    # 要前往的路由名
    go_to: Literal['desk_salesperson_service', 'during_sale_service', 'after_sales_service']
    # 问题表述
    problem: str
    # 动态超时时间设置(单位：分钟)
    timeout: NotRequired[int | float]
    # 可选文件
    file_url: Optional[str]
    # 是否进行 “失忆” 处理
    is_begin: NotRequired[bool]


class SummaryMemoryState(AgentState):
    system_prompt: str
    new_msg_idx: int
