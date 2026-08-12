import datetime
import time
from typing import List, Literal, Optional

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from snowflake import SnowflakeGenerator
from sqlalchemy import select

from SPO.data_model import OrderInfo, UserOrderItem, OrderItemInfoJson, UserOrderJson, Goods, AfterSalesApply, \
    AfterSalesItem, AfterSalesLog, AfterSalesEvidence, AfterSalesRefund, AfterSalesReturn, AfterSalesExchange
from Tools.db import SessionLocal, AsyncSessionLocal
from Tools.registry import register_tool

_EPOCH = 1288834974657
_INSTANCE = 33
# 与Java分布式集群中的雪花配置一致
gen = SnowflakeGenerator(instance=_INSTANCE, epoch=_EPOCH)


def get_id():
    while True:
        uid = next(gen)
        if uid is not None:
            return str(uid)
        time.sleep(0.001)


@register_tool('after_sales_agent')
@tool
def search_after_sales_orders(
        runtime: ToolRuntime,
        # keyword: Optional[str] = None,keyword: 搜索关键词(可选, 必须是商品名称，不能是商品描述（暂不支持）)。
        page: int = 1,
        page_size: int = 10
):
    """
    售后订单搜索，用于搜索用户售后订单。
    参数：
        page: 页码，默认1。
        page_size: 每页数量，默认10。
    返回：
        售后订单列表。
    """
    user_id = runtime.context.user_id
    now_time = datetime.datetime.combine(datetime.date.today(), datetime.time.max)
    start_time = now_time - datetime.timedelta(days=15)
    session = SessionLocal()
    with session:
        order_id_status_list = [
            row for row in session.query(OrderInfo.orderId, OrderInfo.status).filter(
                OrderInfo.userId == user_id,
                OrderInfo.status == 1,
                OrderInfo.updateTime.between(start_time, now_time)
            ).order_by(OrderInfo.orderId.desc()).limit(page_size).offset(
                (page - 1) * page_size
            ).all()
        ]

        order_ids = [row[0] for row in order_id_status_list]

        items = session.query(UserOrderItem).filter(UserOrderItem.orderId.in_(order_ids)).all()
        goods_ids = list({item.goodsId for item in items})
        goods_map = {}
        if goods_ids:
            goods_rows = session.query(Goods.id, Goods.name).filter(
                Goods.id.in_(goods_ids)
            ).all()
            goods_map = {row.id: row.name for row in goods_rows}
        from collections import defaultdict

        grouped = defaultdict(list)
        status_map = {
            0: '未支付',
            1: '已支付',
            2: '已取消'
        }
        for item in items:
            grouped[item.orderId].append(
                OrderItemInfoJson(
                    goods_id=item.goodsId,
                    name=goods_map.get(item.goodsId, ''),
                    quantity=item.quantity,
                    goodsTotalAmount=item.goodsTotalAmount
                )
            )

        result = [
            UserOrderJson(order_id=oid, order_info=grouped.get(oid, []),
                          status=status_map.get(status, '未知状态')).model_dump_json()
            for oid, status in order_id_status_list
        ]
        return result


@register_tool('after_sales_agent')
@tool
async def apply_after_sales(
        runtime: ToolRuntime,
        apply_reason: str,
        order_id: int,
        apply_amount: int,
        file_url: str,
        file_name: str,
        correlation_goods_id_list: List[int],
        after_sales_type: Literal[1, 2, 3, 4] = 1,
        problem_type: Literal[1, 2, 3, 4, 5] = 1,
        evidence_type: Literal[1, 2, 3, 4] = 1,
        apply_num: int = 1,
        apply_desc: Optional[str] = None,
        problem_desc: Optional[str] = None,
) -> str:
    """
    售后申请，用于申请售后服务。
    参数：
        apply_reason: 售后申请原因。
        order_id: 订单ID。
        apply_amount: 售后申请金额(一般为对应申请售后的商品总金额)（单位：分）。
        file_url: 证据文件URL。
        file_name: 证据文件名。
        correlation_goods_id_list: 关联的商品ID列表 -> 一般为用户所需要售后的商品ID列表（可选）。
        after_sales_type: 售后类型，默认1类型（1：仅退款，2：退货退款，3：换货，4：其他售后）。
        problem_type: 问题类型，默认1类型（1：质量问题，2：发错货，3：缺少配件，4：外观破损，5：其他问题）。
        evidence_type: 证据类型，默认1类型（1：商品问题图，2：物流面单，3：开箱视频，4：其他证据）。
        apply_num: 申请售后数量，一般为订单对应的商品数量（默认为1）。
        apply_desc: 问题详细表述（一般用于售后类型为4），可选。
        problem_desc: 问题详细表述(不能超过500个字符)，可选。
    返回：
        售后申请写入的结果。
    """
    if after_sales_type == 4 and apply_desc is None:
        return "其他售后需要问题详细表述"
    if problem_desc and len(problem_desc) > 500:
        return "问题详细表述不能超过500个字符"
    user_id = runtime.context.user_id
    apply_no = get_id()
    try:
        async with AsyncSessionLocal() as session:
            # 查询关联商品信息
            stmt = select(Goods.id, Goods.spec, Goods.image, Goods.price).where(
                Goods.id.in_(correlation_goods_id_list)
            )
            result = await session.execute(stmt)
            goods_rows = result.all()

            # 校验商品是否存在
            if len(goods_rows) != len(correlation_goods_id_list):
                missing = set(correlation_goods_id_list) - {row.id for row in goods_rows}
                return f"商品ID不存在: {missing}"

            # 构建商品信息字典（价格单位：分）
            apply_item_info = {
                row.id: (row.spec, row.image, row.price) for row in goods_rows
            }

            # 单位转换：申请金额（分）→ 元
            apply_amount_yuan = apply_amount / 100.0

            # 1. 创建售后申请主记录（仅一条）
            apply = AfterSalesApply(
                apply_no=apply_no,
                user_id=user_id,
                order_id=order_id,
                apply_reason=apply_reason,
                after_sales_type=after_sales_type,
                apply_desc=apply_desc,
                apply_amount=apply_amount_yuan,
                apply_status=0,  # 待处理
            )
            session.add(apply)
            await session.flush()  # 获取自增 apply_id

            evidence = AfterSalesEvidence(
                apply_id=apply.apply_id,
                evidence_type=evidence_type,
                file_url=file_url,
                file_name=file_name,
            )
            session.add(evidence)

            # 2. 为每个售后商品创建明细记录
            for goods_id in correlation_goods_id_list:
                spec, image, price_fen = apply_item_info[goods_id]
                unit_price_yuan = price_fen / 100.0  # 单价分→元
                total_amount = apply_num * unit_price_yuan

                item = AfterSalesItem(
                    apply_id=apply.apply_id,
                    goods_id=goods_id,
                    spec=spec,
                    image=image,
                    apply_num=apply_num,
                    unit_price=unit_price_yuan,
                    total_amount=total_amount,
                    problem_type=problem_type,
                    problem_desc=problem_desc,
                )
                session.add(item)

            # 不为别的，只为帮商家填写一些非必要的
            if after_sales_type == 1:  # 仅退款：创建退款单
                refund_no = f"RF{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}{user_id}{apply.apply_id}"
                refund = AfterSalesRefund(
                    apply_id=apply.apply_id,
                    refund_no=refund_no,
                    refund_amount=apply_amount_yuan,
                    refund_type=1,  # 默认原路退回
                    pay_channel='alipay',  # 本应该从order表中获取，但是小应用所以直接谢伟alipay
                    refund_status=0,  # 待处理
                    operator_id=user_id,
                    remark='系统自动创建'
                )
                session.add(refund)

            elif after_sales_type == 2:  # 退货退款：创建退货单（待填地址）
                # 地址信息缺失，暂存空值，待商家审核时补全
                return_record = AfterSalesReturn(
                    apply_id=apply.apply_id,
                    receiver_name='必填',  # 待商家填写
                    receiver_phone='必填',
                    receiver_province='必填',
                    receiver_city='必填',
                    receiver_district='必填',
                    receiver_address='必填',
                    receive_status=0,  # 待签收
                    freight_bearer=0,  # 待确认
                )
                session.add(return_record)

            elif after_sales_type == 3:  # 换货：创建换货单（待填新商品信息）
                exchange = AfterSalesExchange(
                    apply_id=apply.apply_id,
                    new_sku_name='必填',  # 待商家审核后指定
                    new_spec_values=None,
                    exchange_num=apply_num if apply_num else 1,
                    receiver_name='必填',  # 收件地址待审核后填写
                    receiver_phone='必填',
                    receiver_address='必填',
                    sign_status=0,
                    price_diff=0.0,  # 差价待审核后计算
                )
                session.add(exchange)
                apply.exchange_status = 1  # 待换货

            # 3. 记录操作日志
            log = AfterSalesLog(
                apply_id=apply.apply_id,
                operator_type=4,  # 由系统进行操作
                operator_id=user_id,
                action_type='create',
                action_desc='用户提交售后申请',
                before_status=0,
                after_status=0,
            )
            session.add(log)

            await session.commit()
            return f"售后申请提交成功，申请编号：{apply_no}"

    except Exception as e:
        return f"在申请售后服务时，出现了错误：{e}"


@register_tool('after_sales_agent')
@tool
def notify_merchants(apply_id: int):
    """
    通知商家售后申请已提交
    :param apply_id: 售后申请ID
    :return: 通知结果
    """
    # 这里假设调用用其他商家的接口进行调用，并且成功申请
    return f"已通知商家售后申请已提交，申请编号：{apply_id}"


@register_tool('after_sales_agent')
@tool
def get_apply_id(
        runtime: ToolRuntime,
        apply_status: Literal[-1, 0, 1, 2, 3, 4, 5, 6] = -1
):
    """
    获取对应售后状态的售后申请部分信息列表
    参数：
        apply_status: 售后状态；-1表示全部状态（默认），0表示待审核，1表示商家同意，2表示商家拒绝，3表示用户取消，4表示处理中，5表示已完成，6表示已关闭
    """
    user_id = runtime.context.user_id
    session = SessionLocal()
    with session:
        apply_ids_sql = session.query(
            AfterSalesApply.apply_id,
            AfterSalesApply.order_id,
            AfterSalesApply.after_sales_type,
            AfterSalesApply.apply_reason,
            AfterSalesApply.apply_status
        ).filter(
            AfterSalesApply.user_id == user_id,
        )
        if apply_status != -1:
            apply_ids_sql = apply_ids_sql.filter(
                AfterSalesApply.apply_status == apply_status
            )
        apply_ids = apply_ids_sql.all()
        return [dict(item) for item in apply_ids]


@register_tool('after_sales_agent')
@tool
def get_apply_status(apply_id: int):
    """
    获取售后情况
    参数：
        apply_id: 售后ID
    返回：
        售后具体情况
    """
    session = SessionLocal()
    try:
        with session:
            status_info = session.query(
                AfterSalesApply.after_sales_type,
                AfterSalesApply.apply_status,
                AfterSalesApply.refund_status,
                AfterSalesApply.return_goods_status,
                AfterSalesApply.exchange_status
            ).filter(AfterSalesApply.apply_id == apply_id).first()
            if not status_info:
                return "售后申请不存在"
            apply_type, apply_status, refund_status, return_goods_status, exchange_status = status_info
            if apply_status == 5:
                return "售后申请已完成"
            elif apply_status == 2:
                action_desc = session.query(AfterSalesLog.action_desc).filter(
                    AfterSalesLog.apply_id == apply_id).first()
                reason = action_desc.action_desc if action_desc else '暂无拒绝原因记录'
                return f"售后被拒绝，原因为：{reason}"
            elif apply_status == 6:
                return f"售后申请已关闭"
            elif apply_status == 3:
                return f"改售后用户自己取消"

            return_goods_status_map = {
                0: "未退货",
                1: "待收货",
                2: "已收货",
                3: "收货异常",
            }
            refund_status_map = {
                0: "未退款",
                1: "退款中",
                2: '退款成功',
            }

            if apply_type == 1:
                if refund_status == 3:
                    fail_reason = session.query(AfterSalesRefund.fail_reason).filter(
                        AfterSalesRefund.apply_id == apply_id).first()
                    return f"售后退款失败，原因为：{fail_reason.fail_reason}"
                return f"售后退款状态：{refund_status_map.get(refund_status, '未知退款状态')}"
            if apply_type == 2:
                fail_reason = None
                if refund_status == 3:
                    fail_reason = session.query(AfterSalesRefund.fail_reason).filter(
                        AfterSalesRefund.apply_id == apply_id).first()
                    fail_reason = f"售后退款失败，原因为：{fail_reason.fail_reason}"
                return_goods_info = return_goods_status_map.get(return_goods_status, "未知退货状态")
                return f"{return_goods_info}" + (f"{fail_reason}" if fail_reason else '')

            exchange_status_map = {
                0: "未换货",
                1: "待换货",
                2: "已换货",
                3: "已签收",
            }

            if apply_type == 3:
                return f"售后换货状态：{exchange_status_map.get(exchange_status, '未知换货状态')}"
    except Exception as e:
        return f"在获取售后状态时，出现了错误：{e}"


if __name__ == '__main__':
    # sf = SnowflakeGenerator(instance=33, epoch=1288834974657)
    # apply_no = get_id(sf)
    # print(apply_no)
    # print(f"长度为{len(str(apply_no))}")
    pass
