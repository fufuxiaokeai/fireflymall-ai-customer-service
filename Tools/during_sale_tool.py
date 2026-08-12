import json
from asyncio import Lock
from datetime import datetime, timedelta
from typing import Literal, Optional, List

import redis
from aio_pika import DeliveryMode, ExchangeType, Message, connect_robust
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from SPO.data_model import OrderInfo, UserOrderItem, Goods, OrderItemInfoJson, UserOrderJson, ShoppingCart
from Tools.after_sales_tool import get_id
from Tools.db import SessionLocal
from Tools.invoice_tool import ItemsDict, InvoiceTool
from Tools.registry import register_tool
from load_config.config import config, ROOT_BASE_DIR_PATH

_redis_config = config.get('redis')
_rabbitmq_config = config.get('AMQP').get('rabbitmq')
if not _redis_config or not _rabbitmq_config:
    raise ValueError("Redis配置或AMQP配置未找到, 请先配置对应的Redis和AMQP配置")

_pool = redis.ConnectionPool(
    host=_redis_config.get('host', 'localhost'),
    password=_redis_config.get('password', None),
    port=_redis_config.get('port', 6379),
    db=_redis_config.get('db', 0),
    max_connections=_redis_config.get('max_connections', 20),
    decode_responses=True,
)
redis_conn = redis.Redis(connection_pool=_pool)

# aio-pika 参数字典（与中间件 memory_rag 的连接写法保持一致）
_rabbit_params = dict(
    host=_rabbitmq_config.get('host', '192.168.152.135'),
    virtualhost=_rabbitmq_config.get('virtual_host', '/'),
    login=_rabbitmq_config.get('username', 'fufu'),
    password=_rabbitmq_config.get('password', '123456'),
    heartbeat=_rabbitmq_config.get('heartbeat', 600),
    # connection_attempts × retry_delay 反用为单次连接超时（初始连接失败为 fail-fast）
    timeout=_rabbitmq_config.get('connection_attempts', 3) * _rabbitmq_config.get('retry_delay', 5),
    # retry_delay 对应断线后的自动重连间隔
    reconnect_interval=_rabbitmq_config.get('retry_delay', 5),
)
rabbit_conn = None
channel = None
_rabbit_lock = Lock()


async def _ensure_rabbit_ready():
    """延迟建立 RabbitMQ 连接并声明交换机/队列（aio-pika 自动重连）：
    不在 import 期连接，避免 RabbitMQ 不可达时整个应用启动失败"""
    global rabbit_conn, channel
    async with _rabbit_lock:
        if channel is not None:
            return channel
        rabbit_conn = await connect_robust(**_rabbit_params)
        channel = await rabbit_conn.channel()
        exchange = await channel.declare_exchange(
            'order_exchange',
            ExchangeType.X_DELAYED_MESSAGE,  # 延迟消息
            durable=True,
            arguments={'x-delayed-type': 'direct'},  # 原direct逻辑
        )
        queue = await channel.declare_queue('orders', durable=True)
        await queue.bind(
            queue='orders',
            exchange='order_exchange',
            routing_key='orders'
        )
        return channel


async def _publish_delayed_order_cancel(order_id: int, delay_seconds: int = 30 * 60):
    """发布延迟取消订单消息（x-delay 头控制延迟时长）"""
    chan = await _ensure_rabbit_ready()
    message = Message(
        body=str(order_id).encode('utf-8'),
        delivery_mode=DeliveryMode.PERSISTENT,
        content_type='text/plain',
        headers={'x-delay': delay_seconds * 1000},
    )
    await chan.default_exchange.publish(
        message,
        routing_key='orders',  # 与队列绑定 key 保持一致（direct 交换要求精确匹配）
    )


@register_tool('during_sale_agent')
@tool
def gen_invoice(
        runtime: ToolRuntime,
        order_id: int
):
    """
    获取发票（PDF格式）
    参数：
        order_id: 指定的订单ID
    """
    user_id = runtime.context.user_id
    session = SessionLocal()
    try:
        with session:
            results = session.query(
                Goods.name,
                (Goods.price / 100).label('price'),
                UserOrderItem.quantity,
                UserOrderItem.goodsTotalAmount.label('total_amount')
            ).join(
                Goods, UserOrderItem.goodsId == Goods.id
            ).filter(
                UserOrderItem.orderId == order_id,
                UserOrderItem.userId == user_id
            ).all()

            items = [ItemsDict(name=row.name, price=row.price, quantity=row.quantity, total_amount=row.total_amount)
                     for row in results]

            invoice = InvoiceTool(items)
            invoice_path = str(ROOT_BASE_DIR_PATH / f"data/invoice/PDF/{user_id}-{order_id}.pdf")
            invoice.generate_invoice(invoice_path)
            return f"发票已生成，文件路径：{invoice_path}"
    except Exception as e:
        return f"发票生成失败：{str(e)}"


@register_tool('during_sale_agent')
@tool
def get_order_info(
        runtime: ToolRuntime
):
    """
    获取已经支付了的订单部分信息，用于配合获取发票的使用。
    不可用于订单状态查询
    返回：
        订单的部分信息
    """
    user_id = runtime.context.user_id
    session = SessionLocal()
    with session:
        order_ids = session.query(
            OrderInfo.orderId
        ).filter(
            OrderInfo.userId == user_id,
            OrderInfo.payTime.isnot(None)
        )

        items = get_order_items_info(order_ids, session)

        result = [
            UserOrderJson(order_id=oid, order_info=items.get(oid, [])).model_dump_json()
            for oid in order_ids
        ]
        return result


def get_order_items_info(order_ids, session):
    user_order_items = session.query(UserOrderItem).filter(UserOrderItem.orderId.in_(order_ids)).all()
    goods_ids = list({item.goodsId for item in user_order_items})
    goods_map = {}
    if goods_ids:
        goods_rows = session.query(Goods.id, Goods.name).filter(
            Goods.id.in_(goods_ids)
        ).all()
        goods_map = {row.id: row.name for row in goods_rows}
    from collections import defaultdict
    grouped = defaultdict(list)
    for item in user_order_items:
        grouped[item.orderId].append(
            OrderItemInfoJson(
                goods_id=item.goodsId,
                name=goods_map.get(item.goodsId, ''),
                quantity=item.quantity,
                goodsTotalAmount=item.goodsTotalAmount
            )
        )
    return grouped


@register_tool('during_sale_agent')
@tool
def cancel_order(
        runtime: ToolRuntime,
        order_id: int
):
    """
    取消订单
    参数：
        order_id: 订单ID
    返回：取消是否成功的消息
    """
    user_id = runtime.context.user_id
    session = SessionLocal()
    try:
        with session:
            session.query(OrderInfo).filter(
                OrderInfo.orderId == order_id,
                OrderInfo.userId == user_id
            ).update({'status': 2})
            session.commit()
            return f"订单{order_id}已取消"
    except Exception as e:
        return f"取消订单失败：{str(e)}"


@register_tool('during_sale_agent')
@tool
def get_order_status(
        runtime: ToolRuntime,
        status: Literal[0, 1, 2] | None = None,
        time_from_the_create_time: Optional[int] = None
):
    """
    获取订单状态和基本信息
    参数：
        status: 需要筛选的订单状态。 0-未支付，1-已支付，2-已取消，若不填则表示忽略状态，获取全部订单状态（默认）
        time_from_the_create_time: 筛选当前时间距离订单创建时间的天数。不填则表示忽略时间（默认不填）
    """
    user_id = runtime.context.user_id
    session = SessionLocal()
    try:
        with session:
            sql = session.query(OrderInfo.orderId, OrderInfo.status)
            if status is not None:
                sql = sql.filter(OrderInfo.status == status)
            if time_from_the_create_time is not None:
                now_time = datetime.now()
                start_time = now_time - timedelta(days=time_from_the_create_time)
                sql = sql.filter(OrderInfo.createTime.between(start_time, now_time))
            order_id_status_list = sql.filter(OrderInfo.userId == user_id).all()

            order_ids = [row[0] for row in order_id_status_list]

            items = get_order_items_info(order_ids, session)

            status_map = {0: '未支付', 1: '已支付', 2: '已取消'}

            result = [
                UserOrderJson(order_id=oid, order_info=items.get(oid, []),
                              status=status_map.get(status, '未知状态')).model_dump_json()
                for oid, status in order_id_status_list
            ]
            return result
    except Exception as e:
        return f"查询订单时出现错误：{str(e)}"


@register_tool('during_sale_agent')
@tool
async def create_order(
        runtime: ToolRuntime,
        receipt_address: str,
        remark: Optional[str] = None
):
    """
    根据购物车中的商品信息创建订单
    参数：
        receipt_address: 收货地址
        remake: 订单备注(可选)
    """
    user_id = runtime.context.user_id
    raw_cart_items = redis_conn.hgetall(f"cart:{user_id}")
    if not raw_cart_items:
        return "该用户还没订购商品"

    cart_items = {int(k): int(v) for k, v in raw_cart_items.items()}
    cart_items_id = list(cart_items.keys())
    order_id = get_id()

    session = SessionLocal()
    try:
        # 查询商品价格（不提交，统一在最后 commit）
        goods_rows = session.query(
            Goods.id, (Goods.price / 100.0).label('price')
        ).filter(Goods.id.in_(cart_items_id)).all()
        goods_map = {row.id: row.price for row in goods_rows}

        total_amount = 0.0
        for item_id, quantity in cart_items.items():
            price = goods_map.get(item_id)
            if not price:
                return f"商品{item_id}不存在"
            goods_total_amount = price * quantity
            total_amount += goods_total_amount
            session.add(
                UserOrderItem(
                    orderId=order_id,
                    userId=user_id,
                    goodsId=item_id,
                    quantity=quantity,
                    goodsTotalAmount=goods_total_amount
                )
            )

        session.add(
            OrderInfo(
                orderId=order_id,
                userId=user_id,
                status=0,
                remark=remark or '',
                receiptAddress=receipt_address,
                totalAmount=total_amount
            )
        )

        # Redis 缓存订单
        await redis_conn.setex(
            f"order:{order_id}", 31 * 60,
            json.dumps({
                'orderId': order_id,
                'goodsIds': cart_items_id,
                'quantities': list(cart_items.values()),
                'totalAmount': total_amount
            })
        )

        # 发布延迟取消消息（先于 commit：发布失败则不落库、购物车保留，用户可安全重试）
        await _publish_delayed_order_cancel(order_id)

        # 外部操作全部成功后，最后统一提交，保证一致性
        session.commit()
        await redis_conn.hdel(f"cart:{user_id}", *cart_items_id)
        return f"订单{order_id}已创建，订单金额为{total_amount}元"
    except Exception as e:
        session.rollback()
        return f"创建订单时失败，原因为：{str(e)}"
    finally:
        session.close()


@register_tool('during_sale_agent')
@tool
def add_to_cart(
        runtime: ToolRuntime,
        cart_items: List[ShoppingCart]
):
    """
    添加单个或多个购物车商品
    参数：
        cart_items: 需要添加的商品列表。
            其中：ShoppingCart为继承了BaseModel的类，其中需要填写的字段为：id-商品id，quantity-购买的商品数量
    """
    user_id = runtime.context.user_id
    session = SessionLocal()
    item_ids = [item.id for item in cart_items]
    try:
        with session:
            goods_rows = session.query(
                Goods.id, Goods.stock, Goods.status
            ).filter(Goods.id.in_(item_ids)).all()
            goods_map = {row.id: (row.stock, row.status) for row in goods_rows}
            for item in cart_items:
                goods = goods_map.get(item.id, None)
                if not goods:
                    return f"商品{item.id}不存在"
                stock, status = goods
                if item.quantity > stock:
                    return f"商品{item.id}库存不足"
                if status != 1:
                    return f"商品{item.id}暂时不上架"

            # 验证通过后，统一扣减库存并写入 Redis
            for item in cart_items:
                session.query(Goods).filter(Goods.id == item.id).update(
                    {'stock': Goods.stock - item.quantity})
                redis_conn.hset(f"cart:{user_id}", str(item.id), str(item.quantity))

            session.commit()
            return f"商品{item_ids}已添加到购物车"

    except Exception as e:
        return f"添加商品到购物车失败：{str(e)}"


@register_tool('during_sale_agent')
@tool
def search_goods_by_name(
        name: str,
        limit: int = 5,
):
    """
    按商品名称模糊搜索正常上架的商品，返回商品ID等基本信息。
    用于把用户口中的商品名/型号映射为商品ID（如售前推荐的商品），
    拿到 ID 后才能调用 add_to_cart 加入购物车。
    参数：
        name: 商品名称关键词
        limit: 最多返回多少条，默认5
    返回：
        商品列表，每项含 id（商品ID）、name、price（单位：分）、stock、status
    """
    session = SessionLocal()
    with session:
        rows = session.query(
            Goods.id, Goods.name, Goods.price, Goods.stock, Goods.status
        ).filter(
            Goods.name.like(f"%{name}%"),
            Goods.status == 1,
            Goods.isAD == False,  # noqa: E712
        ).limit(limit).all()
        goods_list = [
            {'id': row.id, 'name': row.name, 'price': row.price, 'stock': row.stock, 'status': row.status}
            for row in rows
        ]
    if not goods_list:
        return f"未找到名称包含「{name}」的正常上架商品"
    return goods_list


@register_tool('during_sale_agent')
@tool
def get_cart_items_info(
        runtime: ToolRuntime,
):
    """
    获取当前购物车中的购物信息
    返回：
        购物车中的商品列表或错误信息
    """
    user_id = runtime.context.user_id
    cart_items = redis_conn.hgetall(f"cart:{user_id}")
    if not cart_items:
        return "该用户还没订购商品"
    item_ids = cart_items.keys()
    session = SessionLocal()
    try:
        with session:
            items_info = session.query(
                Goods.id, Goods.name, Goods.price, Goods.category, Goods.brand, Goods.spec
            ).filter(Goods.id.in_(item_ids)).all()
            items_info_map = {row.id: [row.name, row.price, row.category, row.brand, row.spec] for row in items_info}
            items_info_list = []
            for item_id, item_quantity in cart_items.items():
                item_info = items_info_map.get(int(item_id), None)
                if not item_info:
                    return f"购物车中出现了不存在的商品{item_id}"
                # Goods.spec 是 SQLAlchemy JSON 列，ORM 查询返回的已是 dict，无需（也不能）json.loads
                spec = item_info[4]
                if isinstance(spec, str):
                    spec = json.loads(spec)
                info_map = {
                    'id': item_id,
                    'name': item_info[0],
                    'price': item_info[1],
                    'category': item_info[2],
                    'brand': item_info[3],
                    'spec': spec,
                }
                items_info_list.append(info_map)
            return items_info_list
    except Exception as e:
        return f"获取购物车商品信息失败：{str(e)}"


@register_tool('during_sale_agent')
@tool
def update_cart_items_info(
        runtime: ToolRuntime,
        item_id: int,
        quantity: int,
        cart_items: Optional[List[ShoppingCart]] = None
):
    """
    更改购物车商品数量
    参数：
        item_id: 需要更改数量的商品ID
        quantity: 更改后需要购买的商品数量
        cart_items: 当需要一次性更改多个商品数量时使用。
            其中：ShoppingCart为继承了BaseModel的类，其中需要填写的字段为：id-商品id，quantity-购买的商品数量
    注意：
        当填写了item_id与quantity参数时，参数cart_items不会生效；同理反之，当填写了参数cart_items，参数item_id与quantity不会生效。
        不允许出现调整购物车中不存在的商品数量的情况，否则将会返回错误信息
    返回：
        更改是否成功或错误信息
    """
    user_id = runtime.context.user_id
    exist_cart_items = redis_conn.hgetall(f"cart:{user_id}")
    if not exist_cart_items:
        return "该用户还没订购商品"

    updates = cart_items if cart_items else [ShoppingCart(id=item_id, quantity=quantity)]
    item_ids = [item.id for item in updates]

    session = SessionLocal()
    try:
        with session:
            goods_rows = session.query(
                Goods.id, Goods.stock, Goods.status
            ).filter(Goods.id.in_(item_ids)).all()
            goods_map = {row.id: (row.stock, row.status) for row in goods_rows}

            for item in updates:
                old_quantity = exist_cart_items.get(str(item.id))
                if not old_quantity:
                    return f"需要更改数量的商品{item.id}不在购物车当中，请添加到购物车"

                goods = goods_map.get(item.id)
                if not goods:
                    return f"商品{item.id}不存在"
                stock, status = goods
                if status != 1:
                    return f"商品{item.id}暂时不上架"

                differ_quantity = item.quantity - int(old_quantity)
                if differ_quantity > stock:
                    return f"商品{item.id}库存不足，当前库存为{stock}"

                session.query(Goods).filter(Goods.id == item.id).update(
                    {'stock': Goods.stock - differ_quantity}
                )
                redis_conn.hset(f"cart:{user_id}", str(item.id), str(item.quantity))

            session.commit()
            updated_ids = [str(item.id) for item in updates]
            return f"商品{updated_ids}数量已更新"

    except Exception as e:
        return f"更新购物车商品数量失败：{str(e)}"


@register_tool('during_sale_agent')
@tool
def delete_all_cart(
        runtime: ToolRuntime,
):
    """
    清空所有购物车商品
    返回：
        清空是否成功或错误信息
    """
    user_id = runtime.context.user_id
    try:
        redis_conn.hdel(f"cart:{user_id}")
        return "购物车已清空"
    except Exception as e:
        return f"清空购物车失败：{str(e)}"


if __name__ == '__main__':
    # gen_invoice(order_id=1984512124113784832)
    pass
