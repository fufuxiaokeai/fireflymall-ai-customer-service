from datetime import datetime
from typing import List

from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, JSON, Boolean, DateTime, Float, BigInteger, Text, DECIMAL, SmallInteger

from Tools.db import Base


class Goods(Base):
    __tablename__ = 'goods'
    id = Column(Integer, primary_key=True, comment='商品ID')
    name = Column(String(255), default='', comment='商品名称')
    price = Column(Integer, default=0, comment='单位：分')
    stock = Column(Integer, default=0, comment='商品库存')
    image = Column(String(500), default='', comment='商品图片')
    category = Column(String(100), default='', comment='商品分类')
    brand = Column(String(100), default='', comment='商品品牌')
    spec = Column(JSON, default=dict, comment='商品规格')
    sold = Column(Integer, default=0, comment='商品已售数量')
    comment_count = Column(Integer, default=0, comment='评论数')
    status = Column(Integer, default=1, comment='商品状态：1-正常上架 2-下架 3-逻辑删除')
    isAD = Column(Boolean, default=False, comment='是否为广告商品')
    create_time = Column(DateTime, default=datetime.now, comment='记录创建时间')
    update_time = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment='记录最后更新时间')


class ActivityInfo(Base):
    __tablename__ = 'activity'
    id = Column(Integer, primary_key=True, comment='活动主键ID')
    name = Column(String(255), default='', comment='活动名称')
    activity_type = Column(String(50), default='', comment='活动类型：满减/折扣/秒杀/团购/预售/优惠券/赠品')
    status = Column(String(50), default='draft',
                    comment='活动状态：draft-草稿 pending-待开始 ongoing-进行中 suspended-已暂停 completed-已结束 cancelled-已取消')
    activity_code = Column(String(100), nullable=True, comment='活动唯一编码，用于外部对接')
    description = Column(String(1000), default='', comment='活动详细规则描述')
    promotional_image = Column(String(500), nullable=True, comment='活动推广Banner图URL')
    goods_scope_type = Column(Integer, default=1,
                              comment='商品适用范围：1-全部商品 2-指定分类 3-指定商品 4-指定品牌 5-排除指定商品')
    user_scope_type = Column(Integer, default=1,
                             comment='用户适用范围：1-全部用户 2-新用户 3-指定会员等级 4-指定用户分组')
    user_scope_value = Column(JSON, default=dict, comment='用户范围扩展值，存储会员等级、用户分组ID等配置')
    total_quota = Column(Integer, nullable=True, comment='活动总名额/总库存，null表示不限制')
    per_user_limit = Column(Integer, nullable=True, comment='单个用户限购/参与次数，null表示不限制')
    full_reduction_threshold = Column(Integer, nullable=True, comment='满减活动门槛金额，单位：分')
    discount_threshold = Column(Float, nullable=True, comment='折扣活动折扣比例，取值范围0-1')
    other = Column(JSON, default=dict, comment='其他活动类型的扩展配置，JSON格式')
    activity_tag = Column(String(100), nullable=True, comment='活动标签，用于前端展示标记')
    start_time = Column(DateTime, default=datetime.now, comment='活动生效开始时间')
    end_time = Column(DateTime, nullable=True, comment='活动结束时间')


class CouponInfo(Base):
    __tablename__ = 'activity_coupon_relation'

    id = Column(Integer, primary_key=True, comment='关联表主键ID')
    activity_id = Column(Integer, default=0, comment='关联的活动ID')
    coupon_id = Column(Integer, default=0, comment='关联的优惠券ID')
    send_total_count = Column(Integer, nullable=True, comment='该活动渠道总发放数量，null表示不限量')
    send_used_count = Column(Integer, default=0, comment='该活动渠道已使用的优惠券数量')
    per_user_limit = Column(Integer, default=1, comment='单个用户最多可使用该渠道优惠券的次数')


class UserOrderItem(Base):
    __tablename__ = 'ordercart'
    id = Column(BigInteger, primary_key=True, comment='主键id')
    orderId = Column(BigInteger, default=0, comment='订单ID')
    userId = Column(BigInteger, default=0, comment='用户ID')
    goodsId = Column(Integer, default=0, comment='商品ID')
    quantity = Column(Integer, default=0, comment='购买的数量')
    goodsTotalAmount = Column(Float, default=0, comment='商品总金额(单位：元)')


class OrderItemInfoJson(BaseModel):
    goods_id: int = Field(default=0, description='商品ID')
    name: str = Field(default='', description='商品名称')
    quantity: int = Field(default=0, description='购买的数量')
    goodsTotalAmount: int = Field(default=0, description='商品总金额(单位：分)')


class UserOrderJson(BaseModel):
    order_id: int = Field(default=0, description='订单ID')
    status: str | None = Field(default='未知状态', description='订单状态')
    order_info: List[OrderItemInfoJson] = Field(default=[], description='订单总和信息')


class OrderInfo(Base):
    __tablename__ = 'order'
    orderId = Column(BigInteger, primary_key=True, comment='订单ID')
    userId = Column(BigInteger, default=0, comment='用户ID')
    totalAmount = Column(Float, default=0, comment='订单总金额(单位：元)')
    receiptAddress = Column(String(255), default='', comment='订单收货地址')
    status = Column(Integer, default=0, comment='订单状态，默认0，可选值为0: 未支付 1: 已支付 2: 已取消')
    remark = Column(String(255), default='', comment='订单备注')
    createTime = Column(DateTime, default=datetime.now, comment='记录创建时间')
    updateTime = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment='记录最后更新时间')
    payTime = Column(DateTime, default=None, comment='完成支付的时间')


class AfterSalesApply(Base):
    __tablename__ = 'after_sales_apply'
    apply_id = Column(BigInteger, primary_key=True, autoincrement=True, comment='售后申请ID')
    apply_no = Column(String(32), nullable=False, comment='售后申请编号(唯一)')
    user_id = Column(BigInteger, default=0, nullable=False, comment='用户ID')
    order_id = Column(Integer, default=0, nullable=False, comment='订单ID')

    after_sales_type = Column(Integer, default=0, nullable=False,
                              comment='售后类型: 1-仅退款 2-退货退款 3-换货 4-其他售后（表实际列名）')
    apply_reason = Column(String(200), default=None, comment='售后申请原因')
    apply_desc = Column(Text, default=None, comment='售后申请详细描述')
    apply_amount = Column(Float, default=0, comment='售后申请金额(单位：元)')
    actual_refund_amount = Column(Float, default=0, comment='实际退款金额(单位：元)')

    apply_status = Column(Integer, default=0,
                          comment='售后申请状态：0-待审核 1-商家同意 2-商家拒绝 3-用户取消 4-处理中 5-已完成 6-已关闭')
    refund_status = Column(Integer, default=0, comment='退款状态：0-未退款 1-退款中 2-退款成功 3-退款失败')
    return_goods_status = Column(Integer, default=0, comment='退货状态：0-未退货 1-待收货 2-已收货 3-收货异常')
    exchange_status = Column(Integer, default=0, comment='换货状态：0-未换货 1-待换货 2-已换货 3-已签收')

    create_time = Column(DateTime, default=datetime.now, comment='记录创建时间')
    update_time = Column(DateTime, default=datetime.now, onupdate=datetime.now, comment='记录最后更新时间')


class AfterSalesItem(Base):
    __tablename__ = 'after_sales_item'
    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='主键ID')
    apply_id = Column(BigInteger, nullable=False, comment='售后申请id')
    goods_id = Column(BigInteger, nullable=False, comment='商品id')
    spec = Column(JSON, default=None, comment='商品规格')
    image = Column(String(255), default=None, comment='商品图片URL')
    apply_num = Column(Integer, default=1, comment='申请售后数量')
    unit_price = Column(DECIMAL(10, 2), nullable=False, comment='商品单价')
    total_amount = Column(DECIMAL(10, 2), nullable=False, comment='商品总金额')
    problem_type = Column(SmallInteger, default=None,
                          comment='问题类型：1-质量问题 2-发错货 3-缺少配件 4-外观破损 5-其他')
    problem_desc = Column(String(500), default=None, comment='商品问题描述')
    create_time = Column(DateTime, default=datetime.now, nullable=False, comment='创建时间')


class AfterSalesEvidence(Base):
    __tablename__ = 'after_sales_evidence'
    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='主键ID')
    apply_id = Column(BigInteger, nullable=False, comment='售后申请ID')
    evidence_type = Column(SmallInteger, nullable=False, comment='凭证类型：1-商品问题图 2-物流面单 3-开箱视频 4-其他')
    file_url = Column(String(500), nullable=False, comment='文件地址')
    file_name = Column(String(200), default=None, comment='文件名')
    create_time = Column(DateTime, default=datetime.now, nullable=False, comment='创建时间')


class AfterSalesReturn(Base):
    __tablename__ = 'after_sales_return'
    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='主键ID')
    apply_id = Column(BigInteger, nullable=False, comment='售后申请ID')
    return_waybill_no = Column(String(64), default=None, comment='退货运单号')
    return_send_time = Column(DateTime, default=None, comment='用户发货时间')
    receiver_name = Column(String(50), nullable=False, comment='收件人姓名')
    receiver_phone = Column(String(20), nullable=False, comment='收件人电话')
    receiver_province = Column(String(32), nullable=False, comment='收件省')
    receiver_city = Column(String(32), nullable=False, comment='收件市')
    receiver_district = Column(String(32), nullable=False, comment='收件区')
    receiver_address = Column(String(200), nullable=False, comment='详细地址')
    receive_time = Column(DateTime, default=None, comment='商家签收时间')
    receive_status = Column(SmallInteger, default=0, comment='签收状态：0-待签收 1-已签收 2-拒签')
    receive_remark = Column(String(500), default=None, comment='收货备注（如破损、少件）')
    freight_amount = Column(DECIMAL(10, 2), default=0.00, comment='退货运费')
    freight_bearer = Column(SmallInteger, default=0, comment='运费承担方：1-用户 2-商家 3-平台')
    create_time = Column(DateTime, default=datetime.now, nullable=False, comment='创建时间')
    update_time = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False, comment='更新时间')


class AfterSalesExchange(Base):
    __tablename__ = 'after_sales_exchange'
    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='主键ID')
    apply_id = Column(BigInteger, nullable=False, comment='售后申请ID')
    new_sku_name = Column(String(200), nullable=False, comment='新商品名称')
    new_spec_values = Column(JSON, default=None, comment='新规格值')
    exchange_num = Column(Integer, nullable=False, default=1, comment='换货数量')
    delivery_waybill_no = Column(String(64), default=None, comment='新货物流单号')
    delivery_express_code = Column(String(32), default=None, comment='发货快递公司编码')
    delivery_express_name = Column(String(50), default=None, comment='发货快递公司名称')
    delivery_time = Column(DateTime, default=None, comment='商家发货时间')
    receiver_name = Column(String(50), nullable=False, comment='收件人姓名')
    receiver_phone = Column(String(20), nullable=False, comment='收件人电话')
    receiver_address = Column(String(500), nullable=False, comment='收件完整地址')
    sign_time = Column(DateTime, default=None, comment='用户签收时间')
    sign_status = Column(SmallInteger, default=0, comment='签收状态：0-待签收 1-已签收 2-拒签')
    price_diff = Column(DECIMAL(10, 2), default=0.00, comment='换货差价（正补负退）')
    create_time = Column(DateTime, default=datetime.now, nullable=False, comment='创建时间')
    update_time = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False, comment='更新时间')


class AfterSalesRefund(Base):
    __tablename__ = 'after_sales_refund'
    refund_id = Column(BigInteger, primary_key=True, autoincrement=True, comment='退款记录ID')
    apply_id = Column(BigInteger, nullable=False, comment='售后申请ID')
    refund_no = Column(String(200), nullable=False, comment='退款流水号')
    refund_amount = Column(DECIMAL(10, 2), nullable=False, comment='退款金额')
    refund_type = Column(SmallInteger, nullable=False, comment='退款类型：1-原路退回 2-余额退款 3-补偿退款 4-运费退款')
    pay_channel = Column(String(32), nullable=False, comment='支付渠道：alipay-支付宝 wechat-微信')
    refund_status = Column(SmallInteger, nullable=False, default=0, comment='退款状态：0-待处理 1-处理中 2-成功 3-失败')
    refund_time = Column(DateTime, default=None, comment='退款完成时间')
    third_party_refund_no = Column(String(64), default=None, comment='第三方退款单号')
    fail_reason = Column(String(200), default=None, comment='失败原因')
    operator_id = Column(BigInteger, default=None, comment='操作人ID(AI为0)')
    remark = Column(String(500), default=None, comment='备注')
    create_time = Column(DateTime, default=datetime.now, nullable=False, comment='创建时间')
    update_time = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False, comment='更新时间')


class AfterSalesLog(Base):
    __tablename__ = 'after_sales_log'
    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='主键ID')
    apply_id = Column(BigInteger, nullable=False, comment='售后申请ID')
    operator_type = Column(SmallInteger, nullable=False, comment='操作人类型：1-用户 2-商家 3-平台管理员 4-系统')
    operator_id = Column(BigInteger, default=None, comment='操作人ID(AI为0)')
    action_type = Column(String(32), nullable=False,
                         comment='操作类型：create-提交申请 audit_pass-审核通过 audit_reject-审核拒绝 ship-发货 receive-收货 refund-退款 '
                                 'cancel-取消')
    action_desc = Column(String(200), nullable=False, comment='操作描述')
    before_status = Column(SmallInteger, default=None, comment='操作前状态')
    after_status = Column(SmallInteger, default=None, comment='操作后状态')
    remark = Column(String(500), default=None, comment='备注信息')
    create_time = Column(DateTime, default=datetime.now, nullable=False, comment='操作时间')


class ShoppingCart(BaseModel):
    id: int = Field(description='购物车ID')
    quantity: int = Field(description='商品数量')
