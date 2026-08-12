import base64
import json
import mimetypes
import os
from typing import Literal, Optional

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import ElasticsearchStore
from langchain_core.tools import tool
from langchain_elasticsearch import DistanceStrategy, ApproxRetrievalStrategy
from langchain_openai import ChatOpenAI

from SPO.data_model import Goods, ActivityInfo, CouponInfo
from Tools.log_settings import LogSetting
from Tools.registry import register_tool

from Tools.db import to_json, SessionLocal
from sqlalchemy import text

from load_config.config import config

logger = LogSetting.create(__name__)
_es_config = config.get('databases').get('rag').get('es')
if not _es_config:
    raise ValueError('Elasticsearch配置不能为空')
_embeddings = DashScopeEmbeddings(model="text-embedding-v4")
_es_client = ElasticsearchStore.connect_to_elasticsearch(
    es_url=_es_config.get('db_url'),
    cloud_id=_es_config.get('cloud_id'),
    username=_es_config.get('username'),
    password=_es_config.get('password'),
    es_params=_es_config.get('es_params'),
)
_es_rag = ElasticsearchStore(
    index_name=_es_config.get('index_name'),
    es_connection=_es_client,
    embedding=_embeddings,
    distance_strategy=DistanceStrategy.COSINE,
    vector_query_field='goods_vector',
    strategy=ApproxRetrievalStrategy(  # type: ignore
        hybrid=False,
    ),
    query_field='name'
)


@register_tool('product_process_agent')
@tool
def get_products_info(
        limit: int = 100,
        offset: int = 1,
        status: Literal[1, 2, 3] = 1,
        isAD: bool = False,
        keyword: Optional[str] = None,
        is_intelligence: bool = False,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        sql: Optional[str] = None,
):
    """
    获取商品信息，默认筛选非广告、正常状态下100个商品的全部信息
    允许通过修改工具参数进行筛选，如筛选广告商品、筛选已删除商品等
    参数：
        limit：返回的商品数量，默认100个
        offset：偏移量，默认1
        status：商品状态，默认1，可选值为1、2、3
        isAD：是否为广告商品，默认False
        category：商品类别，默认None
        brand：商品品牌，默认None
        keyword：查询关键词或者是查询语义，默认为None
        is_intelligence：是否使用向量语义检索商品，默认False；开启时必须提供 keyword
        sql：自定义SQL语句
    """
    goods_list = []
    with SessionLocal() as session:
        if sql:
            # 安全限制：自定义 SQL 只允许 SELECT 查询，防止注入写操作
            sql_stripped = sql.strip()
            if not sql_stripped.upper().startswith('SELECT'):
                return "自定义 SQL 仅支持 SELECT 查询语句"
            if sql_stripped.rstrip(';').count(';'):
                return "自定义 SQL 不支持多语句"
            result = session.execute(text(sql))
            rows = result.mappings().all()
            for row in rows:
                row = dict(row)
                if row.get('spec'):
                    row['spec'] = json.loads(row['spec'])
                goods_list.append(row)
            return goods_list
    query_conf = {
        'bool': {
            'must': [
                {'term': {'status': status}},
                {'term': {'isAD': isAD}}
            ]
        }
    }
    if offset <= 0:
        return "offset 字段不能小于等于0"
    if limit <= 0:
        return "limit 字段不能小于等于0"
    if offset > limit:
        return "offset 不能比 limit 大"
    if category:
        query_conf['bool']['must'].append({'term': {'category': category}})  # type: ignore
    if brand:
        query_conf['bool']['must'].append({'term': {'brand': brand}})  # type: ignore
    if keyword:
        query_conf['bool']['must'].append({'match': {'name': keyword}})  # type: ignore
    if is_intelligence:
        if not keyword:
            return "智能检索需要提供 keyword 参数"
        # filter 是 ES Query DSL 子句列表，进入 knn 的 filter 上下文（候选生成后再过滤）
        # 字段名以 goods 索引实际 mapping 为准：id/name/price/stock/image/category/brand/spec/sold/status/isAD
        knn_filter = [
            {'term': {'status': status}},
            {'term': {'isAD': isAD}},
        ]
        if category:
            knn_filter.append({'term': {'category': category}})
        if brand:
            knn_filter.append({'term': {'brand': brand}})
        docs = _es_rag.similarity_search(
            query=keyword,
            k=limit,
            fetch_k=min(2 * limit, 10000),
            filter=knn_filter,
            fields=['id', 'price', 'stock', 'image', 'category',
                    'brand', 'spec', 'sold', 'status', 'isAD'],
        )
        # page_content 是 name（query_field），其余字段在 metadata 里，
        # 拼回与 raw/SQL 路径一致的完整结构（含 name）
        goods_list = [{'name': doc.page_content, **doc.metadata} for doc in docs]
        logger.info(f"ES智能检索到商品数量：{len(goods_list)}")
        return goods_list
    response = _es_client.search(
        index=_es_config.get('index_name'),
        query=query_conf,
        from_=(offset - 1) * limit,
        size=limit,
        source_excludes=['goods_vector'],
        sort=[{'id': {'order': 'asc'}}],
    )
    for hit in response['hits']['hits']:
        goods_list.append(hit['_source'])
    logger.info(f"ES查询到商品数量：{len(goods_list)}")
    return goods_list


@register_tool('product_process_agent')
@tool
def understand_file(file_url: str, problem_str: str = '请对文件进行描述'):
    """
    对文件进行理解，返回文件的描述
    被动分析：仅在用户明确要求分析文件内容时才调用本工具；
    用户只是上传文件、未要求分析时，不要主动调用
    :param file_url: 用户上传的文件地址
    :param problem_str: 问题
    :return: 文件描述
    """
    image_agent = ChatOpenAI(
        model='qwen3.7-plus',
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        base_url='https://ws-87ass17j0hnx43k7.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'
    )
    # 路径限制：只允许访问用户上传目录内的文件，防止读取服务器任意文件
    upload_dir = os.path.abspath(config['file']['upload_dir'])
    file_abs = os.path.abspath(file_url)
    if not file_abs.startswith(upload_dir + os.sep):
        return "仅支持访问用户上传目录中的文件"

    # 处理文件信息
    mime_type, _ = mimetypes.guess_type(file_url)
    if not mime_type:
        mime_type = 'image/png'  # 保底默认

    with open(file_url, 'rb') as f:
        base64_str = base64.b64encode(f.read()).decode('utf-8')
    logger.info(f"处理后的base_64编码{base64_str[:100]}...{base64_str[-100:]}")
    message = [{
        'role': 'user',
        'content': [
            {'type': 'text', 'text': problem_str},
            {
                'type': 'image',
                'base64': base64_str,
                'mime_type': mime_type,
            }
        ]
    }]
    response = image_agent.invoke(message)
    logger.info(f"图像模型的回答：{response.content}")
    return response.content


@register_tool('product_process_agent')
@tool
def get_category():
    """
    获取商品类别
    返回：
        商品类别列表与对应的数量
    """
    session = SessionLocal()
    with session:
        rows = session.query(Goods.category).group_by(Goods.category).all()
        category_list = [row.category for row in rows]
    return f"共有{len(category_list)}个类别，分别为{category_list}"


@register_tool('product_process_agent')
@tool
def get_brand():
    """
    获取商品品牌
    返回：
        品牌列表与对应的数量
    """
    session = SessionLocal()
    with session:
        rows = session.query(Goods.brand).group_by(Goods.brand).all()
        brand_list = [row.brand for row in rows]
    return f"共有{len(brand_list)}个品牌，分别为{brand_list}"


@register_tool('product_process_agent')
@tool
def get_activity_info(
        name: Optional[str] = None,
        activity_type: Literal[
            'full_reduction',
            'discount',
            'seckill',
            'group_buy',
            'presale',
            'coupon_issue',
            'gift'
        ] = 'full_reduction',
        status: Literal[
            'draft',
            'pending',
            'ongoing',
            'suspended',
            'completed',
            'cancelled',
        ] = 'ongoing',
        goods_scope_type: Literal[1, 2, 3, 4, 5] = 1,
        user_scope_type: Literal[1, 2, 3, 4] = 1,
):
    """
    获取活动的基本信息
    参数：
        name：活动名称，默认None
        activity_type：活动类型（分别对应）：full_reduction-满减/discount-折扣/seckill-秒杀/group_buy-团购/presale-预售/coupon_issue-优惠券/gift-赠品
        status：活动状态（分别对应）：draft-草稿/pending-待开始/ongoing-进行中/suspended-已暂停/completed-已结束/cancelled-已取消
        goods_scope_type: 商品适用范围：1-全部商品/2-指定分类/3-指定商品分类/4-4指定品牌/5-排除指定商品
        user_scope_type: 用户适用范围：1-全部用户/2-新用户/3-指定会员等级/4-指定用户分组
    """
    session = SessionLocal()
    with session:
        query = session.query(ActivityInfo).filter(
            ActivityInfo.activity_type == activity_type,
            ActivityInfo.status == status,
            ActivityInfo.goods_scope_type == goods_scope_type,
            ActivityInfo.user_scope_type == user_scope_type
        )
        if name:
            query = query.filter(ActivityInfo.name.like(f"%{name}%"))

        activity_list = [to_json(item) for item in query.all()]
        logger.info(f"查询到活动数量：{len(activity_list)}")

        if not activity_list:
            return "未查询到符合条件的活动"
    return activity_list


@register_tool('product_process_agent')
@tool
def get_coupon_info(activity_id: int, coupon_id: Optional[int] = None):
    """
    查询优惠卷基本信息，该优惠卷与活动息息相关
    参数：
         activity_id: 活动ID
         coupon_id: 优惠卷ID，可选
    """
    session = SessionLocal()
    with session:
        query = session.query(CouponInfo).filter(
            CouponInfo.activity_id == activity_id
        )
        if coupon_id:
            query = query.filter(CouponInfo.coupon_id == coupon_id)

        coupon_list = [to_json(item) for item in query.all()]
        logger.info(f"查询到优惠券数量：{len(coupon_list)}")

        if not coupon_list:
            return "未查询到符合条件的优惠券"
    return coupon_list


if __name__ == '__main__':
    # goods_list = get_products_info(limit=10, offset=1, status=1)
    # print(goods_list)
    # goods_list = get_products_info(limit=10, offset=1, category='拉杆箱', brand='RIMOWA', status=1)
    # print(goods_list)
    # goods_list = get_products_info(limit=10, offset=1, brand='RIMOWA', status=1)
    # print(goods_list)
    # goods_list = get_products_info(limit=10, offset=1, category='拉杆箱', status=1)
    # print(goods_list)
    # goods_list = get_products_info(limit=10, offset=1, category='拉杆箱', brand='RIMOWA', status=1, keyword='托运箱')
    # print(goods_list)
    # goods_list = get_products_info(limit=10, offset=1, brand='RIMOWA', status=1, keyword='托运箱')
    # print(goods_list)
    # goods_list = get_products_info(limit=10, offset=1, category='拉杆箱', status=1, keyword='托运箱')
    # print(goods_list)
    # goods_list = get_products_info(limit=10, offset=1, status=1, keyword='手机')
    # goods_list = get_products_info(sql='select * from goods limit 10')
    # print(goods_list)
    # image_url = r"D:\matebook D16\Documents\Pictures\IDEA轮询壁纸\小桃桃.png"
    # description = understand_image(image_url)
    # print(description)
    # activity_list = get_activity_info(status='completed')
    # print(activity_list)
    # coupon_info_list = get_coupon_info(114514)
    # print(coupon_info_list)
    # category_list = get_category()
    # print(category_list)
    # brand_list = get_brand()
    # print(brand_list)
    pass
