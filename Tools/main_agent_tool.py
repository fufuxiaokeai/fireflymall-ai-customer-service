from typing import NotRequired, Optional, TypedDict

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Milvus

from Tools import LogSetting
from Tools.registry import register_tool
from langchain_core.tools import tool

from load_config.config import config

logger = LogSetting.create(__name__)
milvus_conf = config.get('databases').get('rag').get('milvus')
if not milvus_conf:
    raise ValueError('Milvus 配置不能为空')


class SearchFAQConfig(TypedDict):
    h1: str
    h2: NotRequired[str]


_embeddings = DashScopeEmbeddings(model="text-embedding-v4")

milvus_rag = Milvus(
    collection_name=milvus_conf.get('collection_name'),
    embedding_function=_embeddings,
    primary_field=milvus_conf.get('primary_field'),
    metadata_field=milvus_conf.get('metadata_field'),
    connection_args=milvus_conf.get('conn_args'),
    index_params=milvus_conf.get('index_params'),
    search_params=milvus_conf.get('search_params'),
    consistency_level=milvus_conf.get('level'),
    drop_old=milvus_conf.get('drop_old'),
    text_field=milvus_conf.get('text_field'),
)


@register_tool('main_agent')
@tool
def search_faq(query: str, k: int = 3, search_config: Optional[SearchFAQConfig] = None):
    """
    搜索 商城FAQ
    参数：
        query: 搜索的关键词（必填）
        k: 返回的FAQ数量（可选，默认3）
        search_config: 搜索元数据，可以让搜索更加准确（可选）
            关于search_config(字典类型)的详细说明：
                h1: 一级分类
                h2: 二级分类（可选）
            若不知道search_config应该填哪些，不允许乱填，先查询对应的元数据再添写
    返回：
        一个包含FAQ的列表，每个FAQ包含对应的元数据和内容
        如果没有找到相关的FAQ，返回空列表
    """
    search_str = None
    metadata_field = milvus_conf.get('metadata_field')
    if search_config:
        search_str = f'{metadata_field}["h1"] == "{search_config.get("h1")}"'
        if search_config.get('h2'):
            search_str += f' and {metadata_field}["h2"] == "{search_config.get("h2")}"'
    try:
        results = milvus_rag.similarity_search(
            query,
            k=k,
            expr=search_str,
        )
        res_list = [{'content': result.page_content, 'metadata': result.metadata} for result in results]
        return f"总共有：{len(res_list)}个结果\n返回：{res_list}"
    except Exception as e:
        logger.error(f"搜索失败，原因为：{str(e)}")
        return f"搜索失败，原因为：{str(e)}"


@register_tool('main_agent')
@tool
def get_all_faq_config(offset: int = 0, limit: int = 1000):
    """
    获取 所有FAQ的元数据
    参数：
        offset: 分页偏移量（可选，默认0）
        limit: 分页数量（可选，默认1000）
    返回：
        返回一个有关FAQ相关元数据列表
        如：[
            {
                'h1': 一级分类,
                'h2': 同一一级分类下可选的二级分类列表
            },
            {
                'h1': ...,
                'h2': [..., ....]
            },
            .....
        ]
        注意：有些一级分类是没有二级分类的，这种情况下，对应的h2字段为空列表[]
    """
    try:
        if offset < 0 or offset >= limit:
            return "不允许分页偏移量小于0或大于等于分页数量"
        if limit <= 0:
            return "不允许分页数量小于等于0"

        from pymilvus import MilvusClient
        client = MilvusClient(
            uri=milvus_conf.get('conn_args').get('uri'),
            token=milvus_conf.get('conn_args').get('token'),
            db_name=milvus_conf.get('conn_args').get('db_name'),
        )
        results = client.query(
            collection_name=milvus_conf.get('collection_name'),
            offset=offset,
            limit=limit,
        )

        metadata_dict = {}
        metadata_field = milvus_conf.get('metadata_field')
        for res in results:
            metadata = res.get(metadata_field)
            if not metadata:
                continue
            h1 = metadata.get('h1')
            h2 = metadata.get('h2')
            if h1 not in metadata_dict:
                metadata_dict[h1] = set()
            if h2:
                metadata_dict[h1].add(h2)

        metadata_list = [
            {'h1': h1, 'h2': list(h2_set)}
            for h1, h2_set in metadata_dict.items()
        ]
        return metadata_list
    except Exception as e:
        logger.error(f"获取所有FAQ元数据失败，原因为：{str(e)}")
        return f"获取所有FAQ元数据失败，原因为：{str(e)}"


if __name__ == '__main__':
    # print(get_all_faq_config())
    # print(search_faq('购物', search_config=SearchFAQConfig(h1='购物与订单基础')))
    # print(search_faq('优惠活动'))
    # print(search_faq('优惠活动', search_config=SearchFAQConfig(h1='优惠活动', h2='通用叠加规则（核心必看）')))
    pass
