"""
该文件面向管理员与对应的商家 普通用户无法访问带此处
"""
from typing import Optional, List

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from fastapi import APIRouter, Request
from pymilvus import MilvusClient

from SPO.route_results import RouteResponse, ResultCode
from Tools import LogSetting
from Tools.after_sales_tool import get_id
from load_config.config import config

logger = LogSetting.create(__name__)
route = APIRouter()

milvus_conf = config.get('databases').get('rag').get('milvus')

_embeddings = DashScopeEmbeddings(model="text-embedding-v4")

text_field = milvus_conf.get('text_field')
primary_field = milvus_conf.get('primary_field')
metadata_field = milvus_conf.get('metadata_field')
milvus_rag = Milvus(
    embedding_function=_embeddings,
    collection_name="firefly_mall_faq",
    connection_args=milvus_conf.get('conn_args'),
    index_params=milvus_conf.get('index_params'),
    search_params=milvus_conf.get('search_params'),
    consistency_level=milvus_conf.get('level'),
    drop_old=milvus_conf.get('drop_old'),
    metadata_field=metadata_field,
    primary_field=primary_field,
    text_field=text_field,
)

milvus_client = MilvusClient(
    uri=milvus_conf.get('conn_args').get('uri'),
    db_name=milvus_conf.get('conn_args').get('db_name'),
    token=milvus_conf.get('conn_args').get('token'),
)


class FAQDataInfo(BaseModel):
    h1: str = Field(description="一级标题")
    h2: Optional[str] = Field(default=None, description="二级标题")
    content: str = Field(description="常见问题内容")


@route.post('/add/faq')
async def add_new_faq(
        request: Request,
        faq_data: List[FAQDataInfo]
):
    login_info = request.state.login_info
    login_type = login_info.get('type')
    if login_type != 'admin':
        return RouteResponse.error(
            code=ResultCode.UNAUTHORIZED,
            msg="只有管理员才能添加常见问题"
        )

    if not faq_data:
        return RouteResponse.error(
            code=ResultCode.NOT_VALUES_ERROR,
            msg="常见问题数据不能为空"
        )
    documents = []
    for faq in faq_data:
        metadata = {
            'h1': faq.h1,
        }
        if faq.h2:
            metadata['h2'] = faq.h2
        documents.append(
            Document(
                id=f"faq-{get_id()}",
                page_content=faq.content,
                metadata=metadata
            )
        )
    try:
        await milvus_rag.aadd_documents(documents)
        return RouteResponse.ok(
            code=ResultCode.SUCCESS,
            msg="添加成功"
        )
    except Exception as e:
        logger.error(f"添加常见问题到 Milvus 失败: {e}")
        return RouteResponse.error(
            code=ResultCode.DB_ERROR,
            msg=f"添加常见问题失败，原因为：{e}"
        )


@route.get('/get/faq')
async def get_faq(
        h1: str,
        h2: Optional[str] = None,
        offset: int = 0,
        limit: int = 1000
):
    if offset < 0 or offset >= limit:
        return RouteResponse.error(
            code=ResultCode.INVALID_VALUE_ERROR,
            msg="不允许页偏移量小于0或大于等于分页数量"
        )
    if limit <= 0:
        return RouteResponse.error(
            code=ResultCode.INVALID_VALUE_ERROR,
            msg="不允许分页数量小于等于0"
        )

    query_filter = f'{metadata_field}["h1"] == "{h1}"' + (f'AND {metadata_field}["h2"] == "{h2}"' if h2 else '')

    results = milvus_client.query(
        collection_name=milvus_conf.get('collection_name'),
        offset=offset,
        limit=limit,
        filter=query_filter
    )

    return [{
        'h1': result.get(metadata_field, {}).get('h1'),
        'h2': result.get(metadata_field, {}).get('h2', ''),
        'content': result.get(text_field, ''),
    } for result in results]


@route.get('/get/all/faq')
async def get_all_faq(
        offset: int = 0,
        limit: int = 1000
):
    if offset < 0 or offset >= limit:
        return RouteResponse.error(
            code=ResultCode.INVALID_VALUE_ERROR,
            msg="不允许页偏移量小于0或大于等于分页数量"
        )
    if limit <= 0:
        return RouteResponse.error(
            code=ResultCode.INVALID_VALUE_ERROR,
            msg="不允许分页数量小于等于0"
        )

    results = milvus_client.query(
        collection_name=milvus_conf.get('collection_name'),
        offset=offset,
        limit=limit,
    )
    return [{
        'h1': result.get(metadata_field, {}).get('h1'),
        'h2': result.get(metadata_field, {}).get('h2', ''),
        'content': result.get(text_field, ''),
    } for result in results]


@route.put('/update/faq')
async def update_faq(
        request: Request,
        faq_data: List[FAQDataInfo]
):
    """
    faq_data说明：当faq_data中的 h1 和 h2 对应匹配时才会进行修改，
        若不存在匹配项，则忽略对应项
    """
    login_info = request.state.login_info
    login_type = login_info.get('type')
    if login_type != 'admin':
        return RouteResponse.error(
            code=ResultCode.UNAUTHORIZED,
            msg="只有管理员才能添加常见问题"
        )
    admin_permission = login_info.get('permission')
    if not admin_permission:
        return RouteResponse.error(
            code=ResultCode.UNAUTHORIZED,
            msg="管理员权限不能为空"
        )
    if admin_permission < 0:
        return RouteResponse.error(
            code=ResultCode.UNAUTHORIZED,
            msg="管理员权限不足"
        )

    update_faq_list = []
    for faq in faq_data:
        query_filter = f'{metadata_field}["h1"] == "{faq.h1}"' + (
            f'AND {metadata_field}["h2"] == "{faq.h2}"' if faq.h2 else '')
        results = milvus_client.query(
            collection_name=milvus_conf.get('collection_name'),
            filter=query_filter
        )
        if not results:
            continue
        # 理应来说，不会发生这种情况，但确保程序的正常运行决定加上这个判断
        if len(results) > 1:
            logger.warning(f"h1={faq.h1}, h2={faq.h2} 对应多个常见问题，当前只能更新一个")
            continue
        # 内容未变化时跳过，避免无意义写入
        if faq.content == results[0].get(text_field, ''):
            continue
        update_faq_list.append((results[0], faq))

    ids = [result.get(primary_field) for result, _ in update_faq_list]
    documents = [
        Document(
            id=result.get(primary_field),
            # 以请求中的新内容更新（原实现误用查询回来的旧内容，导致更新永不生效）
            page_content=faq.content,
            metadata=result.get(metadata_field, {})
        ) for result, faq in update_faq_list
    ]

    try:
        milvus_rag.upsert(ids, documents)
        return RouteResponse.ok(
            code=ResultCode.SUCCESS,
            msg="更新成功"
        )
    except Exception as e:
        logger.error(f"更新常见问题到 Milvus 失败: {e}")
        return RouteResponse.error(
            code=ResultCode.DB_ERROR,
            msg=f"更新常见问题失败，原因为：{e}"
        )


@route.delete('/delete/faq')
async def delete_faq(
        request: Request,
        h1: str,
        h2: Optional[str] = None,
):
    login_info = request.state.login_info
    login_type = login_info.get('type')
    if login_type != 'admin':
        return RouteResponse.error(
            code=ResultCode.UNAUTHORIZED,
            msg="只有管理员才能添加常见问题"
        )
    admin_permission = login_info.get('permission')
    if not admin_permission:
        return RouteResponse.error(
            code=ResultCode.UNAUTHORIZED,
            msg="管理员权限不能为空"
        )
    if admin_permission < 0:
        return RouteResponse.error(
            code=ResultCode.UNAUTHORIZED,
            msg="管理员权限不足"
        )
    query_filter = f'{metadata_field}["h1"] == "{h1}"' + (
        f'AND {metadata_field}["h2"] == "{h2}"' if h2 else '')
    results = milvus_client.query(
        collection_name=milvus_conf.get('collection_name'),
        filter=query_filter
    )
    if not results:
        return RouteResponse.error(
            code=ResultCode.NOT_FOUND,
            msg=f"h1={h1}, h2={h2} 对应常见问题不存在"
        )
    if len(results) > 1:
        logger.warning(f"h1={h1}, h2={h2} 对应多个常见问题，当前只能删除一个")
        return RouteResponse.error(
            code=ResultCode.NOT_FOUND,
            msg=f"h1={h1}, h2={h2} 对应多个常见问题，当前只能删除一个"
        )
    try:
        milvus_client.delete(filter=query_filter, collection_name=milvus_conf.get('collection_name'))
        return RouteResponse.ok(
            code=ResultCode.SUCCESS,
            msg="删除成功"
        )
    except Exception as e:
        logger.error(f"删除常见问题到 Milvus 失败: {e}")
        return RouteResponse.error(
            code=ResultCode.DB_ERROR,
            msg=f"删除常见问题失败，原因为：{e}"
        )


@route.delete('/delete/all/faq')
async def delete_all_faq(
        request: Request,
):
    login_info = request.state.login_info
    login_type = login_info.get('type')
    if login_type != 'admin':
        return RouteResponse.error(
            code=ResultCode.UNAUTHORIZED,
            msg="只有管理员才能添加常见问题"
        )
    admin_permission = login_info.get('permission')
    if not admin_permission:
        return RouteResponse.error(
            code=ResultCode.UNAUTHORIZED,
            msg="管理员权限不能为空"
        )
    if admin_permission == 1:
        return RouteResponse.error(
            code=ResultCode.UNAUTHORIZED,
            msg="管理员权限不足，请向最高管理员申请"
        )
    try:
        milvus_client.delete(
            collection_name=milvus_conf.get('collection_name'),
            filter=f'{primary_field} != "0"'
        )
        return RouteResponse.ok(
            code=ResultCode.SUCCESS,
            msg="全部删除成功"
        )
    except Exception as e:
        logger.error(f"删除全部问题失败，原因为：{e}")
        return RouteResponse.error(
            code=ResultCode.DB_ERROR,
            msg=f"删除失败，原因为：{e}"
        )
