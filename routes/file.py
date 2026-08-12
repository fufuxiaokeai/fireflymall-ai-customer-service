import os
import uuid
from typing import Annotated, Optional
from urllib.parse import quote

import aiofiles
import redis.asyncio as redis
from fastapi import APIRouter, Request, UploadFile, File, Form
from starlette.responses import StreamingResponse

from SPO.route_results import RouteResponse, ResultCode
from load_config.config import config

route = APIRouter()

UPLOAD_DIR = config['file']['upload_dir']
DOWNLOAD_DIR = config['file']['download_dir']

if not os.path.isdir(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR, exist_ok=True)

if not os.path.isdir(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# 单次对话（thread）累计上传文件数量上限
MAX_FILES_PER_CONVERSATION = 3
# 单次请求全部文件累计大小上限
MAX_UPLOAD_TOTAL_SIZE = 100 * 1024 * 1024  # 100MB
# 对话维度累计计数存 Redis
UPLOAD_COUNT_KEY_PREFIX = 'upload:count:'
UPLOAD_COUNT_TTL = 24 * 3600

_redis_conf = config.get('redis', {})
_redis_pool = redis.ConnectionPool(
    host=_redis_conf.get('host', 'localhost'),
    port=_redis_conf.get('port', 6379),
    decode_responses=True,
    password=_redis_conf.get('password', None),
)
_redis_con = redis.Redis(connection_pool=_redis_pool)


@route.post('/upload')
async def create_file(
    files: Annotated[list[UploadFile], File()],
    thread_id: Annotated[Optional[str], Form()] = None,
    request: Request = None,
):
    if not files:
        return RouteResponse.error(msg="未接收到任何文件")

    # 单次对话累计上限：key 用 thread_id（新对话传新 thread_id 天然不继承），
    # 缺省回退 user_id；TTL 无活动自动清零
    login_id = request.state.login_info.get('id')
    key = f'{UPLOAD_COUNT_KEY_PREFIX}{thread_id or login_id}'
    used = await _redis_con.incrby(key, len(files))
    await _redis_con.expire(key, UPLOAD_COUNT_TTL)
    if used > MAX_FILES_PER_CONVERSATION:
        await _redis_con.decrby(key, len(files))
        return RouteResponse.error(
            code=ResultCode.UPLOAD_LIMIT_ERROR,
            msg=f"单次对话最多上传 {MAX_FILES_PER_CONVERSATION} 个文件，"
                f"当前已上传 {used - len(files)} 个，本次 {len(files)} 个将超出上限",
        )

    result = []
    saved_paths = []
    total_size = 0
    try:
        for file in files:
            file_name, file_suffix = os.path.splitext(file.filename)
            safe_name = f"{uuid.uuid4().hex}({file_name}){file_suffix}"
            save_path = os.path.join(UPLOAD_DIR, safe_name)

            async with aiofiles.open(save_path, "wb") as f:
                while chunk := await file.read(1024 * 1024):
                    total_size += len(chunk)
                    if total_size > MAX_UPLOAD_TOTAL_SIZE:
                        raise ValueError(
                            f"单次上传全部文件总大小超过 {MAX_UPLOAD_TOTAL_SIZE // 1024 // 1024}MB 限制"
                        )
                    await f.write(chunk)

            saved_paths.append(save_path)
            result.append({"filename": safe_name, "original_name": file.filename})
    except Exception as e:
        # 保存失败时回滚计数，避免失败轮次占用对话额度；清理已写入的孤儿文件
        await _redis_con.decrby(key, len(files))
        for path in saved_paths:
            try:
                os.remove(path)
            except OSError:
                pass
        return RouteResponse.error(
            code=ResultCode.UPLOAD_LIMIT_ERROR,
            msg=f"上传失败：{e}",
        )

    return RouteResponse.ok(code=ResultCode.UPLOAD_SUCCESS, data=result)


@route.get('/download/{filename}')
async def download_file(filename: str):
    download_dir = os.path.abspath(DOWNLOAD_DIR)
    download_path = os.path.abspath(os.path.join(DOWNLOAD_DIR, filename))

    # 路径穿越防护：文件名可能携带 ../ 等路径段，解析后必须仍位于下载目录内
    if not download_path.startswith(download_dir + os.sep):
        return RouteResponse.error(code=ResultCode.DOWNLOAD_ERROR, msg="非法的文件名")

    if not os.path.exists(download_path):
        return RouteResponse.error(code=ResultCode.DOWNLOAD_ERROR, msg="文件不存在")

    async def file_iterator():
        async with aiofiles.open(download_path, "rb") as f:
            while chunk := await f.read(1024 * 1024):
                yield chunk

    encoded_filename = quote(filename)
    return StreamingResponse(
        file_iterator(),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"
        }
    )
