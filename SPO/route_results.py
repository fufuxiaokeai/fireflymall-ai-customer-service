from enum import IntEnum
from typing import TypeVar, Generic

from pydantic import BaseModel, Field

_T = TypeVar('_T')


class ResultCode(IntEnum):
    SUCCESS = 0
    UPLOAD_SUCCESS = 2001
    DOWNLOAD_SUCCESS = 2002
    SERVER_ERROR = 500
    NOT_VALUES_ERROR = 4001
    DOWNLOAD_ERROR = 4002
    UPLOAD_LIMIT_ERROR = 4003
    INVALID_VALUE_ERROR = 4004
    DB_ERROR = 4010
    UNAUTHORIZED = 401
    NOT_FOUND = 402

    @property
    def default_msg(self) -> str:
        msg_map = {
            ResultCode.SUCCESS: "响应成功",
            ResultCode.SERVER_ERROR: "服务器错误",
            ResultCode.UNAUTHORIZED: "未授权",
            ResultCode.UPLOAD_SUCCESS: "上传成功",
            ResultCode.DOWNLOAD_SUCCESS: "下载成功",
            ResultCode.DOWNLOAD_ERROR: "下载失败",
            ResultCode.UPLOAD_LIMIT_ERROR: "上传文件数量超限",
            ResultCode.NOT_VALUES_ERROR: "无值错误",
            ResultCode.INVALID_VALUE_ERROR: "无效值错误",
            ResultCode.DB_ERROR: "数据库错误",
            ResultCode.NOT_FOUND: "未找到",
        }
        return msg_map.get(self, "未知错误")


class RouteResponse(BaseModel, Generic[_T]):
    code: ResultCode = Field(default=ResultCode.SUCCESS, description="状态码")
    msg: str = Field(default="success", description="状态描述")
    data: _T | None = Field(default=None, description="具体数据")

    @classmethod
    def ok(cls, *,
           code: ResultCode = ResultCode.SUCCESS,
           msg: str = "", data: _T | None = None) -> "RouteResponse[_T]":
        return cls(
            code=code,
            msg=msg or code.default_msg,
            data=data
        )

    @classmethod
    def error(cls, *,
              code: ResultCode = ResultCode.SERVER_ERROR,
              msg: str = "", data: _T | None = None) -> "RouteResponse[_T]":
        return cls(
            code=code,
            msg=msg or code.default_msg,
            data=data
        )
