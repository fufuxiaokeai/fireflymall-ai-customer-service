import logging
import threading
from logging import Formatter
import os
import re
from typing import Union
from logging.handlers import RotatingFileHandler
import warnings

from load_config.config import config, ROOT_BASE_DIR_PATH

LOG_COLOR: dict[int, str] = {
    logging.DEBUG: '\033[36m',  # 青色
    logging.INFO: "\033[32m",  # 绿色
    logging.WARNING: "\033[33m",  # 黄色
    logging.ERROR: "\033[31m",  # 红色
    logging.CRITICAL: "\033[35m"  # 紫色
}
RESET_COLOR = "\033[0m"  # 重置颜色

_LEVEL: dict[str, int] = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}
_UNIT_MAP = {
    # 字节
    'B': 1,
    # 千字节
    'KB': 1024, 'KIB': 1024,
    # 兆字节
    'MB': 1024 ** 2, 'MIB': 1024 ** 2,
    # 吉字节
    'GB': 1024 ** 3, 'GIB': 1024 ** 3,
    # 太字节
    'TB': 1024 ** 4, 'TIB': 1024 ** 4,
    # 拍字节（有备无患）
    'PB': 1024 ** 5, 'PIB': 1024 ** 5,
}

root_path = ROOT_BASE_DIR_PATH
_log_conf = config['logging']
_log_level = _LEVEL.get(_log_conf['level'].upper(), logging.DEBUG)

if _log_conf['file_path']:
    _log_path = (root_path / _log_conf['file_path']).resolve()
else:
    _log_path = None

_log_formatter = _log_conf['formatter']
_log_create = _log_conf['create_file']
_log_backup_count = _log_conf['backup_count']

_SIZE_PATTERN = re.compile(
    r'^\s*([\d.]+)\s*(B|KB|MB|GB|TB|PB|KIB|MIB|GIB|TIB|PIB)\s*$',
    re.IGNORECASE
)


def parse_size(value: Union[str, int, float]) -> int:
    """
    将类似 '5MB'、'10.5 KiB'、'512B' 的字符串转成整数（字节数）。
    如果已经是数字，直接返回整数。
    如果字符串无法识别，抛出 ValueError。
    """
    if isinstance(value, (int, float)):
        return int(value)

    value = value.strip()
    match = _SIZE_PATTERN.match(value)
    if not match:
        raise ValueError(f"无法解析的大小字符串: '{value}'")

    number = float(match.group(1))
    unit = match.group(2).upper()
    multiplier = _UNIT_MAP[unit]
    return int(number * multiplier)


_log_max_bytes = parse_size(_log_conf['max_bytes'])


class ColoredFormatter(Formatter):
    _msg_re = re.compile(r'%\(message(?::(\d+))?\)s')
    _lock = threading.Lock()

    def __init__(self, fmt=None, datefmt=None, style='%', is_command=True):
        self._max_msg_len = None
        clean_fmt = fmt
        if fmt:
            match = self._msg_re.search(fmt)
            if match:
                len_str = match.group(1)
                self._max_msg_len = int(len_str) if len_str else None
                clean_fmt = self._msg_re.sub('%(message)s', fmt)

        super().__init__(clean_fmt, datefmt, style)
        self.is_command = is_command

    def format(self, record: logging.LogRecord):
        with self._lock:
            full_msg = record.getMessage()
            color = LOG_COLOR.get(record.levelno, RESET_COLOR)
            if self._max_msg_len is not None and len(full_msg) > self._max_msg_len:
                record.msg = full_msg[:self._max_msg_len] + '...'
                record.args = None

            log_msg = super().format(record)

            return f"{color}{log_msg}{RESET_COLOR}" if self.is_command else log_msg


class LogSetting:
    def __init__(self, logger_name: str):
        self.file_name = logger_name
        self.logger = logging.getLogger(self.file_name)
        self.logger.setLevel(_log_level)
        self.formatter = _log_formatter
        self.log_level = _log_level

    def create_logger(self):
        """
        创建日志记录器

        :return: 日志记录器
        """
        handler = logging.StreamHandler()
        formatter = ColoredFormatter(self.formatter, datefmt='%Y-%m-%d %H:%M:%S')
        if _log_create and _log_path:
            file = self.name(_log_path)
            file_handler = RotatingFileHandler(
                filename=file,
                maxBytes=_log_max_bytes,
                backupCount=_log_backup_count,
                encoding='utf-8',
            )
            file_handler.setFormatter(ColoredFormatter(self.formatter, datefmt='%Y-%m-%d %H:%M:%S', is_command=False))
            self.logger.addHandler(file_handler)
        else:
            warnings.warn("create_file为False或file_name为空字符串，将不创建日志文件")

        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        return self.logger

    @classmethod
    def create(cls, logger_name: str):
        logger = cls(logger_name)
        return logger.create_logger()

    def name(self, file_parent_path: str) -> str:
        if not os.path.exists(file_parent_path):
            os.makedirs(file_parent_path, exist_ok=True)
        if '.' in self.file_name:
            file_name_list = self.file_name.split('.')
        else:
            file_name_list = [self.file_name]

        for file_name in file_name_list[:-1]:
            file_parent_path = os.path.join(file_parent_path, file_name)
            if not os.path.exists(file_parent_path):
                os.makedirs(file_parent_path, exist_ok=True)

        return os.path.join(file_parent_path, f"{file_name_list[-1]}.log")
