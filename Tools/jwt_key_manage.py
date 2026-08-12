import base64
import os
import time
from datetime import datetime
from typing import Final

import redis
from apscheduler.schedulers.background import BackgroundScheduler

from load_config.config import config
from Tools.log_settings import LogSetting


class KeyManage:
    """
    JWT密钥管理类
    Python端不管理JWT的密钥更新，更新应在Java端处理
    """
    KEY_EXPIRATION: Final[int] = 35 * 24 * 60 * 60  # 35天
    KEY_ROTATION: Final[int] = 30 * 24 * 60 * 60  # 30天

    def __init__(self):
        redis_config = config['redis']
        _redis_pool = redis.ConnectionPool(
            host=redis_config['host'],
            port=redis_config['port'],
            db=redis_config['db'],
            password=redis_config['password']
        )
        # 调度任务只存内存：job 是绑定方法（含连接池等不可 pickle 对象）
        self._JOB_ID = 'jwt_key_manage'
        self._redis_con = redis.Redis(connection_pool=_redis_pool)
        self.scheduler = BackgroundScheduler()

    def start(self):
        # 启动期任何异常都不允许静默：密钥初始化/轮换失败会直接影响 token 校验，直接终止程序
        try:
            stored_expire = self._redis_con.get('jwt_key:expire_at')
            now = time.time()
            if not stored_expire:
                # 初始化，不存在过期时间
                next_expire_at = self.rotate_jwt_key()
                self._update_scheduler_job(next_expire_at)
                self.scheduler.start()
                return

            stored_expire = float(stored_expire)  # type: ignore
            remaining = stored_expire - now

            if remaining <= 0:
                # 密钥过期，需要旋转
                next_expire_at = self.rotate_jwt_key()
                self._update_scheduler_job(next_expire_at)
                self.scheduler.start()
                return

            # 密钥未过期，无需旋转
            self._update_scheduler_job(stored_expire)
            self.scheduler.start()
        except Exception as e:
            logger = LogSetting.create(__name__)
            logger.error(f"JWT密钥管理初始化失败，程序终止: {e}")
            raise

    def shutdown(self):
        self.scheduler.shutdown(wait=False)

    def rotate_jwt_key(self):
        # 旋转JWT密钥
        old_key = self._redis_con.hget('jwt_key', 'new')
        new_key = self._create_key()
        # 注意：HSETEX（hash 字段级 TTL）需要 Redis 8.0+ 支持，当前服务器不支持，
        # 改用 pipeline 原子写入：整 key 统一按 KEY_EXPIRATION 设置过期，
        # old 的 5 天过渡期随之延长至整 key 生命周期（旧 token 仍可验证，功能无碍）
        pipe = self._redis_con.pipeline()
        if old_key:
            # 若存在旧密钥，则将旧密钥移动到old密钥
            pipe.hset('jwt_key', 'old', old_key)
        pipe.hset('jwt_key', 'new', new_key)
        pipe.expire('jwt_key', self.KEY_EXPIRATION)
        pipe.execute()
        next_expire_at = time.time() + self.KEY_ROTATION
        self._redis_con.set('jwt_key:expire_at', next_expire_at)
        return next_expire_at

    def _update_scheduler_job(self, run_timestamp: float):
        self.scheduler.add_job(
            func=self.rotate_jwt_key,
            trigger='date',
            # apscheduler 的 run_date 只接受 datetime，时间戳需转换
            run_date=datetime.fromtimestamp(run_timestamp),
            id=self._JOB_ID,
            replace_existing=True,
            misfire_grace_time=300,
            coalesce=True,
        )

    @staticmethod
    def _create_key():
        raw_key = os.urandom(32)
        base64_key = base64.b64encode(raw_key).decode('utf-8')
        return base64_key
