from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import QueuePool
from load_config.config import config
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.pool import AsyncAdaptedQueuePool

product_mysql_config = config['databases']['mysql']

DATABASE_URL = (
    f"mysql+pymysql://{product_mysql_config['user']}:{product_mysql_config['password']}"
    f"@{product_mysql_config['host']}:{product_mysql_config['port']}/{product_mysql_config['db']}"
    "?charset=utf8mb4"
)

ASYNC_DATABASE_URL = (
    f"mysql+aiomysql://{product_mysql_config['user']}:{product_mysql_config['password']}"
    f"@{product_mysql_config['host']}:{product_mysql_config['port']}/{product_mysql_config['db']}"
    "?charset=utf8mb4"
)

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,  # 连接池大小
    max_overflow=20,  # 超出 pool_size 后最多再创建的连接数
    pool_recycle=3600,  # 连接回收时间（秒），防止 MySQL 8小时超时
    pool_pre_ping=True,  # 每次使用前 ping 一下，自动重连
    echo=False,
)

async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    poolclass=AsyncAdaptedQueuePool,
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,
    pool_pre_ping=True,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
AsyncSessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=async_engine)
Base = declarative_base()


def to_json(obj):
    return {col.key: getattr(obj, col.key) for col in obj.__table__.columns}
