"""
AI 客户服务
项目运行文件
"""
import os
import asyncio

if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from load_config.config import config, ROOT_BASE_DIR_PATH
hf_conf = config['huggingface']

if hf_conf['mirror']:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

if hf_conf['download_dir']:
    os.environ['HF_HOME'] = str(ROOT_BASE_DIR_PATH / hf_conf['download_dir'])

import uvicorn
from main import start_app
from load_config.config import config

app = start_app()

if __name__ == '__main__':
    host = config['server']['host']
    port = config['server']['port']
    uvicorn.run(
        app,
        host=host,
        port=port,
        loop="asyncio",
        reload=False,
        log_level="warning",
    )
