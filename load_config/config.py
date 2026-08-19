import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from load_config.analysis_yaml_file import AnalysisYaml


def get_root_path():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    # 以 load_config 模块自身位置为锚（load_config/ 恒在仓库根，向上取一级即仓库根）。
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


_BASE_DIR = get_root_path()  # 只在本文件中使用

ROOT_BASE_DIR_PATH = Path(_BASE_DIR).resolve()  # 提供给其他模块使用

load_dotenv(os.path.join(_BASE_DIR, '.env'))


def is_absolute_path(path: str) -> bool:
    return Path(path).is_absolute()


def get_first_db_by_conf(db_conf):
    if not db_conf:
        return None, None
    first_db, first_value = next(iter(db_conf.items()))
    return first_db, first_value.copy()


def read_sub_agent_prompt(file_url: str):
    file_path = Path(file_url)
    if not file_path.is_file():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    if file_path.suffix != '.md':
        raise ValueError(f"文件格式错误: {file_path}")

    return file_path.read_text(encoding='utf-8')


with open(os.path.join(_BASE_DIR, 'config.yaml'), 'r', encoding='utf-8') as f:
    _config = yaml.safe_load(f)
    analysis_yaml = AnalysisYaml(yaml_dict=_config)
    config = analysis_yaml.analysis()

if __name__ == '__main__':
    # text = read_sub_agent_prompt(
    #     r'D:\pycharm\pycharm study\intelligent_customer_service\agent\prompt\after_sales_service_prompt.md')
    # print(text)
    pass
