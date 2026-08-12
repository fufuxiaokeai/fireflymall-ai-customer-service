import re
from typing import Optional

import yaml


class AnalysisYaml:
    def __init__(self, *,
                 yaml_file_path: Optional[str] = None,
                 yaml_dict: Optional[dict] = None):
        if yaml_dict is None:
            if yaml_file_path is None:
                raise ValueError("当未提供yaml_dict时，yaml_file_path不允许为None")

            with open(yaml_file_path, "r", encoding="utf-8") as f:
                yaml_dict = yaml.safe_load(f)

        self.yaml_dict = yaml_dict

    def analysis(self):

        def resolve_placeholders(config, config_root=None):
            """
            递归解析 config 中所有字符串的 ${path:default} 占位符。
            :param config: 当前处理的配置节点（dict / list / str 等）
            :param config_root: 整个配置树的根，用于查找引用路径（首次调用时传 None 即用 config）
            """
            if config_root is None:
                config_root = config

            if isinstance(config, dict):
                return {k: resolve_placeholders(v, config_root) for k, v in config.items()}
            elif isinstance(config, list):
                return [resolve_placeholders(item, config_root) for item in config]
            elif isinstance(config, str):
                # 匹配 ${path:default}，其中 path 可包含点号，default 为可选
                pattern = re.compile(r'\$\{([^}:]+)(?::([^}]*))?}')

                def replacer(match):
                    path = match.group(1)
                    default = match.group(2) if match.group(2) is not None else ''
                    # 按点号分隔路径，逐级从 config_root 中取值
                    keys = path.split('.')
                    value = config_root
                    try:
                        for key in keys:
                            value = value[key]
                        return str(value)
                    except (KeyError, TypeError):
                        return default

                return pattern.sub(replacer, config)
            else:
                return config

        yaml_config = resolve_placeholders(self.yaml_dict)
        return yaml_config
