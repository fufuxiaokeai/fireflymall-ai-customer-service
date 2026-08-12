"""
一个专门用于计算token数量的文件
可用类：
    TokenCalculatorFactory: 一个工厂类，用于创建不同的token计算器实例
"""
import tiktoken
import torch
import wordninja
from typing import Optional, Tuple, Iterable
from tiktoken.model import MODEL_TO_ENCODING, MODEL_PREFIX_TO_ENCODING

from langchain.agents.middleware.summarization import TokenCounter

from langchain_core.messages import MessageLikeRepresentation
from transformers import AutoTokenizer, TokenizersBackend, SentencePieceBackend

from Tools.log_settings import LogSetting

logger = LogSetting.create(logger_name=__name__)

_PROVIDER_MAPPING: dict[str, str] = {
    'deepseek': 'deepseek-ai',  # deepseek
    'meta': 'meta-llama',  # Llama 模型
    'openai': 'openai',  # OpenAI 模型
    'qwen': 'Qwen',  # 千问模型
    'google': 'google',  # gemma, gemma-2, flan-t5 等
    'mistral': 'mistralai',  # Mistral-7B, Mixtral-8x7B, Mistral-Nemo 等
    'microsoft': 'microsoft',  # Phi-3, Florence, DeepSpeed 等
    'tii': 'tiiuae',  # Falcon 系列（阿联酋）
    'stabilityai': 'stabilityai',  # Stable Diffusion, Stable LM 等
    'baichuan': 'baichuan-inc',  # 百川模型 Baichuan2, Baichuan-M1 等
    '01-ai': '01-ai',  # Yi 系列（零一万物）
    'cohere': 'CohereForAI',  # Command-R, Aya 等
    'nous': 'NousResearch',  # Hermes, Capybara 等 Llama 微调模型
    'upstage': 'upstage',  # Solar 系列
}

_TOKENIZER_MAPPING: dict[str, TokenizersBackend | SentencePieceBackend] = {}


def _default_token_counter() -> TokenCounter:
    def count_token(messages: Iterable[MessageLikeRepresentation]) -> int:
        total = 0
        for message in messages:
            total += len(message.content) / 3.3
        return total

    return count_token


def _tiktoken_counter(model: str) -> TokenCounter:
    encoding = tiktoken.encoding_for_model(model)

    def count_token(messages: Iterable[MessageLikeRepresentation]) -> int:
        total_tokens = 0
        for message in messages:
            total_tokens += len(encoding.encode(message.content))
        return total_tokens

    return count_token


def _huggingface_token_counter(
        hf_id: str,
        use_fast: bool = True,
        add_special_tokens: bool = False,
) -> TokenCounter:
    """
    专门的token计算方法，用于计算huggingface模型的token数
    :param hf_id: huggingface 模型id
    :param use_fast: 是否使用快速分词器
    :param add_special_tokens: 是否添加特殊token（如 <s>、</s> 等）
    :return: token数
    """
    if hf_id not in _TOKENIZER_MAPPING:
        _TOKENIZER_MAPPING[hf_id] = AutoTokenizer.from_pretrained(
            hf_id,
            use_fast=use_fast,
            padding_side='right',
            trust_remote_code=True
        )
    tokenizer = _TOKENIZER_MAPPING[hf_id]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer.device = device

    def count_token(messages: Iterable[MessageLikeRepresentation]) -> int:
        total_tokens = 0
        for message in messages:
            token_ids = tokenizer.encode(message.content,
                                         add_special_tokens=add_special_tokens,
                                         return_tensors="pt"
                                         ).to(device)  # type: ignore
            total_tokens += token_ids.shape[1]  # type: ignore
        return total_tokens

    return count_token


def to_pascal_case_with_big_camel(name: str) -> str:
    """
    将字符串转换为大驼峰命名法，每个单词之间用短横线分隔
    :param name: 模型名称
    :return: 转换后的字符串
    """
    if '-' in name:
        split_list = name.split('-')
    else:
        split_list = [name]
    big_camels = []
    for s in split_list:
        parts = wordninja.split(s)
        big_camels.append(''.join(p.capitalize() for p in parts))
    return '-'.join(big_camels)


def check_model_support(hf_id: str, use_fast: bool = True) -> bool:
    """
    检查模型是否支持 huggingface
    :param hf_id: huggingface 模型id
    :param use_fast: 是否使用快速分词器
    :return: 是否支持，映射后的模型名称
    """
    try:
        AutoTokenizer.from_pretrained(
            hf_id,
            use_fast=use_fast,
            local_files_only=False,
            resume_download=False,
            trust_remote_code=True
        )
        return True
    except Exception as e:
        logger.error(f"检查模型 {hf_id} 是否支持 huggingface 时出错: {e}")
        return False


class TokenCalculatorFactory:
    """
        专门的token计算工厂类
        将会根据不同的模型类型来调用不同的token计算方法
        计算方法将会从 用户提供、 huggingface、 tokenizer 中获取
        若模型类型不在工厂类中，不会报错，但是会进行粗略计算

        支持用户自己传入计算方式，若为提供 则将会：
            先从 tokenizer 中获取，若不存在，再从 huggingface 中获取
            若都不存在则会使用默认的计算方式：字数 / 3.3 约等于 1 token
    """

    def __init__(
            self,
            token_counter: Optional[TokenCounter] = None,
            model_name: Optional[str] = None,
            model_huggingface_id: Optional[str] = None,
    ):
        try:
            import wordninja
            import tiktoken
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "tiktoken 库、wordninja 库和 transformers 库未安装，无法进行 token计算"
                "请先运行 `pip install wordninja tiktoken transformers` 安装依赖"
            )
        self.model_name = model_name
        self.model_huggingface_id = model_huggingface_id
        if token_counter is None:
            self.token_counter = self._get_token_counter()
        else:
            self.token_counter = token_counter

    def calculate_tokens(self, messages: Iterable[MessageLikeRepresentation]) -> int:
        return self.token_counter(messages)

    def _get_token_counter(self) -> TokenCounter:
        if self.model_huggingface_id:
            return _huggingface_token_counter(self.model_huggingface_id)
        if self.model_name is None:
            return _default_token_counter()
        model = self.model_name
        if ":" in model:
            # 获取模型名称
            model = model.split(":", 1)[1]
        # 判断 tiktoken 是否支持该模型
        if self._is_tiktoken_model(model):
            return _tiktoken_counter(model)
        # 判断 huggingface 是否支持该模型
        is_hf, hf_id = self._is_huggingface_model()
        if is_hf and check_model_support(hf_id):
            return _huggingface_token_counter(hf_id)
        # 若都不存在则会使用默认的计算方式
        return _default_token_counter()

    @staticmethod
    def _is_tiktoken_model(model: str) -> bool:
        # 精准匹配
        if model in MODEL_TO_ENCODING:
            return True
        # 前缀匹配
        for prefix in MODEL_PREFIX_TO_ENCODING.keys():
            if model.startswith(prefix):
                return True
        logger.warning(f"模型 {model} 不支持 tiktoken，尝试从 transformers 中获取")
        return False

    def _is_huggingface_model(self) -> Tuple[bool, str]:
        name = self.model_name
        prefix = None
        if ':' in name:
            prefix, name = name.split(':', 1)

        return self._map_custom_prefix_to_hf_id(prefix, name)

    def _map_custom_prefix_to_hf_id(self, prefix: str, name: str) -> Tuple[bool, str]:
        """
        将自定义前缀映射为 huggingface 模型名称
        :param prefix: 提供商
        :param name: 模型名称
        :return: 是否成功映射，映射后的模型名称
        """
        if self.model_huggingface_id is not None:
            return True, self.model_huggingface_id
        # 提供商映射
        if prefix in _PROVIDER_MAPPING:
            return True, _PROVIDER_MAPPING[prefix] + '/' + to_pascal_case_with_big_camel(name)
        # 模型名称模糊匹配
        for model in _PROVIDER_MAPPING.keys():
            if name.startswith(model):
                return True, _PROVIDER_MAPPING[model] + '/' + to_pascal_case_with_big_camel(name)

        logger.warning(f"模型 {name} 不支持 huggingface, 请确认模型是否正确，"
                       "或者在 config.yaml 中配置 model.chat.huggingface_id 准确指出 huggingface 的名称"
                       "已为你使用默认的计算方式")
        return False, ""
