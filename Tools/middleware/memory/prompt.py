"""
辅助与BalancedMultiDimensionMemory的提示词操作
"""
from langchain_core.documents import Document
from pydantic import Field, BaseModel


class SystemPrompt(BaseModel):
    core: str = Field(default="", description="核心系统提示词，一般是角色设定")
    profile: str = Field(default="", description="用户画像（半静态层，位于核心与记忆片段之间，避免破坏缓存前缀）")
    memory_fragments: list[Document] = Field(default=[], description="用户记忆片段列表")

    def __str__(self):
        # 拼接顺序按缓存前缀规则：静态核心（稳定）→ 用户画像（半静态）→ 记忆片段（每轮变化）
        # 变量内容越靠前，可复用的缓存前缀越短，缓存命中率越低
        fragments = [self.core]
        if self.profile:
            fragments.append(f"【当前用户画像】\n{self.profile}")
        if self.memory_fragments:
            fragments.append("以下为相关片段：")
        for fragment in self.memory_fragments:
            fragments.append(fragment.page_content)
        return "\n".join(fragments)


class SystemPromptOperation:
    def __init__(self, initial_prompt: str = ""):
        self.prompt = SystemPrompt()
        self.prompt.core = initial_prompt

    def add_memory(self, memory: Document) -> None:
        self.prompt.memory_fragments.append(memory)

    def add_memorys(self, memorys: list[Document]) -> tuple[list[Document], list[str]]:
        old_ids = []
        self.prompt.memory_fragments.clear()
        for memory in memorys:
            old_ids.append(memory.id)
            num_id = int(memory.metadata['strengthen_num'])
            new_id = num_id + 1
            memory.metadata['strengthen_num'] = new_id
        self.prompt.memory_fragments.extend(memorys)
        return memorys, old_ids

    def get_prompt(self) -> str:
        return str(self.prompt)
