"""流式路由 JSON 过滤器：打字机保留下的 JSON 泄漏防线。

本过滤器对每个"模型调用段"的流式增量做实时判断：
- 非 JSON 文本：原样直通（打字机零影响）
- 路由 JSON：抑制 node/file_url 等结构文本，只增量转发 answer 字段的值
  （转义还原），打字机效果保留；段结束时 flush() 兜底补发未转发内容。

用法（每段新建实例，段边界 = chat_node 重放消息）：
    f = RouteJsonStreamFilter()
    for chunk in chunks:
        visible = f.push(chunk)      # 返回本 chunk 应转发的文本
        ...
    leftover = f.flush()             # 段结束：返回应补发的文本
"""
import json
import re

_ESCAPE_MAP = {
    '"': '"',
    '\\': '\\',
    '/': '/',
    'n': '\n',
    'r': '\r',
    't': '\t',
    'b': '\b',
    'f': '\f',
}


def _strip_json_wrappers(text: str) -> str:
    """剥离 ```json 代码块 / 首尾引号包装，返回最内层候选 JSON（失败返回原文）"""
    s = text.strip()
    match = re.match(r'^```(?:json)?\s*(.*?)\s*```$', s, re.S)
    if match:
        s = match.group(1).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1].strip()
    return s


class RouteJsonStreamFilter:
    """流式增量过滤：只转发路由 JSON 中 answer 字段的值增量。

    状态机：
    - state: undecided（等待判定）→ plain（非 JSON，直通）| json（JSON 段）
    - json 段内 json_state: key（找 "answer" 键）→ after_key → after_colon
      → value（answer 值字符串内，增量转发）→ done（值结束，抑制后续）
    """

    def __init__(self, answer_key: str = 'answer'):
        self.escape = None
        self.pos = None
        self.json_state = None
        self.buf = None
        self.state = None
        self.answer_key = answer_key
        self.reset()

    def reset(self) -> None:
        self.state = 'undecided'
        self.buf = ''            # 未判定/JSON 段的累积缓冲
        self.json_state = 'key'  # 见类注释
        self.pos = 0             # 已扫描位置（buf 内游标）
        self.escape = False      # answer 值内是否处于转义状态

    def push(self, text: str) -> str:
        """输入流式增量，返回应转发给用户的文本（answer 值增量或原文直通）"""
        if not text:
            return ''
        if self.state == 'undecided':
            combined = self.buf + text
            if combined.lstrip().startswith('{'):
                # 疑似路由 JSON 段：进入抑制模式，从累积缓冲开始扫描
                self.state = 'json'
                self.buf = combined
                return self._scan_json()
            # 非 JSON：直通（连同之前累积的空白等一并补发）
            self.state = 'plain'
            self.buf = ''
            return combined
        if self.state == 'plain':
            return text
        # json 段：累积后扫描增量
        self.buf += text
        return self._scan_json()

    def _scan_json(self) -> str:
        """扫描 buf 内 pos 之后的内容，返回可转发的增量文本；异常结构不推进游标"""
        out: list[str] = []
        i = self.pos
        n = len(self.buf)
        while i < n:
            c = self.buf[i]
            if self.json_state == 'key':
                # 找 "answer" 键（pydantic 字段序列化顺序固定，直接 find 即可；
                # 增量边界处键名不完整时 find 失败，不推进游标等下一 chunk）
                idx = self.buf.find(f'"{self.answer_key}"', i)
                if idx == -1:
                    break
                i = idx + len(self.answer_key) + 2
                self.json_state = 'after_key'
            elif self.json_state == 'after_key':
                if c in ' \t\n\r':
                    i += 1
                    continue
                if c == ':':
                    self.json_state = 'after_colon'
                    i += 1
                    continue
                break  # 结构异常，等 flush 兜底
            elif self.json_state == 'after_colon':
                if c in ' \t\n\r':
                    i += 1
                    continue
                if c == '"':
                    self.json_state = 'value'
                    i += 1
                    continue
                break  # answer 不是字符串（异常），等 flush 兜底
            elif self.json_state == 'value':
                while i < n:
                    c2 = self.buf[i]
                    if self.escape:
                        out.append(_ESCAPE_MAP.get(c2, c2))
                        self.escape = False
                        i += 1
                        continue
                    if c2 == '\\':
                        self.escape = True
                        i += 1
                        continue
                    if c2 == '"':
                        self.json_state = 'done'
                        break
                    out.append(c2)
                    i += 1
            else:  # done：answer 已结束，后续结构（file_url 等）全部抑制
                break
        self.pos = i
        return ''.join(out)

    def flush(self) -> str:
        """段结算：返回应补发的文本。

        正常路径（answer 完整增量转发）返回 ''。异常路径兜底：
        - JSON 完整但 answer 未走完（理论上不发生，防御）：解析后补发完整 answer
        - 畸形 JSON：补发缓冲原文（内容优先，宁可见畸形文本不吞内容）
        """
        leftover = ''
        if self.state == 'json' and self.buf.strip():
            buf = self.buf
            if self.json_state != 'done':
                answer = None
                for candidate in (buf, _strip_json_wrappers(buf)):
                    try:
                        data = json.loads(candidate)
                    except Exception:
                        continue
                    if isinstance(data, dict) and isinstance(data.get('answer'), str):
                        answer = data['answer']
                        break
                leftover = answer if answer is not None else buf
        self.reset()
        return leftover
