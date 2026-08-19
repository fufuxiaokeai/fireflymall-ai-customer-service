"""
ONNX 推理 + 三级分级 + 恶意值 Redis 存储（全原子操作）。

定稿参数（tokenizers 0.22 / ONNX 推理）（相关公式参考与记忆中间件的公式）：
  分级:  score < 0.05 放行 | 0.05~0.2 疑似(仅旁白) | 0.2~0.9 明确(加分) | >=0.9 硬拦(加分)
  恶意值: 明确 +0.2·(1+0.3n) / 硬拦 +0.5·(1+0.3n)，n=累计注入次数；τ=24h 指数衰减（同记忆中间件族公式）
  禁言:  malice ≥ 1.0 → T(n) = 30min·2^(n-1) + 转人工标记；禁言后恶意值清零重新累计
  Redis 键:  malice:{user_id} = {score(整数×100), hits, mutes, mute_until, ts}

原子性：加分+衰减+禁言判断合并为单个 Lua 脚本（Redis 单线程 + EVAL 原子），
高并发下不会多加/漏加；禁言状态查询用 HGET（原子读）。
"""
import asyncio
import math
import time
from pathlib import Path

import onnxruntime as ort
import redis.asyncio as redis
from tokenizers import Tokenizer

from load_config.config import config

ROOT = Path(__file__).resolve().parent
ONNX_PATH = ROOT / 'roberta_inj.onnx'
HF_SNAPSHOT = ROOT / 'hfl' / 'chinese-roberta-wwm-ext' / 'hub' / 'models--hfl--chinese-roberta-wwm-ext' / 'snapshots'

MAX_LEN = 128
THR_SUSPECT, THR_WARN, THR_HARD = 0.05, 0.2, 0.9

# ---------------- 恶意值参数（×100 整数化，INCR 友好） ----------------
MALICE_ADD_WARN = 20      # 明确风险加分 0.2
MALICE_ADD_HARD = 50      # 硬拦加分 0.5
MALICE_MUTE_AT = 100      # 禁言阈值 1.0
TAU_HOURS = 24            # 衰减时间尺度
MUTE_BASE_MIN = 30        # 首次禁言分钟数，之后 2^(n-1) 翻倍

# 加分 + 惰性衰减 + 禁言判断，单脚本原子完成。
# ARGV[1]=增量基数(20/50) ARGV[2]=当前时间戳(秒) 返回 {score, hits, mutes, mute_until}
_ADD_AND_DECAY_LUA = f"""
local n = tonumber(redis.call('HGET', KEYS[1], 'hits') or '0')
local inc = math.floor(tonumber(ARGV[1]) * (1 + 0.3 * n))
local score = tonumber(redis.call('HGET', KEYS[1], 'score') or '0')
local ts = tonumber(redis.call('HGET', KEYS[1], 'ts') or '0')
local now = tonumber(ARGV[2])
if ts > 0 then
    local dt = (now - ts) / 3600.0
    if dt > 0 then
        score = math.floor(score * math.exp(-dt / {TAU_HOURS}))
        if score < 0 then score = 0 end
    end
end
score = score + inc
redis.call('HINCRBY', KEYS[1], 'hits', 1)
local mutes = tonumber(redis.call('HGET', KEYS[1], 'mutes') or '0')
local mute_until = 0
if score >= {MALICE_MUTE_AT} then
    mutes = mutes + 1
    mute_until = now + {MUTE_BASE_MIN * 60} * math.pow(2, mutes - 1)
    -- HSET 单字段两次调用：兼容 Redis <4.0（本机环境约束，多字段语法会报错）
    redis.call('HSET', KEYS[1], 'mutes', mutes)
    redis.call('HSET', KEYS[1], 'mute_until', mute_until)
    score = 0  -- 禁言后清零，重新累计
end
redis.call('HSET', KEYS[1], 'score', score)
redis.call('HSET', KEYS[1], 'ts', now)
return {{score, n + 1, mutes, mute_until}}
"""


class InjectionDetector:
    """注入检测器：单例使用（模型/Redis 连接懒加载）"""

    def __init__(self):
        self._tok: Tokenizer | None = None
        self._sess: ort.InferenceSession | None = None
        self._redis: redis.Redis | None = None

    # ---------------- 模型 ----------------
    def _ensure_model(self):
        if self._sess is not None:
            return
        if not ONNX_PATH.exists():
            raise FileNotFoundError(f'缺少模型文件: {ONNX_PATH}（需从 Kaggle 导出后放入 model/）')
        snaps = sorted(HF_SNAPSHOT.iterdir(), key=lambda p: p.stat().st_mtime)
        if not snaps:
            raise FileNotFoundError(f'缺少 tokenizer 缓存: {HF_SNAPSHOT}')
        self._tok = Tokenizer.from_file(str(snaps[-1] / 'tokenizer.json'))
        self._tok.enable_truncation(max_length=MAX_LEN)
        self._tok.enable_padding(pad_id=0, pad_token='[PAD]')
        self._sess = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])

    def score(self, text: str) -> float:
        """注入概率（0~1），单条推理"""
        self._ensure_model()
        enc = self._tok.encode(text)
        import numpy as np
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        logits = self._sess.run(['logits'], {
            'input_ids': ids, 'attention_mask': mask, 'token_type_ids': np.zeros_like(ids)})[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return float(e[0, 1] / e.sum(axis=1))

    def classify(self, text: str) -> tuple[str, float]:
        """返回 (action, score)；action: pass/suspect/warn/hard"""
        s = self.score(text)
        if s >= THR_HARD:
            return 'hard', s
        if s >= THR_WARN:
            return 'warn', s
        if s >= THR_SUSPECT:
            return 'suspect', s
        return 'pass', s

    # ---------------- 恶意值（Redis，全原子） ----------------
    def _ensure_redis(self):
        if self._redis is not None:
            return
        redis_conf = config.get('redis', {})
        pool = redis.ConnectionPool(
            host=redis_conf.get('host', 'localhost'),
            port=redis_conf.get('port', 6379),
            db=redis_conf.get('db', 0),
            password=redis_conf.get('password', None),
            decode_responses=True,
        )
        self._redis = redis.Redis(connection_pool=pool)

    async def is_muted(self, user_id: str) -> int:
        """返回禁言截止时间戳（0=未禁言）。原子读。"""
        self._ensure_redis()
        until = await self._redis.hget(f'malice:{user_id}', 'mute_until')
        return int(until or 0)

    async def add_malice(self, user_id: str, action: str) -> dict:
        """注入命中后更新恶意值（原子 Lua：加分+衰减+禁言判断）。
        返回 {score, hits, mutes, mute_until}（score/mute_until 为更新后值）"""
        self._ensure_redis()
        base = MALICE_ADD_HARD if action == 'hard' else MALICE_ADD_WARN
        result = await self._redis.eval(
            _ADD_AND_DECAY_LUA, 1, f'malice:{user_id}', base, math.floor(time.time()))
        score, hits, mutes, mute_until = (int(x) for x in result)
        return {'score': score, 'hits': hits, 'mutes': mutes, 'mute_until': mute_until}

    # ---------------- 总入口 ----------------
    async def check(self, user_id: str, text: str) -> dict:
        """完整检测流程：禁言检查 → 分类 → 恶意值更新。
        返回 {action, score, malice, mute_until, mutes}；action: pass/suspect/warn/hard/banned"""
        mute_until = await self.is_muted(user_id)
        if mute_until > time.time():
            return {'action': 'banned', 'score': 0.0, 'malice': 0, 'mute_until': mute_until, 'mutes': 0}
        # 模型推理 ~60ms，放线程池避免阻塞事件循环
        action, score = await asyncio.to_thread(self.classify, text)
        malice = 0
        if action in ('warn', 'hard'):
            upd = await self.add_malice(user_id, action)
            malice, mute_until = upd['score'], upd['mute_until']
        return {'action': action, 'score': score, 'malice': malice,
                'mute_until': mute_until, 'mutes': 0}


# 进程级单例（FastAPI 生命周期内复用模型与连接池）
detector = InjectionDetector()
