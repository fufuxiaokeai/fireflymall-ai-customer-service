# -*- coding: utf-8 -*-
"""
检索召回率对比评估：向量-only vs BM25-only vs Ensemble(BM25+向量)

- 打分方式：RAGAS context_recall / context_precision（LLM 判定，deepseek-v4-flash）
- 同时输出无 LLM 的精确匹配指标：hit@5（top5 中命中的 gold 数）、recall@5、precision@5
- ground truth 用"名称关键词命中"构建（词面 gold，对 BM25 略有利；语义查询对向量有利，两者混合取平衡）

用法：python tests/eval_retrieval_hybrid.py
"""
import json
import os
import sys
from pathlib import Path
from typing import Any, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 仓库根，保证 Tools/load_config 可导入

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / '.env')

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic import Field
from langchain_classic.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI

from Tools.product_process import _es_client, _es_rag, _es_config

TOPK = 5          # 每个配置取前 5 条做评估（贴近工具实际返回）
FETCH = 20        # 各检索器内部候选数，融合后截取 TOPK

# 查询集与人工标注 gold 从 tests/data/eval_gold.json 读取（词面查询与语义改写查询混合）
GOLD_FILE = Path(__file__).resolve().parent / 'data' / 'eval_gold.json'


class ESBM25Retriever(BaseRetriever):
    """ES 原生 BM25 检索器（match 查询 name 字段），支持 filter 与 k。
    langchain_community 自带的 ElasticSearchBM25Retriever 硬编码 content 字段且无 filter/k，
    不可用于 goods 索引，故自写。"""
    client: Any
    index_name: str
    k: int = TOPK
    filter_clauses: List[dict] = Field(default_factory=list)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        res = self.client.search(
            index=self.index_name,
            query={'bool': {'must': [{'match': {'name': query}}] + self.filter_clauses}},
            size=self.k,
            source=['name', 'id'],
        )
        return [
            Document(page_content=h['_source'].get('name', ''),
                     metadata={'id': h['_source'].get('id')})
            for h in res['hits']['hits']
        ]


def load_gold() -> dict:
    """从 eval_gold.json 加载人工标注 gold：{查询: {"category":..., "relevant": [{"id","name"}...]}}"""
    with open(GOLD_FILE, encoding='utf-8') as f:
        return json.load(f)


def main():
    knn_filter = [{'term': {'status': 1}}, {'term': {'isAD': False}}]

    # ---- 三种检索配置 ----
    vec_retriever = VectorStoreRetriever(
        vectorstore=_es_rag, search_type='similarity',
        search_kwargs={'k': FETCH, 'filter': knn_filter, 'fields': ['id']},
    )
    bm25_retriever = ESBM25Retriever(
        client=_es_client, index_name=_es_config.get('index_name'),
        k=FETCH, filter_clauses=knn_filter,
    )
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vec_retriever],
        weights=[0.5, 0.5], c=60, id_key='id',
    )

    configs = {
        'vector': vec_retriever,
        'bm25': bm25_retriever,
        'ensemble': ensemble,
    }

    # ---- 逐查询检索 ----
    rows = []
    gold_data = load_gold()
    for q, info in gold_data.items():
        gold_items = info['relevant']
        gold_ids = {g['id'] for g in gold_items}
        # reference 覆盖全部相关文档（gold 已人工标注、可穷举）
        gold_names = '；'.join(g['name'][:80] for g in gold_items)
        print(f"[{q}] gold 数量: {len(gold_items)}")
        for cfg_name, retriever in configs.items():
            docs = retriever.invoke(q)[:TOPK]
            ids = [d.metadata.get('id') for d in docs]
            hits = len([i for i in ids if i in gold_ids])
            hit_rate = hits / len(gold_ids) if gold_ids else 0
            rows.append({
                'config': cfg_name, 'query': q,
                'contexts': [d.page_content for d in docs],
                'ground_truth': gold_names,
                'hit@5': hits, 'recall@5': hit_rate,
                'precision@5': hits / max(len(ids), 1),
            })

    # ---- 无 LLM 指标汇总 ----
    print('\n===== 精确匹配指标（无 LLM） =====')
    for cfg in configs:
        sub = [r for r in rows if r['config'] == cfg]
        print(f"{cfg:10s} 平均 hit@5: {sum(r['hit@5'] for r in sub)/len(sub):.1f}  "
              f"recall@5: {sum(r['recall@5'] for r in sub)/len(sub):.4f}  "
              f"precision@5: {sum(r['precision@5'] for r in sub)/len(sub):.3f}")

    # ---- RAGAS 打分 ----
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import context_precision, context_recall

    judge_llm = LangchainLLMWrapper(ChatOpenAI(
        model='deepseek-v4-flash',
        api_key=os.getenv('DEEPSEEK_API_KEY'),
        base_url=os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com'),
    ))
    print('\n===== RAGAS 打分（LLM 判定，需几分钟） =====')
    for cfg in configs:
        sub = [r for r in rows if r['config'] == cfg]
        dataset = EvaluationDataset(samples=[
            SingleTurnSample(user_input=r['query'], retrieved_contexts=r['contexts'],
                             reference=r['ground_truth'])
            for r in sub
        ])
        result = evaluate(dataset=dataset, metrics=[context_recall, context_precision], llm=judge_llm)
        # ragas 0.4.x 的 evaluate 返回逐样本分数列表，取均值
        recalls = result['context_recall']
        precisions = result['context_precision']
        print(f"{cfg:10s} context_recall: {sum(recalls)/len(recalls):.4f}  "
              f"context_precision: {sum(precisions)/len(precisions):.4f}")


if __name__ == '__main__':
    main()
