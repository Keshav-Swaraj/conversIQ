"""Retriever Module - Converted from Notebooks 02 & 03."""

import re
import time
from typing import List, Dict, Any, Tuple, Optional
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from .document import HierarchicalChunk

# Shared CrossEncoder singleton cache to prevent reloading model weights
_RERANKER_CACHE: Dict[str, CrossEncoder] = {}


def get_shared_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    """Retrieve or load shared CrossEncoder singleton model instance."""
    if model_name not in _RERANKER_CACHE:
        _RERANKER_CACHE[model_name] = CrossEncoder(model_name)
    return _RERANKER_CACHE[model_name]


class BM25RerankRetriever:
    """Sparse BM25 retrieval combined with CrossEncoder reranking & Dynamic Context Compression."""

    def __init__(
        self,
        chunks: List[HierarchicalChunk],
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        bm25_k: int = 15,
        rerank_top_n: int = 3
    ):
        self.bm25_k = bm25_k
        self.rerank_top_n = rerank_top_n
        
        self.documents = [
            Document(
                page_content=c.content,
                metadata={
                    "chunk_id": c.chunk_id,
                    "section": c.section,
                    "paragraph_index": c.paragraph_index,
                    "word_count": c.word_count
                }
            ) for c in chunks
        ]
        
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = bm25_k
        self.reranker = get_shared_reranker(reranker_model_name)

    @staticmethod
    def determine_dynamic_context_limit(query: str, default_top_n: int = 3) -> Tuple[int, str]:
        """Intelligently limit context chunks based on query intent to minimize prompt overhead."""
        q_lower = query.lower().strip()
        
        # Summary or broad overview -> allow top 3 to 5 chunks
        if any(w in q_lower for w in ["summarize", "summary", "overview", "architecture", "main points", "explain all"]):
            return min(5, max(3, default_top_n)), "SUMMARY"
            
        # Comparison -> top 2 chunks
        if any(w in q_lower for w in ["compare", "versus", " vs ", "difference", "differ"]):
            return 2, "COMPARISON"
            
        # Factual / definition / specific entity -> top 2 chunks for lean prompt & high recall
        if any(q_lower.startswith(w) for w in ["who is", "what is", "when did", "where is", "define", "how fast", "how much", "latency"]):
            return 2, "FACTUAL"
            
        return default_top_n, "STANDARD"

    def retrieve_bm25(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Retrieve candidate documents using sparse BM25 retrieval."""
        self.bm25_retriever.k = k or self.bm25_k
        return self.bm25_retriever.invoke(query)

    def retrieve_and_rerank(
        self,
        query: str,
        k: Optional[int] = None,
        top_n: Optional[int] = None,
        dynamic_compression: bool = True
    ) -> Tuple[List[Dict[str, Any]], float]:
        """BM25 sparse candidate retrieval followed by CrossEncoder semantic reranking with dynamic context compression."""
        results, timings = self.retrieve_and_rerank_with_timings(query, k=k, top_n=top_n, dynamic_compression=dynamic_compression)
        return results, timings.get("retrieval_total_ms", 0.0)

    def retrieve_and_rerank_with_timings(
        self,
        query: str,
        k: Optional[int] = None,
        top_n: Optional[int] = None,
        dynamic_compression: bool = True
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Retrieve and rerank returning detailed micro-profiling timings dictionary."""
        timings: Dict[str, float] = {}
        
        # Determine dynamic top_n limit based on query intent if enabled
        if dynamic_compression and top_n is None:
            effective_top_n, intent = self.determine_dynamic_context_limit(query, self.rerank_top_n)
        else:
            effective_top_n = top_n or self.rerank_top_n
            intent = "MANUAL"

        # 1. BM25 Sparse Candidate Search
        t_bm25 = time.perf_counter()
        candidates = self.retrieve_bm25(query, k=k or self.bm25_k)
        bm25_latency_ms = (time.perf_counter() - t_bm25) * 1000.0
        timings["bm25_ms"] = bm25_latency_ms
        
        if not candidates:
            timings["rerank_ms"] = 0.0
            timings["retrieval_total_ms"] = bm25_latency_ms
            return [], timings
            
        # 2. CrossEncoder Batch Reranking
        t_rerank = time.perf_counter()
        pairs = [[query, doc.page_content] for doc in candidates]
        # Batch predict all candidate pairs at once
        scores = self.reranker.predict(pairs, batch_size=len(pairs))
        
        scored_docs = []
        for doc, score in zip(candidates, scores):
            scored_docs.append({
                "content": doc.page_content,
                "score": float(score),
                "chunk_id": doc.metadata["chunk_id"],
                "section": doc.metadata["section"],
                "metadata": doc.metadata,
                "intent": intent
            })
            
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        timings["rerank_ms"] = (time.perf_counter() - t_rerank) * 1000.0
        timings["retrieval_total_ms"] = timings["bm25_ms"] + timings["rerank_ms"]
        
        return scored_docs[:effective_top_n], timings

