"""Full LangGraph Pipeline Module - Converted from Notebook 06."""

import os
import time
from typing import List, Dict, Any, TypedDict, Optional, Generator, Tuple
from langgraph.graph import StateGraph, START, END

from .document import PDFDocumentProcessor, HierarchicalChunk
from .retriever import BM25RerankRetriever
from .llm import LLMGenerator
from .memory import SlidingWindowMemoryManager


class RAGState(TypedDict):
    """Lightweight query-specific LangGraph state (heavy objects are held in pipeline session)."""
    query: str
    history: str
    context_str: str
    reranked_results: List[Dict[str, Any]]
    answer: str
    error: Optional[str]
    timings: Dict[str, float]


class DocumentSession:
    """Persistent session holding loaded document chunks, BM25 retriever index, and ingestion metrics."""

    def __init__(self, pdf_path: str, chunks: List[HierarchicalChunk], retriever: BM25RerankRetriever, ingestion_timings: Dict[str, float]):
        self.pdf_path = os.path.abspath(pdf_path)
        self.chunks = chunks
        self.retriever = retriever
        self.ingestion_timings = ingestion_timings


class ConverseIQPipeline:
    """Orchestrates document loading, retrieval, reranking, generation, and memory using LangGraph & DocumentSession caching."""

    def __init__(
        self,
        ollama_model: str = "llama3",
        ollama_host: str = "http://localhost:11434",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        memory_k: int = 5
    ):
        self.doc_processor = PDFDocumentProcessor()
        self.llm_gen = LLMGenerator(model_name=ollama_model, base_url=ollama_host)
        self.memory = SlidingWindowMemoryManager(k=memory_k)
        self.reranker_model = reranker_model
        
        self.active_session: Optional[DocumentSession] = None
        self._build_graph()

    def load_document(self, pdf_path: str) -> DocumentSession:
        """Load, parse, and build BM25 retriever index for a PDF document once, caching it in active_session."""
        abs_path = os.path.abspath(pdf_path)
        if self.active_session and self.active_session.pdf_path == abs_path:
            return self.active_session

        chunks, ingestion_timings = self.doc_processor.process_with_timings(abs_path)
        
        t_retriever = time.perf_counter()
        retriever = BM25RerankRetriever(chunks, reranker_model_name=self.reranker_model)
        ingestion_timings["retriever_init_ms"] = (time.perf_counter() - t_retriever) * 1000.0

        session = DocumentSession(
            pdf_path=abs_path,
            chunks=chunks,
            retriever=retriever,
            ingestion_timings=ingestion_timings
        )
        self.active_session = session
        return session

    def get_or_create_session(self, pdf_path: str) -> DocumentSession:
        """Retrieve active document session or load PDF if not already cached."""
        abs_path = os.path.abspath(pdf_path)
        if self.active_session and self.active_session.pdf_path == abs_path:
            return self.active_session
        return self.load_document(abs_path)

    def _build_graph(self):
        """Construct lightweight LangGraph StateGraph pipeline."""
        
        def node_retrieve_rerank(state: RAGState) -> RAGState:
            t0 = time.perf_counter()
            if not self.active_session:
                return {**state, "reranked_results": [], "error": "No active document session."}

            results, retrieval_timings = self.active_session.retriever.retrieve_and_rerank_with_timings(state["query"])
            context_str, prompt_construction_ms = self.llm_gen.format_context_string(results)

            timings = state.get("timings", {})
            timings.update(retrieval_timings)
            timings["prompt_construction_ms"] = prompt_construction_ms

            return {
                **state,
                "reranked_results": results,
                "context_str": context_str,
                "timings": timings
            }

        def node_generate(state: RAGState) -> RAGState:
            t0 = time.perf_counter()
            if state.get("error"):
                return state

            answer, gen_timings = self.llm_gen.generate_with_timings(
                question=state["query"],
                context=state.get("context_str", "No context found."),
                history=state.get("history", "None")
            )

            timings = state.get("timings", {})
            timings.update(gen_timings)
            
            return {
                **state,
                "answer": answer,
                "timings": timings
            }

        workflow = StateGraph(RAGState)
        workflow.add_node("retrieve_rerank", node_retrieve_rerank)
        workflow.add_node("generate", node_generate)

        workflow.add_edge(START, "retrieve_rerank")
        workflow.add_edge("retrieve_rerank", "generate")
        workflow.add_edge("generate", END)

        self.app = workflow.compile()

    def query(self, pdf_path_or_query: str, user_query: Optional[str] = None) -> Dict[str, Any]:
        """Execute query pipeline with micro-profiling and backward-compatible signature.
        
        Supports both:
          - pipeline.query(pdf_path, query_str)
          - pipeline.query(query_str) [uses active_session]
        """
        t_total_0 = time.perf_counter()
        
        if user_query is not None:
            pdf_path = pdf_path_or_query
            query_text = user_query
        else:
            if not self.active_session:
                raise ValueError("No active document session. Call pipeline.load_document(pdf_path) first.")
            pdf_path = self.active_session.pdf_path
            query_text = pdf_path_or_query

        # Ensure session is active (caches parsed PDF & retriever)
        session = self.get_or_create_session(pdf_path)

        initial_state: RAGState = {
            "query": query_text,
            "history": self.memory.get_formatted_history(),
            "context_str": "",
            "reranked_results": [],
            "answer": "",
            "error": None,
            "timings": {}
        }

        final_state = self.app.invoke(initial_state)
        answer = final_state.get("answer", "")
        self.memory.add_turn(query_text, answer)

        timings = final_state.get("timings", {})
        timings["total_latency_ms"] = (time.perf_counter() - t_total_0) * 1000.0
        
        # Include ingestion timings if available
        if session.ingestion_timings:
            timings["pdf_loading_ms"] = session.ingestion_timings.get("pdf_loading_ms", 0.0)
            timings["text_cleaning_ms"] = session.ingestion_timings.get("text_cleaning_ms", 0.0)
            timings["text_chunking_ms"] = session.ingestion_timings.get("text_chunking_ms", 0.0)
            timings["hierarchy_formatting_ms"] = session.ingestion_timings.get("hierarchy_formatting_ms", 0.0)

        result_dict = {
            "answer": answer,
            "reranked_results": final_state.get("reranked_results", []),
            "history": self.memory.get_formatted_history(),
            "timings": timings
        }

        self.print_latency_table(query_text, timings)
        return result_dict

    def stream_query(self, pdf_path_or_query: str, user_query: Optional[str] = None) -> Generator[str, None, None]:
        """Stream generated answer tokens in real-time."""
        if user_query is not None:
            pdf_path = pdf_path_or_query
            query_text = user_query
        else:
            if not self.active_session:
                raise ValueError("No active document session. Call pipeline.load_document(pdf_path) first.")
            pdf_path = self.active_session.pdf_path
            query_text = pdf_path_or_query

        session = self.get_or_create_session(pdf_path)
        
        # Retrieve & Rerank
        results, _ = session.retriever.retrieve_and_rerank(query_text)
        context_str, _ = self.llm_gen.format_context_string(results)
        history_str = self.memory.get_formatted_history()

        collected_tokens = []
        for token in self.llm_gen.stream(question=query_text, context=context_str, history=history_str):
            collected_tokens.append(token)
            yield token

        full_answer = "".join(collected_tokens).strip()
        self.memory.add_turn(query_text, full_answer)

    @staticmethod
    def print_latency_table(query: str, timings: Dict[str, float]):
        """Print a clean ASCII latency profiling breakdown table."""
        pdf_load = timings.get("pdf_loading_ms", 0.0)
        clean = timings.get("text_cleaning_ms", 0.0)
        chunk = timings.get("text_chunking_ms", 0.0)
        bm25 = timings.get("bm25_ms", 0.0)
        rerank = timings.get("rerank_ms", 0.0)
        prompt_fmt = timings.get("prompt_construction_ms", 0.0)
        llm = timings.get("llm_generation_ms", 0.0)
        total = timings.get("total_latency_ms", 0.0)

        print("\n" + "=" * 65)
        print(f"LATENCY PROFILING BREAKDOWN — Query: '{query[:45]}...'")
        print("=" * 65)
        print(f"  PDF Loading (PyMuPDF)       : {pdf_load:>8.2f} ms")
        print(f"  Text Cleaning & Split       : {(clean + chunk):>8.2f} ms")
        print(f"  BM25 Candidate Search       : {bm25:>8.2f} ms")
        print(f"  CrossEncoder Reranking      : {rerank:>8.2f} ms")
        print(f"  Prompt Formatting           : {prompt_fmt:>8.2f} ms")
        print(f"  LLM Token Generation (GPU)  : {llm:>8.2f} ms ({(llm/1000):>5.2f} s)")
        print("-" * 65)
        print(f"  TOTAL PIPELINE LATENCY      : {total:>8.2f} ms ({(total/1000):>5.2f} s)")
        print("=" * 65 + "\n")

    def clear_memory(self):
        """Reset conversation memory."""
        self.memory.clear()

