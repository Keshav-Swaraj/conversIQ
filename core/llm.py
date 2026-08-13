"""LLM Generation Module - Converted from Notebook 04."""

import os
import time
from typing import List, Dict, Any, Generator, Tuple
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


SYSTEM_GROUNDING_TEMPLATE = """You are a precise document Q&A assistant.

INSTRUCTIONS:
1. Answer strictly using ONLY the provided context chunks.
2. If the question refers to something mentioned in Conversation History (e.g. 'the boy', 'he', 'it'), use history to resolve the subject, then find the answer in the context chunks.
3. For summary questions, synthesize information across provided context chunks concisely.
4. If the answer is NOT in the context, state EXACTLY: 'I could not find that information in the provided document.'
5. Keep answers concise (1 to 3 sentences). Start your response DIRECTLY with the answer. Do NOT use phrases like 'According to the context' or 'Based on the provided text'.

Conversation History:
{history}

Context Chunks:
{context}

Question: {question}
Answer:
"""


class LLMGenerator:
    """Ollama LLM generation wrapper tuned for GPU acceleration (RTX 4050 / CUDA) and fast RAG inference."""

    def __init__(
        self,
        model_name: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        num_predict: int = 150,
        num_ctx: int = 2048,
        num_gpu: int = -1,
        keep_alive: str = "10m"
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            num_predict=num_predict,
            num_ctx=num_ctx,
            keep_alive=keep_alive,
            repeat_penalty=1.1,
            top_k=20,
            top_p=0.9,
            num_gpu=num_gpu,
            options={"thinking": False}
        )
        self.prompt = ChatPromptTemplate.from_template(SYSTEM_GROUNDING_TEMPLATE)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def format_context_string(self, results: List[Dict[str, Any]]) -> Tuple[str, float]:
        """Format reranked chunks into clean context string with timing metric."""
        t0 = time.perf_counter()
        if not results:
            context_str = "No context found."
        else:
            context_str = "\n\n".join([f"[{r['chunk_id']} | Section: {r['section']}]\n{r['content']}" for r in results])
        prompt_construction_ms = (time.perf_counter() - t0) * 1000.0
        return context_str, prompt_construction_ms

    def generate(self, question: str, context: str, history: str = "None") -> str:
        """Synchronously generate grounded answer."""
        answer, _ = self.generate_with_timings(question, context, history)
        return answer

    def generate_with_timings(self, question: str, context: str, history: str = "None") -> Tuple[str, Dict[str, float]]:
        """Synchronous generation with detailed time.perf_counter() micro-profiling."""
        timings: Dict[str, float] = {}
        
        t0 = time.perf_counter()
        payload = {
            "question": question,
            "context": context,
            "history": history
        }
        timings["prompt_construction_ms"] = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        response = self.chain.invoke(payload)
        timings["llm_generation_ms"] = (time.perf_counter() - t1) * 1000.0
        
        return response.strip(), timings

    def stream(self, question: str, context: str, history: str = "None") -> Generator[str, None, None]:
        """Stream generated answer tokens in real-time."""
        payload = {
            "question": question,
            "context": context,
            "history": history
        }
        for token in self.chain.stream(payload):
            yield token

