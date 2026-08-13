"""Unit tests for core/retriever.py."""

import pytest
from core.document import PDFDocumentProcessor
from core.retriever import BM25RerankRetriever
from scripts.create_sample_pdf import create_sample_pdf


@pytest.fixture
def retriever_instance(tmp_path):
    pdf_path = tmp_path / "test_retriever.pdf"
    create_sample_pdf(str(pdf_path))
    processor = PDFDocumentProcessor()
    chunks = processor.process(str(pdf_path))
    return BM25RerankRetriever(chunks)


def test_bm25_retrieval(retriever_instance):
    docs = retriever_instance.retrieve_bm25("What sparse retriever does ConverseIQ use?")
    assert len(docs) > 0
    assert any("bm25" in doc.page_content.lower() for doc in docs)


def test_crossencoder_reranking(retriever_instance):
    results, latency_ms = retriever_instance.retrieve_and_rerank("What is the average CPU latency for BM25?")
    assert len(results) > 0
    assert latency_ms < 1000.0  # Latency under 1 sec
    assert results[0]["score"] >= results[-1]["score"]
