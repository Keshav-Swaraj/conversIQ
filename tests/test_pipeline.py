"""Integration tests for core/pipeline.py."""

import pytest
from core.pipeline import ConverseIQPipeline
from scripts.create_sample_pdf import create_sample_pdf


@pytest.fixture
def test_pdf_path(tmp_path):
    pdf_path = tmp_path / "pipeline_test.pdf"
    create_sample_pdf(str(pdf_path))
    return str(pdf_path)


def test_pipeline_execution(test_pdf_path):
    pipeline = ConverseIQPipeline()
    res1 = pipeline.query(test_pdf_path, "What sparse retriever does ConverseIQ use?")
    assert len(res1["answer"]) > 0
    assert "bm25" in res1["answer"].lower()

    res2 = pipeline.query(test_pdf_path, "What is the population of Tokyo in 2026?")
    fallback_phrases = ["could not find", "not mentioned", "not provided", "no information", "doesn't provide", "does not contain", "unable"]
    assert any(p in res2["answer"].lower() for p in fallback_phrases)
