"""Unit tests for core/document.py."""

import os
import pytest
from core.document import PDFDocumentProcessor, HierarchicalChunk
from scripts.create_sample_pdf import create_sample_pdf


@pytest.fixture
def sample_pdf(tmp_path):
    pdf_path = tmp_path / "test_report.pdf"
    create_sample_pdf(str(pdf_path))
    return str(pdf_path)


def test_pdf_extraction_and_chunking(sample_pdf):
    processor = PDFDocumentProcessor()
    chunks = processor.process(sample_pdf)

    assert len(chunks) > 0, "No chunks generated from test PDF"
    for chunk in chunks:
        assert isinstance(chunk, HierarchicalChunk)
        assert chunk.content.endswith(('.', '!', '?', ':', ';'))
        assert chunk.word_count > 0
        assert len(chunk.sentences) > 0
