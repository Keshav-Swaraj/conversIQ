"""Tests for deterministic PDF text extraction."""

from pathlib import Path

import pymupdf
import pytest

from src.extraction import ExtractedDocument, extract_pdf


@pytest.fixture
def valid_pdf(tmp_path: Path) -> Path:
    """Create a two-page PDF with deterministic text."""
    pdf_path = tmp_path / "sample.pdf"
    with pymupdf.open() as document:
        first_page = document.new_page()
        first_page.insert_text((72, 72), "First page text")
        second_page = document.new_page()
        second_page.insert_text((72, 72), "Second page text")
        document.save(pdf_path)
    return pdf_path


def test_extract_pdf_returns_ordered_page_records(valid_pdf: Path) -> None:
    """A valid PDF returns an ordered, typed document record."""
    extracted = extract_pdf(valid_pdf)

    assert isinstance(extracted, ExtractedDocument)
    assert extracted.filename == valid_pdf.name
    assert extracted.num_pages == 2
    assert len(extracted.pages) == extracted.num_pages
    assert [page.page_number for page in extracted.pages] == [1, 2]
    assert extracted.pages[0].text.strip() == "First page text"
    assert extracted.pages[1].text.strip() == "Second page text"


def test_extract_pdf_uses_visible_top_to_bottom_reading_order(tmp_path: Path) -> None:
    """PDF drawing order must not make narration skip the visible first paragraph."""
    pdf_path = tmp_path / "out-of-order.pdf"
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 320), "Later paragraph")
        page.insert_text((72, 100), "Opening paragraph")
        document.save(pdf_path)

    assert [line for line in extract_pdf(pdf_path).pages[0].text.splitlines() if line.strip()] == [
        "Opening paragraph", "Later paragraph",
    ]


def test_extract_pdf_rejects_non_pdf_extension(tmp_path: Path) -> None:
    """Existing non-PDF files are rejected before opening."""
    text_path = tmp_path / "not-a-pdf.txt"
    text_path.write_text("not a PDF", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a .pdf file"):
        extract_pdf(text_path)


def test_extract_pdf_rejects_missing_file(tmp_path: Path) -> None:
    """Missing PDF paths raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="PDF file does not exist"):
        extract_pdf(tmp_path / "missing.pdf")


def test_extract_pdf_rejects_corrupt_pdf(tmp_path: Path) -> None:
    """Files with a PDF extension but invalid content are rejected."""
    corrupt_pdf = tmp_path / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"This is not a PDF.")

    with pytest.raises(ValueError, match="Unable to open or read PDF"):
        extract_pdf(corrupt_pdf)


def test_extract_pdf_preserves_blank_pages(tmp_path: Path) -> None:
    """Blank pages remain in the result with an explicit empty text value."""
    blank_pdf = tmp_path / "blank.pdf"
    with pymupdf.open() as document:
        document.new_page()
        document.save(blank_pdf)

    extracted = extract_pdf(blank_pdf)

    assert extracted.num_pages == 1
    assert len(extracted.pages) == 1
    assert extracted.pages[0].page_number == 1
    assert extracted.pages[0].text == ""
