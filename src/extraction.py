"""Deterministic, page-by-page PDF text extraction using PyMuPDF."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import pymupdf


@dataclass(frozen=True)
class ExtractedPage:
    """Text extracted from one PDF page."""

    page_number: int
    text: str


@dataclass(frozen=True)
class ExtractedDocument:
    """Structured text extracted from a PDF document."""

    filename: str
    num_pages: int
    pages: tuple[ExtractedPage, ...]


def generate_doc_id(file_path: str | Path) -> str:
    """Return a stable document identifier derived from a filename and file size.

    The identifier does not read the file contents, so it can be used for cache
    lookup before PDF processing begins.

    Args:
        file_path: Path to the document file.

    Returns:
        A 16-character SHA-256 prefix based on the filename and byte size.

    Raises:
        FileNotFoundError: If ``file_path`` does not refer to an existing file.
    """
    document_path = Path(file_path)
    if not document_path.is_file():
        raise FileNotFoundError(f"Document file does not exist: {document_path}")

    identifier_source = f"{document_path.name}:{document_path.stat().st_size}"
    return sha256(identifier_source.encode("utf-8")).hexdigest()[:16]


def extract_pdf(file_path: str | Path) -> ExtractedDocument:
    """Extract 1-based pages in visible reading order from a valid PDF file.

    Args:
        file_path: Path to the PDF to extract.

    Returns:
        A document record containing the filename, page count, and every page's text.

    Raises:
        FileNotFoundError: If ``file_path`` does not refer to an existing file.
        ValueError: If the path is not a PDF or PyMuPDF cannot read it.
    """
    pdf_path = Path(file_path)

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, received: {pdf_path.name}")

    try:
        with pymupdf.open(pdf_path) as document:
            pages = tuple(
                ExtractedPage(page_number=index, text=page.get_text(sort=True))
                for index, page in enumerate(document, start=1)
            )
    except (OSError, RuntimeError) as error:
        raise ValueError(f"Unable to open or read PDF: {pdf_path}") from error

    return ExtractedDocument(
        filename=pdf_path.name,
        num_pages=len(pages),
        pages=pages,
    )
