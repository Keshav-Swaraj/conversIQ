"""LLM-free PageIndex Flash document-tree building and JSON cache management."""

import json
from pathlib import Path
from typing import Any

from pageindex.flash import page_index_flash

from src.config import CACHE_DIR


def _cache_path(doc_id: str) -> Path:
    """Return the cache location for a single safe document identifier."""
    if not doc_id or Path(doc_id).name != doc_id:
        raise ValueError("doc_id must be a non-empty filename-safe identifier.")
    return CACHE_DIR / f"{doc_id}.json"


def load_tree(doc_id: str) -> dict[str, Any]:
    """Load a cached PageIndex tree without invoking PageIndex or Groq.

    Raises:
        FileNotFoundError: If no cached tree exists for ``doc_id``.
        ValueError: If the cache content is not a JSON object.
    """
    cache_path = _cache_path(doc_id)
    if not cache_path.is_file():
        raise FileNotFoundError(f"No cached PageIndex tree for doc_id: {doc_id}")

    with cache_path.open("r", encoding="utf-8") as cache_file:
        tree = json.load(cache_file)
    if not isinstance(tree, dict):
        raise ValueError(f"Cached PageIndex tree is not a JSON object: {cache_path}")
    return tree


def build_tree(
    file_path: str | Path,
    doc_id: str,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """Build an LLM-free PageIndex Flash tree for a PDF and cache it as JSON.

    An existing cache is returned before PageIndex Flash runs unless
    ``force_rebuild`` is true. Flash runs with summaries and optimization
    disabled, so this path does not call an LLM provider.

    Args:
        file_path: Path to the PDF to index with PageIndex.
        doc_id: Stable caller-supplied identifier used for the cache filename.
        force_rebuild: Rebuild the tree even when a cache file already exists.

    Returns:
        The native tree returned by ``page_index_flash``.

    Raises:
        FileNotFoundError: If the supplied PDF does not exist.
        ValueError: If the supplied path does not have a PDF extension.
    """
    cache_path = _cache_path(doc_id)
    if cache_path.is_file() and not force_rebuild:
        return load_tree(doc_id)

    pdf_path = Path(file_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, received: {pdf_path.name}")

    tree = page_index_flash(pdf_path, summary=False, optimize=False)
    if not isinstance(tree, dict):
        raise ValueError("PageIndex Flash returned an invalid tree structure.")

    with cache_path.open("w", encoding="utf-8") as cache_file:
        json.dump(tree, cache_file, ensure_ascii=False, indent=2)

    return tree
