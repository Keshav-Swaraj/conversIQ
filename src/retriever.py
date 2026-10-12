"""Tree-guided retrieval of authoritative PDF source text."""

from dataclasses import dataclass
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from src.config import ANSWER_MODEL, GROQ_API_KEY, PAGEINDEX_MODEL
from src.extraction import ExtractedPage, extract_pdf
from src.indexer import load_tree


_MAX_SELECTOR_CANDIDATES = 200
_NO_RELEVANT_INFORMATION = (
    "The document does not contain relevant information to answer this question."
)


@dataclass(frozen=True)
class RetrievedNode:
    """A PageIndex tree node selected as relevant to a query."""

    node_id: str
    title: str
    start_index: int
    end_index: int


@dataclass(frozen=True)
class RetrievalResult:
    """Tree-selected PDF pages and their exact extracted source text."""

    relevant_pages: tuple[int, ...]
    source_text: str
    selected_nodes: tuple[RetrievedNode, ...]
    source_pages: tuple[ExtractedPage, ...]


class _NodeSelection(BaseModel):
    """Structured output returned by the retrieval-selection model."""

    selected_node_ids: list[str] = Field(
        default_factory=list,
        description="Node IDs selected exclusively from the supplied candidates.",
    )


def retrieve_document(
    query: str,
    doc_id: str,
    file_path: str | Path,
) -> RetrievalResult:
    """Select relevant PageIndex nodes and return their exact PDF source text.

    The selector receives only compact PageIndex node metadata. Page ranges are
    derived deterministically from the selected nodes, and source text always
    comes from :func:`src.extraction.extract_pdf`.

    Args:
        query: User information need used only to select document regions.
        doc_id: Identifier of a cached native PageIndex tree.
        file_path: Original PDF associated with the cached tree.

    Returns:
        Selected nodes, ascending unique page numbers, and exact page text.

    Raises:
        ValueError: If the query/tree is invalid or the selector returns an
            unknown node identifier.
        FileNotFoundError: If the cached tree or PDF cannot be found.
    """
    if not query.strip():
        raise ValueError("query must not be empty.")

    tree = load_tree(doc_id)
    document = extract_pdf(file_path)
    candidates = _flatten_tree(tree)

    if not candidates:
        return RetrievalResult((), "", (), ())
    if len(candidates) > _MAX_SELECTOR_CANDIDATES:
        raise ValueError(
            "Cached tree has too many selector candidates for one retrieval call."
        )

    selected_ids = _select_node_ids(query, candidates)
    candidates_by_id = {node.node_id: node for node in candidates}
    unknown_ids = [node_id for node_id in selected_ids if node_id not in candidates_by_id]
    if unknown_ids:
        raise ValueError(
            "Retrieval selector returned unknown PageIndex node IDs: "
            + ", ".join(unknown_ids)
        )

    selected_nodes = tuple(
        candidates_by_id[node_id]
        for node_id in dict.fromkeys(selected_ids)
    )
    relevant_page_numbers = _page_numbers(selected_nodes, document.num_pages)
    source_pages = tuple(
        page for page in document.pages if page.page_number in relevant_page_numbers
    )
    source_text = _source_text_from_pages(source_pages)

    return RetrievalResult(
        relevant_pages=relevant_page_numbers,
        source_text=source_text,
        selected_nodes=selected_nodes,
        source_pages=source_pages,
    )


def generate_answer(
    query: str,
    source_text: str,
    relevant_pages: tuple[int, ...],
) -> str:
    """Generate a concise answer grounded only in retrieved document text.

    No model call is made when retrieval supplied no usable source text or page
    references. In that case the deterministic no-information response is
    returned instead.

    Args:
        query: User question to answer from the supplied source text.
        source_text: Exact text returned by :func:`retrieve_document`.
        relevant_pages: Document pages associated with ``source_text``.

    Returns:
        A grounded answer, or a deterministic no-information response.
    """
    if not source_text.strip() or not relevant_pages:
        return _NO_RELEVANT_INFORMATION

    prompt = _answer_prompt()
    model = _answer_model()
    response = (prompt | model).invoke(
        {
            "query": query,
            "source_text": source_text,
            "relevant_pages": ", ".join(str(page) for page in relevant_pages),
        }
    )
    return str(response.content).strip()


def stream_answer(
    query: str, source_text: str, relevant_pages: tuple[int, ...]
) -> Iterator[str]:
    """Yield grounded answer text as the configured Groq model produces it."""
    if not source_text.strip() or not relevant_pages:
        yield _NO_RELEVANT_INFORMATION
        return
    for chunk in (_answer_prompt() | _answer_model()).stream({
        "query": query,
        "source_text": source_text,
        "relevant_pages": ", ".join(str(page) for page in relevant_pages),
    }):
        if isinstance(chunk.content, str) and chunk.content:
            yield chunk.content


def _answer_prompt() -> ChatPromptTemplate:
    """Keep streamed and non-streamed answers on the same grounded prompt."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer only from the supplied source text. Do not use outside "
                "knowledge or fabricate facts. If the source does not contain "
                "enough information, say so clearly. Answer in a short, natural "
                "paragraph, usually two to four sentences. Do not use Markdown "
                "or numbered lists. Cite only relevant page numbers when useful; "
                "do not invent line-number citations.",
            ),
            (
                "human",
                "Question:\n{query}\n\nSource:\n{source_text}\n\n"
                "Relevant pages:\n{relevant_pages}",
            ),
        ]
    )


def _answer_model() -> ChatGroq:
    """Use the same model parameters for both answer delivery modes."""
    return ChatGroq(
        model=ANSWER_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
        max_tokens=500,
    )


def _flatten_tree(tree: dict[str, Any]) -> tuple[RetrievedNode, ...]:
    """Return valid native Flash nodes as compact retrieval candidates."""
    structure = tree.get("structure")
    if not isinstance(structure, list):
        raise ValueError("Cached PageIndex tree has no list-valued structure.")

    candidates: list[RetrievedNode] = []

    def visit(nodes: list[Any]) -> None:
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = node.get("node_id")
            title = node.get("title")
            start_index = node.get("start_index")
            end_index = node.get("end_index")
            if (
                isinstance(node_id, str)
                and node_id
                and isinstance(title, str)
                and isinstance(start_index, int)
                and not isinstance(start_index, bool)
                and isinstance(end_index, int)
                and not isinstance(end_index, bool)
                and start_index <= end_index
            ):
                candidates.append(
                    RetrievedNode(node_id, title, start_index, end_index)
                )
            children = node.get("nodes")
            if isinstance(children, list):
                visit(children)

    visit(structure)
    return tuple(candidates)


def _select_node_ids(query: str, candidates: tuple[RetrievedNode, ...]) -> list[str]:
    """Use the retrieval model to select only known candidate node IDs."""
    candidate_lines = "\n".join(
        f"{node.node_id} | {node.title} | pages {node.start_index}-{node.end_index}"
        for node in candidates
    )
    prompt = (
        "You select document regions; you do not answer the user's question. "
        "Choose only node IDs from the candidate list that are needed to ground "
        "the query. Prefer the smallest relevant set. If the query cannot "
        "reasonably be grounded in this document structure, return no IDs. "
        "Return JSON with exactly one key: selected_node_ids.\n\n"
        f"Query: {query}\n\n"
        f"Candidates:\n{candidate_lines}"
    )
    selector = ChatGroq(
        model=_chatgroq_model_name(PAGEINDEX_MODEL),
        api_key=GROQ_API_KEY,
        temperature=0,
    ).with_structured_output(_NodeSelection, method="json_mode")
    selection = selector.invoke(prompt)
    if not isinstance(selection, _NodeSelection):
        raise ValueError("Retrieval selector returned an invalid structured response.")
    return selection.selected_node_ids


def _page_numbers(
    selected_nodes: tuple[RetrievedNode, ...], num_pages: int
) -> tuple[int, ...]:
    """Return ascending unique page numbers, safely clamped to the PDF."""
    page_numbers: set[int] = set()
    for node in selected_nodes:
        start = max(1, node.start_index)
        end = min(num_pages, node.end_index)
        if start <= end:
            page_numbers.update(range(start, end + 1))
    return tuple(sorted(page_numbers))


def _source_text_from_pages(source_pages: tuple[ExtractedPage, ...]) -> str:
    """Format selected page text in order without re-extracting the PDF."""
    if not any(page.text for page in source_pages):
        return ""
    return "\n\n".join(
        f"[Page {page.page_number}]\n{page.text}" for page in source_pages
    )


def _chatgroq_model_name(model_name: str) -> str:
    """Adapt a LiteLLM provider-prefixed model name for ChatGroq."""
    return model_name.removeprefix("groq/")
