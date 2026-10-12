"""LangGraph workflows for grounded answers and narration interruptions."""

from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.extraction import ExtractedPage
from src.indexer import load_tree
from src.narrator import NarrationState, pause_narration, resume_narration
from src.retriever import RetrievedNode, generate_answer, retrieve_document
from src.stt import transcribe_audio
from src.tts import play_wav, prepare_spoken_answer, synthesize_speech
from src.tts_contract import TTSProvider


class ConversIQState(TypedDict, total=False):
    """Intermediate state for the Layer 5 retrieval-answer workflow."""

    doc_id: str
    file_path: str | Path
    query: str
    tree: dict[str, Any]
    selected_nodes: tuple[RetrievedNode, ...]
    relevant_pages: tuple[int, ...]
    source_pages: tuple[ExtractedPage, ...]
    source_text: str
    answer: str
    error: str
    node_timings: dict[str, float]
    node_call_counts: dict[str, int]


class InterruptState(ConversIQState, total=False):
    """State for one spoken question during paused document narration."""

    narration_state: NarrationState
    saved_position: tuple[int, int]
    audio_path: str | Path
    answer_audio_path: str | Path
    answer_audio_played: bool
    tts_provider: TTSProvider
    tts_client: Any  # Older notebook callers may still pass a Cartesia client.


def _node_diagnostics(
    state: ConversIQState, node_name: str, started_at: float
) -> dict[str, dict[str, float] | dict[str, int]]:
    """Return state-local timing and execution-count updates for one node."""
    node_timings = dict(state.get("node_timings", {}))
    node_call_counts = dict(state.get("node_call_counts", {}))
    node_timings[node_name] = perf_counter() - started_at
    node_call_counts[node_name] = node_call_counts.get(node_name, 0) + 1
    return {
        "node_timings": node_timings,
        "node_call_counts": node_call_counts,
    }


def load_tree_node(state: ConversIQState) -> ConversIQState:
    """Load the cached PageIndex tree without rebuilding it."""
    started_at = perf_counter()
    tree = load_tree(state["doc_id"])
    return {"tree": tree, **_node_diagnostics(state, "load_tree", started_at)}


def retrieve_node(state: ConversIQState) -> ConversIQState:
    """Run the existing tree-guided retrieval implementation."""
    started_at = perf_counter()
    result = retrieve_document(
        query=state["query"],
        doc_id=state["doc_id"],
        file_path=state["file_path"],
    )
    return {
        "selected_nodes": result.selected_nodes,
        "relevant_pages": result.relevant_pages,
        "source_pages": result.source_pages,
        "source_text": result.source_text,
        **_node_diagnostics(state, "retrieve", started_at),
    }


def generate_answer_node(state: ConversIQState) -> ConversIQState:
    """Generate an answer from the existing retrieved source text."""
    started_at = perf_counter()
    answer = generate_answer(
        query=state["query"],
        source_text=state["source_text"],
        relevant_pages=state["relevant_pages"],
    )
    return {
        "answer": answer,
        **_node_diagnostics(state, "generate_answer", started_at),
    }


def get_graph() -> CompiledStateGraph[ConversIQState]:
    """Build and compile the Layer 5 retrieval-answer workflow graph."""
    workflow = StateGraph(ConversIQState)
    workflow.add_node("load_tree", load_tree_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_edge(START, "load_tree")
    workflow.add_edge("load_tree", "retrieve")
    workflow.add_edge("retrieve", "generate_answer")
    workflow.add_edge("generate_answer", END)
    return workflow.compile()


def pause_narration_node(state: InterruptState) -> InterruptState:
    """Save the narration cursor and pause before processing the question."""
    narration = state["narration_state"]
    return {
        "saved_position": (narration.page_number, narration.text_offset),
        "narration_state": pause_narration(narration),
    }


def transcribe_question_node(state: InterruptState) -> InterruptState:
    """Transcribe the recorded question with the existing STT layer."""
    query = transcribe_audio(state["audio_path"]).strip()
    return {
        "query": query,
        "error": "" if query else "No speech recognized in the question audio.",
    }


def _route_question(state: InterruptState) -> str:
    """Skip retrieval and answer audio when no question was recognized."""
    return "retrieve" if state["query"] else "resume_narration"


def synthesize_answer_node(state: InterruptState) -> InterruptState:
    """Speak a clean copy of the answer without changing its displayed text."""
    spoken_answer = prepare_spoken_answer(state["answer"])
    if "tts_provider" in state:
        audio_path = synthesize_speech(
            spoken_answer, state["answer_audio_path"], provider=state["tts_provider"],
        )
    elif "tts_client" in state:  # Compatibility with an older notebook state.
        audio_path = synthesize_speech(
            spoken_answer, state["answer_audio_path"], client=state["tts_client"],
        )
    else:
        audio_path = synthesize_speech(spoken_answer, state["answer_audio_path"])
    return {"answer_audio_path": audio_path}


def resume_narration_node(state: InterruptState) -> InterruptState:
    """Resume only if the original page and character offset are intact."""
    narration = state["narration_state"]
    position = (narration.page_number, narration.text_offset)
    if position != state["saved_position"]:
        raise ValueError("Narration position changed during the interruption.")
    return {"narration_state": resume_narration(narration)}


def get_interrupt_graph(
    play_answer: Callable[[Path], None] = play_wav,
) -> CompiledStateGraph[InterruptState]:
    """Compile one pause → spoken question → answer → resume workflow.

    ``play_answer`` must block until playback completes. The default uses
    synchronous Windows WAV playback; a caller can supply another player.
    """

    def play_answer_node(state: InterruptState) -> InterruptState:
        play_answer(Path(state["answer_audio_path"]))
        return {"answer_audio_played": True}

    workflow = StateGraph(InterruptState)
    workflow.add_node("pause_narration", pause_narration_node)
    workflow.add_node("transcribe_question", transcribe_question_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("synthesize_answer", synthesize_answer_node)
    workflow.add_node("play_answer", play_answer_node)
    workflow.add_node("resume_narration", resume_narration_node)
    workflow.add_edge(START, "pause_narration")
    workflow.add_edge("pause_narration", "transcribe_question")
    workflow.add_conditional_edges(
        "transcribe_question",
        _route_question,
        {"retrieve": "retrieve", "resume_narration": "resume_narration"},
    )
    workflow.add_edge("retrieve", "generate_answer")
    workflow.add_edge("generate_answer", "synthesize_answer")
    workflow.add_edge("synthesize_answer", "play_answer")
    workflow.add_edge("play_answer", "resume_narration")
    workflow.add_edge("resume_narration", END)
    return workflow.compile()
