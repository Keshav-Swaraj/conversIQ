"""Integration tests for the spoken interruption state transitions."""

from pathlib import Path

import pytest

from src import graph
from src.extraction import ExtractedDocument, ExtractedPage
from src.narrator import advance_narration, resume_narration, start_narration
from src.retriever import RetrievalResult


@pytest.fixture
def narration_state():
    document = ExtractedDocument(
        "question.pdf", 1, (ExtractedPage(1, "A Portkey transports people."),)
    )
    return advance_narration(start_narration(document), 4)


def test_spoken_question_answers_then_resumes_at_saved_position(
    narration_state, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    session_client = object()

    def transcribe(audio_path: str | Path) -> str:
        events.append("transcribe")
        return "What is a Portkey?"

    def retrieve(query: str, doc_id: str, file_path: str | Path) -> RetrievalResult:
        events.append("retrieve")
        assert (query, doc_id) == ("What is a Portkey?", "test-doc")
        return RetrievalResult((1,), "[Page 1]\nA Portkey transports people.", (), ())

    def answer(query: str, source_text: str, relevant_pages: tuple[int, ...]) -> str:
        events.append("answer")
        assert relevant_pages == (1,)
        assert "Portkey" in source_text
        return "**A Portkey** transports people. 【1†L1-L2】"

    def synthesize(
        text: str, output_path: str | Path, *, client: object
    ) -> Path:
        events.append("synthesize")
        assert text == "A Portkey transports people."
        assert client is session_client
        return Path(output_path)

    def play(path: Path) -> None:
        events.append("play")
        assert path == tmp_path / "answer.wav"

    def resume(state):
        events.append("resume")
        return resume_narration(state)

    monkeypatch.setattr(graph, "transcribe_audio", transcribe)
    monkeypatch.setattr(graph, "retrieve_document", retrieve)
    monkeypatch.setattr(graph, "generate_answer", answer)
    monkeypatch.setattr(graph, "synthesize_speech", synthesize)
    monkeypatch.setattr(graph, "resume_narration", resume)

    workflow = graph.get_interrupt_graph(play_answer=play)
    final_state = workflow.invoke(
        {
            "narration_state": narration_state,
            "doc_id": "test-doc",
            "file_path": tmp_path / "question.pdf",
            "audio_path": tmp_path / "question.wav",
            "answer_audio_path": tmp_path / "answer.wav",
            "tts_client": session_client,
        }
    )

    assert events == [
        "transcribe", "retrieve", "answer", "synthesize", "play", "resume"
    ]
    assert final_state["answer_audio_played"] is True
    assert final_state["saved_position"] == (1, 4)
    assert final_state["relevant_pages"] == (1,)
    assert final_state["answer"] == "**A Portkey** transports people. 【1†L1-L2】"
    assert final_state["narration_state"] == narration_state
    assert not final_state["narration_state"].is_paused


def test_no_recognized_question_resumes_without_llm_or_audio(
    narration_state, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(graph, "transcribe_audio", lambda path: "  ")

    def unexpected(*args, **kwargs):
        raise AssertionError("No retrieval, answer, synthesis, or playback is expected")

    monkeypatch.setattr(graph, "retrieve_document", unexpected)
    monkeypatch.setattr(graph, "generate_answer", unexpected)
    monkeypatch.setattr(graph, "synthesize_speech", unexpected)
    workflow = graph.get_interrupt_graph(play_answer=unexpected)

    final_state = workflow.invoke(
        {
            "narration_state": narration_state,
            "audio_path": tmp_path / "question.wav",
        }
    )
    assert final_state["query"] == ""
    assert final_state["error"] == "No speech recognized in the question audio."
    assert final_state["narration_state"] == narration_state
    assert "answer_audio_played" not in final_state


def test_unanswerable_question_speaks_grounded_fallback(
    narration_state, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spoken: list[str] = []
    monkeypatch.setattr(graph, "transcribe_audio", lambda path: "What is the capital of France?")
    monkeypatch.setattr(
        graph,
        "retrieve_document",
        lambda **kwargs: RetrievalResult((), "", (), ()),
    )

    def synthesize(text: str, output_path: str | Path) -> Path:
        spoken.append(text)
        return Path(output_path)

    monkeypatch.setattr(graph, "synthesize_speech", synthesize)
    workflow = graph.get_interrupt_graph(play_answer=lambda path: None)
    final_state = workflow.invoke(
        {
            "narration_state": narration_state,
            "doc_id": "test-doc",
            "file_path": tmp_path / "question.pdf",
            "audio_path": tmp_path / "question.wav",
            "answer_audio_path": tmp_path / "answer.wav",
        }
    )

    assert final_state["relevant_pages"] == ()
    assert final_state["source_text"] == ""
    assert spoken == [
        "The document does not contain relevant information to answer this question."
    ]
    assert final_state["narration_state"] == narration_state


def test_resume_rejects_a_changed_cursor(narration_state) -> None:
    changed = advance_narration(narration_state, 1)
    with pytest.raises(ValueError, match="position changed"):
        graph.resume_narration_node(
            {"narration_state": changed, "saved_position": (1, 4)}
        )
