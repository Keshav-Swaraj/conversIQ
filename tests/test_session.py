"""Continuous-session tests with fake audio and graph boundaries."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import session
from src.extraction import ExtractedDocument, ExtractedPage
from src.narrator import remaining_page_text


def test_session_replays_interrupted_segment_and_reuses_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "silero.onnx"
    model.write_bytes(b"model fixture")
    document = ExtractedDocument("document.pdf", 1, (ExtractedPage(1, "abcdefg"),))
    events: list[str] = []
    playback = iter([True, False, True, True])
    offsets: list[int] = []

    monkeypatch.setattr(session, "extract_pdf", lambda path: events.append("extract") or document)
    monkeypatch.setattr(session, "generate_doc_id", lambda path: "doc-id")
    monkeypatch.setattr(session, "build_tree", lambda path, doc_id: events.append("index"))
    client = SimpleNamespace(close=lambda: events.append("close client"))
    monkeypatch.setattr(session, "create_tts_provider", lambda: events.append("create client") or client)

    def synthesize(state, output, count, *, provider):
        assert provider is client_for_test
        offsets.append(state.text_offset)
        return remaining_page_text(state)[:count], Path(output)

    client_for_test = client

    monkeypatch.setattr(session, "synthesize_narration_segment", synthesize)
    monkeypatch.setattr(
        session, "play_wav_interruptible",
        lambda path, check: next(playback),
    )

    @contextmanager
    def fake_monitor(model_path, input_device):
        events.append("monitor")
        assert input_device == 3
        yield SimpleNamespace(
            speech_detected=lambda: False,
            capture_question=lambda path, seconds: events.append("capture") or Path(path),
        )

    monkeypatch.setattr(session, "monitor_microphone", fake_monitor)

    class FakeGraph:
        def invoke(self, values):
            events.append("question")
            assert values["narration_state"].text_offset == 3
            assert values["tts_provider"] is client
            return {"narration_state": values["narration_state"], "answer": "Grounded answer."}

    monkeypatch.setattr(
        session, "get_interrupt_graph",
        lambda: events.append("compile") or FakeGraph(),
    )

    result = session.run_continuous_session(
        tmp_path / "document.pdf", model, tmp_path,
        record_seconds=2, segment_characters=3, input_device=3,
    )

    assert result.narration_state.is_complete
    assert result.segments_attempted == 4
    assert result.interruptions == 1
    assert result.last_answer == "Grounded answer."
    assert offsets == [0, 3, 6]
    assert events.count("extract") == 1
    assert events.count("index") == 1
    assert events.count("compile") == 1
    assert events.count("create client") == 1
    assert events.count("close client") == 1
    assert events.count("question") == 1
    assert events.count("capture") == 1


def test_session_segment_limit_keeps_resumable_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "silero.onnx"
    model.write_bytes(b"model fixture")
    document = ExtractedDocument("document.pdf", 1, (ExtractedPage(1, "abcdefghij"),))
    monkeypatch.setattr(session, "extract_pdf", lambda path: document)
    monkeypatch.setattr(session, "generate_doc_id", lambda path: "doc-id")
    monkeypatch.setattr(session, "build_tree", lambda path, doc_id: {})
    monkeypatch.setattr(session, "get_interrupt_graph", lambda: SimpleNamespace(invoke=None))
    monkeypatch.setattr(session, "create_tts_provider", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(
        session, "synthesize_narration_segment",
        lambda state, output, count, *, provider: (
            remaining_page_text(state)[:count], Path(output)
        ),
    )
    monkeypatch.setattr(session, "play_wav_interruptible", lambda path, check: True)

    @contextmanager
    def fake_monitor(model_path, input_device):
        assert input_device is None
        yield SimpleNamespace(speech_detected=lambda: False)

    monkeypatch.setattr(session, "monitor_microphone", fake_monitor)

    result = session.run_continuous_session(
        tmp_path / "document.pdf", model, tmp_path,
        segment_characters=3, max_segments=2,
    )
    assert not result.narration_state.is_complete
    assert result.narration_state.text_offset == 6
    assert result.segments_attempted == 2
    assert result.interruptions == 0


def test_stop_during_playback_does_not_capture_or_advance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "silero.onnx"
    model.write_bytes(b"model fixture")
    document = ExtractedDocument("document.pdf", 1, (ExtractedPage(1, "A sentence."),))
    events: list[str] = []
    stop = False

    monkeypatch.setattr(session, "extract_pdf", lambda path: document)
    monkeypatch.setattr(session, "generate_doc_id", lambda path: "doc-id")
    monkeypatch.setattr(session, "build_tree", lambda path, doc_id: {})
    monkeypatch.setattr(session, "get_interrupt_graph", lambda: SimpleNamespace(invoke=None))
    monkeypatch.setattr(session, "create_tts_provider", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(
        session, "synthesize_narration_segment",
        lambda state, output, count, *, provider: ("A sentence.", Path(output)),
    )

    def fake_playback(path, should_interrupt):
        nonlocal stop
        stop = True
        assert should_interrupt()
        return False

    monkeypatch.setattr(session, "play_wav_interruptible", fake_playback)

    @contextmanager
    def fake_monitor(model_path, input_device):
        assert input_device is None
        yield SimpleNamespace(
            speech_detected=lambda: False,
            capture_question=lambda path, seconds: events.append("captured"),
        )

    monkeypatch.setattr(session, "monitor_microphone", fake_monitor)
    result = session.run_continuous_session(
        tmp_path / "document.pdf", model, tmp_path,
        stop_requested=lambda: stop, on_update=events.append,
    )

    assert result.segments_attempted == 1
    assert result.interruptions == 0
    assert result.narration_state.text_offset == 0
    assert not result.narration_state.is_complete
    assert "captured" not in events
    assert events[-1] == "Session stopped."


def test_session_validates_limit_and_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_segments"):
        session.run_continuous_session(
            tmp_path / "doc.pdf", tmp_path / "model.onnx",
            tmp_path, max_segments=0,
        )
    with pytest.raises(FileNotFoundError, match="Silero VAD model"):
        session.run_continuous_session(
            tmp_path / "doc.pdf", tmp_path / "model.onnx",
            tmp_path,
        )
