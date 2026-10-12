"""Tests for the one-cycle application entry point without remote or audio calls."""

from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace
import sys

import pytest

import main
from src import graph
from src.extraction import ExtractedDocument, ExtractedPage
from src.retriever import RetrievalResult


@pytest.mark.parametrize(
    ("record", "interrupt", "speech"),
    [(False, False, False), (True, False, False),
     (False, True, False), (True, True, False), (True, False, True)],
)
def test_run_once_narrates_answers_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    record: bool, interrupt: bool, speech: bool,
) -> None:
    pdf = tmp_path / "document.pdf"
    question_audio = tmp_path / "question.wav"
    question_audio.write_bytes(b"recorded question")
    model = tmp_path / "silero.onnx"
    model.write_bytes(b"model fixture")
    events: list[str] = []
    document = ExtractedDocument("document.pdf", 1, (ExtractedPage(1, "abcdefghij"),))

    monkeypatch.setattr(main, "extract_pdf", lambda path: document)
    monkeypatch.setattr(main, "generate_doc_id", lambda path: "doc-id")
    monkeypatch.setattr(main, "build_tree", lambda path, doc_id: events.append("index"))

    def narrate(state, output_path, max_characters):
        events.append("narrate")
        assert max_characters == 4
        return document.pages[0].text[state.text_offset : state.text_offset + 4], Path(output_path)

    monkeypatch.setattr(main, "synthesize_narration_segment", narrate)
    monkeypatch.setattr(main, "play_wav", lambda path: events.append("play narration"))
    monkeypatch.setattr(
        main,
        "play_wav_interruptible",
        lambda path, should_interrupt: events.append("play interruptible") or False,
    )
    monkeypatch.setattr(
        main,
        "record_microphone",
        lambda path, seconds: events.append("record") or Path(path),
    )

    @contextmanager
    def fake_monitor(model_path):
        events.append("monitor open")
        yield SimpleNamespace(
            speech_detected=lambda: True,
            capture_question=lambda path, seconds: events.append("capture") or Path(path),
        )
        events.append("monitor close")

    monkeypatch.setattr(main, "monitor_microphone", fake_monitor)
    monkeypatch.setattr(graph, "transcribe_audio", lambda path: events.append("transcribe") or "What?")
    monkeypatch.setattr(
        graph,
        "retrieve_document",
        lambda **kwargs: events.append("retrieve") or RetrievalResult((1,), "source", (), ()),
    )
    monkeypatch.setattr(
        graph,
        "generate_answer",
        lambda **kwargs: events.append("answer") or "Grounded answer.",
    )
    monkeypatch.setattr(
        graph,
        "synthesize_speech",
        lambda text, output_path: events.append("synthesize answer") or Path(output_path),
    )
    monkeypatch.setattr(
        main,
        "get_interrupt_graph",
        lambda: graph.get_interrupt_graph(
            play_answer=lambda path: events.append("play answer")
        ),
    )

    result = main.run_once(
        pdf, None if record else question_audio, tmp_path, 4,
        record_seconds=2 if record else None,
        interrupt_on_enter=interrupt,
        interrupt_on_speech=speech,
        vad_model_path=model,
    )

    assert result["answer"] == "Grounded answer."
    assert result["saved_position"] == (1, 0 if interrupt or speech else 4)
    assert (result["narration_state"].page_number, result["narration_state"].text_offset) == (1, 4 if interrupt or speech else 8)
    assert not result["narration_state"].is_paused
    assert events == [
        "index", "narrate",
        *(["monitor open"] if speech else []),
        "play interruptible" if interrupt or speech else "play narration",
        *(["capture", "monitor close"] if speech else []),
        *(["record"] if record and not speech else []),
        "transcribe", "retrieve", "answer", "synthesize answer", "play answer",
        "narrate", "play narration",
    ]


def test_run_once_rejects_missing_question_audio(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Question audio does not exist"):
        main.run_once(tmp_path / "document.pdf", tmp_path / "missing.wav", tmp_path)


def test_run_once_requires_one_question_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="either question_audio_path or record_seconds"):
        main.run_once(tmp_path / "document.pdf", None, tmp_path)
    with pytest.raises(ValueError, match="either question_audio_path or record_seconds"):
        main.run_once(
            tmp_path / "document.pdf", tmp_path / "question.wav",
            tmp_path, record_seconds=2,
        )


@pytest.mark.parametrize("duration", [0, float("nan"), float("inf")])
def test_run_once_rejects_bad_record_duration_before_playback(
    tmp_path: Path, duration: float
) -> None:
    with pytest.raises(ValueError, match="record_seconds"):
        main.run_once(
            tmp_path / "document.pdf", None, tmp_path, record_seconds=duration,
        )


def test_speech_mode_without_detected_speech_skips_question_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    question_audio = tmp_path / "question.wav"
    model = tmp_path / "silero.onnx"
    model.write_bytes(b"model fixture")
    document = ExtractedDocument("document.pdf", 1, (ExtractedPage(1, "abcde"),))
    monkeypatch.setattr(main, "extract_pdf", lambda path: document)
    monkeypatch.setattr(main, "generate_doc_id", lambda path: "doc-id")
    monkeypatch.setattr(main, "build_tree", lambda path, doc_id: {})
    monkeypatch.setattr(
        main, "synthesize_narration_segment",
        lambda state, output, count: ("abc", Path(output)),
    )
    monkeypatch.setattr(main, "play_wav_interruptible", lambda path, check: True)

    @contextmanager
    def fake_monitor(model_path):
        yield SimpleNamespace(speech_detected=lambda: False)

    monkeypatch.setattr(main, "monitor_microphone", fake_monitor)
    monkeypatch.setattr(
        main, "get_interrupt_graph",
        lambda: pytest.fail("No graph or Groq call is expected without speech"),
    )

    result = main.run_once(
        tmp_path / "document.pdf", None, tmp_path,
        segment_characters=3, record_seconds=2, interrupt_on_speech=True,
        vad_model_path=model,
    )
    assert result["error"] == "No speech detected during the narration segment."
    assert result["narration_state"].text_offset == 3
    assert not question_audio.exists()


def test_cli_dispatches_continuous_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        sys, "argv",
        [
            "main.py", "--pdf", str(tmp_path / "document.pdf"),
            "--record-seconds", "3", "--interrupt-on-speech",
            "--continuous", "--max-segments", "2",
        ],
    )
    monkeypatch.setattr(
        main, "run_continuous_session",
        lambda *args: observed.append(args) or SimpleNamespace(
            narration_state=SimpleNamespace(
                is_complete=False, page_number=1, text_offset=6
            ),
            segments_attempted=2, interruptions=0, last_answer=None,
        ),
    )
    monkeypatch.setattr(
        main, "run_once", lambda *args: pytest.fail("One-cycle mode should not run")
    )

    main.main()

    assert len(observed) == 1
    assert observed[0][3:6] == (3.0, 140, 2)
    assert "Segments attempted: 2" in capsys.readouterr().out
