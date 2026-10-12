"""Offline checks for the removable conversation A/B experiment."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
import wave

import pytest

import experiments.conversation_ab as comparison
from experiments.conversation_ab import bounded_evidence, gemini_live_answer
from src import config
from src.extraction import ExtractedPage
from src.retriever import RetrievalResult
from src.tts_audio import wav_duration


def test_bounded_evidence_preserves_page_labels_and_reports_truncation() -> None:
    result = RetrievalResult(
        relevant_pages=(7, 8),
        source_text="unused",
        selected_nodes=(),
        source_pages=(ExtractedPage(7, "word " * 100), ExtractedPage(8, "later evidence")),
    )

    evidence, pages, truncated = bounded_evidence(result, 220)

    assert evidence.startswith("[Page 7]\n")
    assert "[Page 8]" not in evidence
    assert pages == (7,)
    assert truncated
    assert len(evidence) <= 220


def test_bounded_evidence_skips_blank_pages() -> None:
    result = RetrievalResult(
        relevant_pages=(1, 2),
        source_text="unused",
        selected_nodes=(),
        source_pages=(ExtractedPage(1, ""), ExtractedPage(2, "Only evidence.")),
    )

    assert bounded_evidence(result, 200) == ("[Page 2]\nOnly evidence.", (2,), False)
    with pytest.raises(ValueError, match="at least 200"):
        bounded_evidence(result, 199)


def test_bounded_evidence_can_use_one_retrieved_page() -> None:
    result = RetrievalResult(
        relevant_pages=(1, 4),
        source_text="unused",
        selected_nodes=(),
        source_pages=(ExtractedPage(1, "author contact"), ExtractedPage(4, "Accuracy: 98.69%.")),
    )

    assert bounded_evidence(result, 200, allowed_pages=frozenset({4})) == (
        "[Page 4]\nAccuracy: 98.69%.", (4,), False
    )


def test_gemini_live_collects_transcript_and_pcm_without_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from google import genai

    sent: list[object] = []

    class FakeSession:
        async def send_client_content(self, *, turns: object, turn_complete: bool) -> None:
            sent.append((turns, turn_complete))

        async def receive(self):  # type: ignore[no-untyped-def]
            audio = SimpleNamespace(mime_type="audio/pcm;rate=24000", data=b"\x00\x00" * 240)
            content = SimpleNamespace(
                output_transcription=SimpleNamespace(text="Frank Bryce."),
                model_turn=SimpleNamespace(parts=[SimpleNamespace(inline_data=audio)]),
            )
            yield SimpleNamespace(server_content=content)

    class FakeLive:
        @asynccontextmanager
        async def connect(self, *, model: str, config: object):  # type: ignore[no-untyped-def]
            sent.append((model, config))
            yield FakeSession()

    class FakeClient:
        aio = SimpleNamespace(live=FakeLive())

        def close(self) -> None:
            sent.append("closed")

    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(genai, "Client", lambda **kwargs: FakeClient())
    output = tmp_path / "answer.wav"

    result = asyncio.run(
        gemini_live_answer("Who?", "[Page 7]\nFrank Bryce.", (7,), output)
    )

    assert result["answer_transcript"] == "Frank Bryce."
    assert result["first_audio_seconds"] is not None
    assert wav_duration(output) == pytest.approx(0.01)
    assert sent[-1] == "closed"


def test_groq_kokoro_uses_spoken_answer_after_streaming(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeProvider:
        voice = "af_heart"

        def close(self) -> None:
            pass

    seen: list[str] = []

    def fake_synthesize(text: str, output: Path, *, provider: FakeProvider) -> Path:
        seen.append(text)
        with wave.open(str(output), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(24_000)
            audio.writeframes(b"\x00\x00" * 240)
        return output

    monkeypatch.setattr(comparison, "stream_answer", lambda *args: iter(("Frank ", "Bryce. 【7†L1】")))
    monkeypatch.setattr(comparison, "create_tts_provider", lambda name: FakeProvider())
    monkeypatch.setattr(comparison, "synthesize_speech", fake_synthesize)
    output = tmp_path / "groq.wav"

    result = comparison.groq_kokoro_answer("Who?", "[Page 7]\nFrank Bryce.", (7,), output)

    assert result["answer_transcript"] == "Frank Bryce. 【7†L1】"
    assert seen == ["Frank Bryce."]
    assert result["first_audio_seconds"] >= result["answer_ready_seconds"]
    assert wav_duration(output) == pytest.approx(0.01)
