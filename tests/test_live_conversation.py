"""Offline checks for grounded Gemini Live conversation events."""

import asyncio
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import pytest

from src.extraction import ExtractedPage
from src.live_conversation import LiveConversationError, normalize_history, source_excerpts, stream_live_answer, stream_live_speech
from src.retriever import RetrievalResult
from src.web_service import WorkspaceService


def _result(*texts: str) -> RetrievalResult:
    pages = tuple(ExtractedPage(index, text) for index, text in enumerate(texts, 1))
    return RetrievalResult(tuple(range(1, len(pages) + 1)), "", (), pages)


def test_empty_retrieval_does_not_call_gemini():
    with patch("src.live_conversation._live_events", side_effect=AssertionError("Live call")):
        events = list(stream_live_answer("Question?", _result("")))
    assert events[0] == {"type": "sources", "relevant_pages": []}
    assert events[-1] == {"type": "done"}
    assert "couldn't find" in events[1]["text"]


def test_source_excerpts_keep_page_order_and_text():
    evidence, pages = source_excerpts(_result("First page", "", "Third page"))
    assert pages == (1, 3)
    assert evidence == "[Page 1]\nFirst page\n\n[Page 3]\nThird page"


def test_workspace_live_answer_retrieves_once():
    result = _result("Evidence.")
    expected = [{"type": "sources", "relevant_pages": [1]}, {"type": "done"}]
    with (
        patch.object(WorkspaceService, "_document_path", return_value=Path("example.pdf")),
        patch("src.web_service.retrieve_document", return_value=result) as retrieve,
        patch("src.web_service.stream_live_answer", return_value=iter(expected)) as live,
    ):
        actual = list(WorkspaceService().ask_live_stream("0123456789abcdef", " What? "))
    assert actual == expected
    retrieve.assert_called_once_with("What?", "0123456789abcdef", Path("example.pdf"))
    live.assert_called_once_with("What?", result)


def test_live_answer_uses_bounded_conversation_context_once():
    result = _result("Evidence.")
    history = [{"question": "Who is the guide?", "answer": "The guide is unnamed."}]
    with (
        patch.object(WorkspaceService, "_document_path", return_value=Path("example.pdf")),
        patch("src.web_service.retrieve_document", return_value=result) as retrieve,
        patch("src.web_service.stream_live_answer", return_value=iter([{"type": "done"}])) as live,
    ):
        assert list(WorkspaceService().ask_live_stream("0123456789abcdef", "What did they do?", history)) == [{"type": "done"}]
    assert retrieve.call_count == 1
    assert "Who is the guide?" in retrieve.call_args.args[0]
    live.assert_called_once_with("What did they do?", result, tuple(history))


def test_live_history_rejects_unbounded_or_malformed_context():
    with pytest.raises(ValueError, match="at most eight"):
        normalize_history([{"question": "Q", "answer": "A"}] * 9)
    with pytest.raises(ValueError, match="question/answer"):
        normalize_history([{"question": "Q"}])


def test_live_stream_forwards_transcript_and_pcm(monkeypatch):
    import src.live_conversation as live

    monkeypatch.setattr(live, "GEMINI_API_KEY", "test-placeholder")
    monkeypatch.setattr(live, "GEMINI_LIVE_VOICE", "Kore")
    content = SimpleNamespace(
        output_transcription=SimpleNamespace(text="A grounded answer."),
        model_turn=SimpleNamespace(parts=[SimpleNamespace(
            inline_data=SimpleNamespace(mime_type="audio/pcm;rate=24000", data=b"\x00\x00")
        )]),
        turn_complete=True,
    )

    sent_prompts = []
    configs = []

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def send_client_content(self, *, turns, turn_complete):
            sent_prompts.append(turns["parts"][0]["text"])
            assert turn_complete is True

        async def receive(self):
            yield SimpleNamespace(server_content=content)

    class FakeClient:
        closed = False

        def __init__(self, *, api_key):
            assert api_key == "test-placeholder"
            def connect(**kwargs):
                configs.append(kwargs["config"])
                return FakeSession()
            self.aio = SimpleNamespace(live=SimpleNamespace(connect=connect))

        def close(self):
            self.closed = True

    client = FakeClient(api_key="test-placeholder")
    monkeypatch.setattr("google.genai.Client", lambda **_kwargs: client)
    events = list(stream_live_answer("What?", _result("Evidence.")))
    assert events == [
        {"type": "sources", "relevant_pages": [1]},
        {"type": "delta", "text": "A grounded answer."},
        {"type": "audio", "pcm_base64": "AAA="},
        {"type": "done"},
    ]
    assert client.closed
    speech_events = list(stream_live_speech("Hey Keshav."))
    assert speech_events[-2:] == [{"type": "audio", "pcm_base64": "AAA="}, {"type": "done"}]
    assert "Document excerpts" in sent_prompts[0]
    assert "Hey Keshav." in sent_prompts[1]
    assert "Document excerpts" not in sent_prompts[1]
    assert configs[0]["speech_config"] == configs[1]["speech_config"]
    assert configs[0]["speech_config"]["voice_config"]["prebuilt_voice_config"]["voice_name"] == live.GEMINI_LIVE_VOICE
    monkeypatch.setattr(live, "GEMINI_LIVE_VOICE", "")
    list(stream_live_speech("Hello again."))
    assert "speech_config" not in configs[2]


def test_sarah_prompt_is_cached_without_a_second_live_call(tmp_path):
    service = WorkspaceService(audio_dir=tmp_path)
    streamed = [
        {"type": "audio", "pcm_base64": "AAA="},
        {"type": "done"},
    ]
    with patch("src.web_service.stream_live_speech", return_value=iter(streamed)) as live:
        first = list(service.sarah_speech("Hey Keshav."))
    live.assert_called_once_with("Hey Keshav.")
    with patch("src.web_service.stream_live_speech", side_effect=AssertionError("cache miss")):
        second = list(service.sarah_speech("Hey Keshav."))
    assert first == second
    assert len(list(tmp_path.glob("live-prompt-*.pcm"))) == 1


def test_live_stream_times_out_before_first_answer(monkeypatch):
    import src.live_conversation as live

    monkeypatch.setattr(live, "GEMINI_API_KEY", "test-placeholder")
    monkeypatch.setattr(live, "FIRST_OUTPUT_TIMEOUT_SECONDS", 0.01)

    class WaitingSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def send_client_content(self, **_kwargs):
            return None

        async def receive(self):
            await asyncio.sleep(1)
            yield SimpleNamespace(server_content=None)

    client = SimpleNamespace(
        aio=SimpleNamespace(live=SimpleNamespace(connect=lambda **_kwargs: WaitingSession())),
        close=lambda: None,
    )
    monkeypatch.setattr("google.genai.Client", lambda **_kwargs: client)
    with pytest.raises(LiveConversationError, match="did not start answering"):
        list(stream_live_answer("What?", _result("Evidence.")))
