"""Offline tests for the persistent Gemini Live voice bridge."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.extraction import ExtractedPage
from src.live_voice import _TurnGate, _config, _forward_response, _read_browser, _read_gemini, _tool_response
from src.retriever import RetrievalResult
from src.web_service import WorkspaceService


class _Connection:
    def __init__(self, messages=()):
        self.messages = list(messages)
        self.sent = []

    def __aiter__(self):
        async def iterate():
            for message in self.messages:
                yield message
        return iterate()

    async def send(self, message):
        self.sent.append(json.loads(message))


class _Session:
    def __init__(self):
        self.audio = []
        self.turns = []
        self.responses = []

    async def send_realtime_input(self, **values):
        self.audio.append(values)

    async def send_client_content(self, **values):
        self.turns.append(values)

    async def send_tool_response(self, **values):
        self.responses.append(values["function_responses"])


def test_live_config_has_input_audio_and_blocking_document_tool():
    config = _config()
    assert config.input_audio_transcription is not None
    assert config.output_audio_transcription is not None
    declarations = config.tools[0].function_declarations
    assert [tool.name for tool in declarations] == ["lookup_document", "resume_narration"]
    assert declarations[0].behavior.value == "BLOCKING"


def test_browser_pcm_and_text_share_one_live_session():
    connection = _Connection([b"\x00\x00", json.dumps({"type": "audio_end"}), json.dumps({"type": "text", "text": "What happened?"})])
    session = _Session()
    asyncio.run(_read_browser(connection, session))
    assert session.audio[0]["audio"].data == b"\x00\x00"
    assert session.audio[1] == {"audio_stream_end": True}
    assert session.turns[0]["turns"]["parts"][0]["text"] == "What happened?"


def test_speech_start_resets_previous_question_and_signals_new_turn():
    connection = _Connection([json.dumps({"type": "speech_start"})])
    session = _Session()
    gate = _TurnGate(greeting=False, input_text="Previous question?", last_query="Previous question?")
    asyncio.run(_read_browser(connection, session, gate))
    assert gate.input_text == gate.last_query == ""
    assert gate.pending and gate.awaiting_transcript
    assert connection.sent == [{"type": "input_start"}]


def test_previous_turn_completion_does_not_discard_new_untranscribed_question():
    connection = _Connection()
    gate = _TurnGate(greeting=False, pending=True, awaiting_transcript=True)
    response = SimpleNamespace(tool_call=None, server_content=SimpleNamespace(
        interrupted=False, input_transcription=None,
        output_transcription=None, model_turn=None, turn_complete=True,
    ))
    asyncio.run(_forward_response(
        connection, _Session(), WorkspaceService(), "0123456789abcdef",
        Path("book.pdf"), response, gate,
    ))
    assert connection.sent == []
    assert gate.pending and gate.awaiting_transcript


def test_document_lookup_uses_pageindex_retrieval_before_tool_reply():
    connection = _Connection()
    session = _Session()
    result = RetrievalResult(
        (7,), "", (), (ExtractedPage(7, "The Riddle House stood above the village."),),
    )
    gate = _TurnGate(greeting=False, pending=True, buffered=[{"type": "delta", "text": "Unsupported"}])
    call = SimpleNamespace(name="lookup_document", id="lookup-1", args={"query": "What is the Riddle House?"})
    with patch("src.live_voice.retrieve_document", return_value=result) as retrieve:
        resumed = asyncio.run(_tool_response(
            connection, session, WorkspaceService(), "0123456789abcdef", Path("book.pdf"), call, gate,
        ))
    assert not resumed
    retrieve.assert_called_once_with("What is the Riddle House?", "0123456789abcdef", Path("book.pdf"))
    assert connection.sent == [{"type": "sources", "relevant_pages": [7]}]
    assert session.responses[0].response["source_text"].startswith("[Page 7]")
    assert not gate.pending and gate.lookup_for_turn and not gate.buffered


def test_resume_command_ends_voice_mode_without_document_lookup():
    connection = _Connection()
    session = _Session()
    call = SimpleNamespace(name="resume_narration", id="resume-1", args={})
    with patch("src.live_voice.retrieve_document", side_effect=AssertionError("unexpected lookup")):
        resumed = asyncio.run(_tool_response(
            connection, session, WorkspaceService(), "0123456789abcdef", Path("book.pdf"), call,
        ))
    assert resumed
    assert connection.sent == [{"type": "resume"}]


def test_one_session_receives_multiple_model_turns():
    connection = _Connection()

    class TurnSession(_Session):
        calls = 0

        def receive(self):
            self.calls += 1

            async def turn():
                if self.calls <= 2:
                    yield SimpleNamespace(
                        tool_call=None,
                        server_content=SimpleNamespace(
                            interrupted=False, input_transcription=None,
                            output_transcription=SimpleNamespace(text=f"Turn {self.calls}"),
                            model_turn=None, turn_complete=True,
                        ),
                    )
            return turn()

    session = TurnSession()
    asyncio.run(_read_gemini(
        connection, session, WorkspaceService(), "0123456789abcdef", Path("book.pdf"),
    ))
    assert session.calls == 3
    assert connection.sent == [
        {"type": "delta", "text": "Turn 1"}, {"type": "done"},
        {"type": "delta", "text": "Turn 2"}, {"type": "done"},
    ]


def test_finished_continue_transcript_resumes_without_retrieval():
    connection = _Connection()
    content = SimpleNamespace(
        interrupted=False,
        input_transcription=SimpleNamespace(text="Continue narration.", finished=True),
        output_transcription=None, model_turn=None, turn_complete=False,
    )
    response = SimpleNamespace(tool_call=None, server_content=content)
    with patch("src.live_voice.retrieve_document", side_effect=AssertionError("unexpected lookup")):
        resumed = asyncio.run(_forward_response(
            connection, _Session(), WorkspaceService(), "0123456789abcdef",
            Path("book.pdf"), response, _TurnGate(),
        ))
    assert resumed
    assert connection.sent[-1] == {"type": "resume"}


def test_document_answer_without_lookup_is_not_spoken():
    connection = _Connection()
    gate = _TurnGate(greeting=False, pending=True)
    content = SimpleNamespace(
        interrupted=False, input_transcription=None,
        output_transcription=SimpleNamespace(text="An unsupported answer."),
        model_turn=SimpleNamespace(parts=[SimpleNamespace(inline_data=SimpleNamespace(
            mime_type="audio/pcm;rate=24000", data=b"\x00\x00",
        ))]),
        turn_complete=True,
    )
    response = SimpleNamespace(tool_call=None, server_content=content)
    asyncio.run(_forward_response(
        connection, _Session(), WorkspaceService(), "0123456789abcdef",
        Path("book.pdf"), response, gate,
    ))
    assert connection.sent == [{"type": "not_grounded"}]


def test_missed_document_tool_retrieves_and_requests_spoken_grounded_answer():
    connection = _Connection()
    session = _Session()
    gate = _TurnGate(greeting=False, pending=True, last_query="Who is Voldemort?")
    result = RetrievalResult(
        (14,), "", (), (ExtractedPage(14, "Voldemort spoke to Wormtail."),),
    )
    response = SimpleNamespace(tool_call=None, server_content=SimpleNamespace(
        interrupted=False, input_transcription=None,
        output_transcription=SimpleNamespace(text="An unsupported answer."),
        model_turn=SimpleNamespace(parts=[SimpleNamespace(inline_data=SimpleNamespace(
            mime_type="audio/pcm;rate=24000", data=b"\x00\x00",
        ))]), turn_complete=True,
    ))
    with patch("src.live_voice.retrieve_document", return_value=result) as retrieve:
        resumed = asyncio.run(_forward_response(
            connection, session, WorkspaceService(), "0123456789abcdef",
            Path("book.pdf"), response, gate,
        ))
    assert not resumed
    retrieve.assert_called_once_with("Who is Voldemort?", "0123456789abcdef", Path("book.pdf"))
    assert connection.sent == [{"type": "sources", "relevant_pages": [14]}]
    assert "[Page 14]" in session.turns[0]["turns"]["parts"][0]["text"]
    assert not gate.pending and gate.lookup_for_turn and not gate.buffered
    spoken = SimpleNamespace(tool_call=None, server_content=SimpleNamespace(
        interrupted=False, input_transcription=None,
        output_transcription=SimpleNamespace(text="On page 14, Voldemort speaks to Wormtail."),
        model_turn=SimpleNamespace(parts=[SimpleNamespace(inline_data=SimpleNamespace(
            mime_type="audio/pcm;rate=24000", data=b"\x00\x00",
        ))]), turn_complete=True,
    ))
    asyncio.run(_forward_response(
        connection, session, WorkspaceService(), "0123456789abcdef",
        Path("book.pdf"), spoken, gate,
    ))
    assert [event["type"] for event in connection.sent] == [
        "sources", "delta", "audio", "cited_pages", "done",
    ]


def test_grounded_spoken_answer_reports_only_pages_it_cites():
    connection = _Connection()
    gate = _TurnGate(greeting=False, lookup_for_turn=True, selected_pages=(7, 8, 14, 21))
    response = SimpleNamespace(tool_call=None, server_content=SimpleNamespace(
        interrupted=False, input_transcription=None,
        output_transcription=SimpleNamespace(text="On page 14, Voldemort speaks to Wormtail. On page 21, Frank dies."),
        model_turn=None, turn_complete=True,
    ))
    asyncio.run(_forward_response(
        connection, _Session(), WorkspaceService(), "0123456789abcdef",
        Path("book.pdf"), response, gate,
    ))
    assert connection.sent == [
        {"type": "delta", "text": "On page 14, Voldemort speaks to Wormtail. On page 21, Frank dies."},
        {"type": "cited_pages", "relevant_pages": [14, 21]},
        {"type": "done"},
    ]
