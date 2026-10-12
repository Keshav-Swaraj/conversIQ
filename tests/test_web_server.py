"""HTTP audio responses used by the browser narration player."""

from io import BytesIO
from http import HTTPStatus
import json
from types import SimpleNamespace
from unittest.mock import Mock

from web_server import WorkspaceHandler
from src.live_conversation import LiveConversationError


def test_audio_file_supports_seekable_byte_ranges(tmp_path):
    audio_path = tmp_path / "narration.wav"
    audio_path.write_bytes(b"0123456789")
    handler = SimpleNamespace(
        headers={"Range": "bytes=2-5"},
        send_response=Mock(),
        send_header=Mock(),
        end_headers=Mock(),
        wfile=BytesIO(),
    )

    WorkspaceHandler._audio_file(handler, audio_path)

    handler.send_response.assert_called_once_with(HTTPStatus.PARTIAL_CONTENT)
    assert handler.wfile.getvalue() == b"2345"
    assert ("Content-Range", "bytes 2-5/10") in [
        call.args for call in handler.send_header.call_args_list
    ]
    assert ("Accept-Ranges", "bytes") in [
        call.args for call in handler.send_header.call_args_list
    ]


def test_audio_file_serves_complete_wav_without_range(tmp_path):
    audio_path = tmp_path / "narration.wav"
    audio_path.write_bytes(b"0123456789")
    handler = SimpleNamespace(
        headers={}, send_response=Mock(), send_header=Mock(),
        end_headers=Mock(), wfile=BytesIO(),
    )

    WorkspaceHandler._audio_file(handler, audio_path)

    handler.send_response.assert_called_once_with(HTTPStatus.OK)
    assert handler.wfile.getvalue() == b"0123456789"


def test_answer_events_are_flushed_before_the_next_model_delta():
    class TrackingBuffer(BytesIO):
        flush_count = 0

        def flush(self):
            self.flush_count += 1

    output = TrackingBuffer()
    handler = SimpleNamespace(
        send_response=Mock(), send_header=Mock(),
        end_headers=Mock(), wfile=output,
    )

    def events():
        yield {"type": "sources", "relevant_pages": [1]}
        assert output.flush_count == 1
        yield {"type": "delta", "text": "First sentence."}
        assert output.flush_count == 2
        yield {"type": "done"}

    WorkspaceHandler._ndjson(handler, events())

    handler.send_response.assert_called_once_with(HTTPStatus.OK)
    assert [json.loads(line) for line in output.getvalue().splitlines()] == [
        {"type": "sources", "relevant_pages": [1]},
        {"type": "delta", "text": "First sentence."},
        {"type": "done"},
    ]
    assert output.flush_count == 3


def test_live_failure_is_visible_after_sources():
    output = BytesIO()
    handler = SimpleNamespace(
        send_response=Mock(), send_header=Mock(),
        end_headers=Mock(), wfile=output,
    )

    def events():
        yield {"type": "sources", "relevant_pages": [1]}
        raise LiveConversationError("Gemini Live did not start answering.")

    WorkspaceHandler._ndjson(handler, events())
    lines = [json.loads(line) for line in output.getvalue().splitlines()]
    assert lines == [
        {"type": "sources", "relevant_pages": [1]},
        {"type": "error", "message": "Gemini Live did not start answering."},
    ]
