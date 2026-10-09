"""HTTP audio responses used by the browser narration player."""

from io import BytesIO
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock

from web_server import WorkspaceHandler


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
