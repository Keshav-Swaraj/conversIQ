"""Serve the local ConversIQ workspace and its existing Python pipeline."""

import argparse
import asyncio
from collections.abc import Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import mimetypes
from pathlib import Path
import re
from threading import Event, Thread
from urllib.parse import parse_qs, urlsplit

from src.web_service import WorkspaceService
from src.live_conversation import LiveConversationError
from src.live_voice import serve_live_voice
from src.tts_contract import TTSSetupError


DIST_DIR = Path(__file__).resolve().parent / "landing_page" / "dist"
MAX_REQUEST_BYTES = 64 * 1024 * 1024
LOGGER = logging.getLogger(__name__)


class WorkspaceHandler(BaseHTTPRequestHandler):
    """Small, loopback-only HTTP adapter for the local workspace service."""

    service: WorkspaceService

    def do_GET(self) -> None:
        """Return workspace data, generated audio, or the built frontend."""
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/health":
                self._json({"status": "ready"})
            elif parsed.path == "/api/live-voice-config":
                doc_id = self._one(query, "doc_id")
                self.service._document_path(doc_id)
                self._json({
                    "url": f"ws://127.0.0.1:{self.voice_port}/api/live-voice?doc_id={doc_id}",
                })
            elif parsed.path == "/api/document":
                self._json(self.service.document(self._one(query, "id")))
            elif parsed.path == "/api/pdf-page":
                self._png(self.service.pdf_page_image(
                    self._one(query, "id"), int(self._one(query, "page"))
                ))
            elif parsed.path == "/api/pdf-words":
                self._json(self.service.pdf_page_words(
                    self._one(query, "id"), int(self._one(query, "page"))
                ))
            elif parsed.path == "/api/audio":
                self._audio_file(self.service.audio_path(self._one(query, "token")))
            elif parsed.path.startswith("/api/"):
                self._error(HTTPStatus.NOT_FOUND, "Unknown API endpoint.")
            else:
                self._static(parsed.path)
        except (ValueError, FileNotFoundError) as error:
            self._error(
                HTTPStatus.NOT_FOUND if isinstance(error, FileNotFoundError) else HTTPStatus.BAD_REQUEST,
                str(error),
            )
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return  # Browser abandoned an audio request while seeking or closing.
        except Exception:
            LOGGER.exception("Workspace GET failed")
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "The workspace request failed.")

    def do_POST(self) -> None:
        """Accept a bounded local PDF, question, recording, or narration request."""
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).hostname not in {"localhost", "127.0.0.1"}:
            self._error(HTTPStatus.FORBIDDEN, "Only this computer can use the workspace.")
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if not 0 <= length <= MAX_REQUEST_BYTES:
                raise ValueError("Request is empty, too large, or missing its length.")
            body = self.rfile.read(length)
            if parsed.path == "/api/upload":
                if self.headers.get("Content-Type") != "application/pdf":
                    raise ValueError("Upload must be a PDF file.")
                response = self.service.upload(self._one(query, "filename"), body)
            elif parsed.path == "/api/transcribe":
                response = {
                    "transcript": self.service.transcribe(
                        body, self._one(query, "extension"),
                        wake_word=query.get("wake", []) == ["1"],
                    )
                }
            else:
                data = json.loads(body)
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object.")
                if parsed.path == "/api/ask":
                    response = self.service.ask(str(data["doc_id"]), str(data["query"]))
                elif parsed.path == "/api/ask-stream":
                    self._ndjson(self.service.ask_stream(str(data["doc_id"]), str(data["query"])))
                    return
                elif parsed.path == "/api/ask-live-stream":
                    self._ndjson(self.service.ask_live_stream(
                        str(data["doc_id"]), str(data["query"]), data.get("history"),
                    ))
                    return
                elif parsed.path == "/api/sarah-speech":
                    self._ndjson(self.service.sarah_speech(str(data["text"])))
                    return
                elif parsed.path == "/api/narrate":
                    response = self.service.narrate(
                        str(data["doc_id"]), int(data["page"]), int(data["offset"])
                    )
                elif parsed.path == "/api/answer-audio":
                    response = self.service.answer_audio(str(data["answer"]))
                else:
                    self._error(HTTPStatus.NOT_FOUND, "Unknown API endpoint.")
                    return
            self._json(response)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError, TTSSetupError) as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))
        except FileNotFoundError as error:
            self._error(HTTPStatus.NOT_FOUND, str(error))
        except Exception:
            LOGGER.exception("Workspace POST failed")
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "The workspace operation failed. Check the server log.")

    @staticmethod
    def _one(query: dict[str, list[str]], name: str) -> str:
        values = query.get(name, [])
        if len(values) != 1 or not values[0]:
            raise ValueError(f"Missing {name} parameter.")
        return values[0]

    def _json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _ndjson(self, events: Iterator[dict[str, object]]) -> None:
        """Flush each answer event as a JSON line without buffering the full reply."""
        first = next(events)  # Validation/retrieval errors still use the normal JSON error path.
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write((json.dumps(first, ensure_ascii=False) + "\n").encode("utf-8"))
            self.wfile.flush()
            for event in events:
                self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        except LiveConversationError as error:
            LOGGER.warning("Workspace spoken answer unavailable: %s", error)
            try:
                self.wfile.write((json.dumps({"type": "error", "message": str(error)}) + "\n").encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
        except Exception:
            LOGGER.exception("Workspace answer stream failed")
            try:
                self.wfile.write(b'{"type":"error","message":"Answer generation failed. Check the server log."}\n')
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
        finally:
            close = getattr(events, "close", None)
            if close is not None:
                close()

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json({"error": message}, status)

    def _file(self, path: Path, content_type: str) -> None:
        payload = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _png(self, payload: bytes) -> None:
        """Serve a rendered PDF page without exposing the underlying upload path."""
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "private, max-age=3600")
        self.end_headers()
        self.wfile.write(payload)

    def _audio_file(self, path: Path) -> None:
        """Serve WAV bytes with HTTP ranges so browser seeking remains accurate."""
        size = path.stat().st_size
        requested = self.headers.get("Range")
        start, end = 0, size - 1
        if requested:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested.strip())
            if match is None or not any(match.groups()) or size == 0:
                self._invalid_range(size)
                return
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
            else:
                suffix = int(last)
                if suffix <= 0:
                    self._invalid_range(size)
                    return
                start = max(0, size - suffix)
            if start >= size or start > end:
                self._invalid_range(size)
                return
        self.send_response(HTTPStatus.PARTIAL_CONTENT if requested else HTTPStatus.OK)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if requested:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as audio:
            audio.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = audio.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return  # The reader sought elsewhere or closed the page.
                remaining -= len(chunk)

    def _invalid_range(self, size: int) -> None:
        self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
        self.send_header("Content-Range", f"bytes */{size}")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _static(self, request_path: str) -> None:
        if not DIST_DIR.is_dir():
            self._error(HTTPStatus.NOT_FOUND, "Frontend not built. Run npm run build in landing_page.")
            return
        path = (DIST_DIR / request_path.lstrip("/")).resolve()
        if not path.is_relative_to(DIST_DIR.resolve()):
            raise ValueError("Invalid asset path.")
        if not path.is_file():
            path = DIST_DIR / "index.html"
        self._file(path, mimetypes.guess_type(path.name)[0] or "application/octet-stream")


def main() -> None:
    """Start the built UI and API on the local machine only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    WorkspaceHandler.service = WorkspaceService()
    WorkspaceHandler.voice_port = args.port + 1
    voice_ready = Event()
    voice_errors: list[Exception] = []

    def run_voice() -> None:
        try:
            asyncio.run(serve_live_voice(args.port + 1, WorkspaceHandler.service, voice_ready))
        except Exception as error:
            voice_errors.append(error)
            voice_ready.set()

    Thread(target=run_voice, name="conversiq-live-voice", daemon=True).start()
    if not voice_ready.wait(5) or voice_errors:
        raise RuntimeError("Could not start the local Gemini Live voice server.") from (
            voice_errors[0] if voice_errors else None
        )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), WorkspaceHandler)
    print(f"ConversIQ workspace: http://127.0.0.1:{args.port}/workspace")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
