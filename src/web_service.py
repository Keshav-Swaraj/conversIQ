"""Local workspace operations built from the existing ConversIQ layers."""

from dataclasses import dataclass, field
from collections.abc import Iterator
import base64
from hashlib import sha256
import json
import logging
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from threading import RLock
from time import perf_counter
from typing import Any

import pymupdf

from src.config import GEMINI_LIVE_MODEL, GEMINI_LIVE_VOICE, PROJECT_ROOT
from src.extraction import ExtractedDocument, extract_pdf, generate_doc_id
from src.graph import get_graph
from src.indexer import build_tree
from src.live_conversation import ConversationTurn, normalize_history, stream_live_answer, stream_live_speech
from src.narrator import NarrationState, advance_narration, synthesize_narration_segment
from src.retriever import retrieve_document, stream_answer
from src.stt import transcribe_audio
from src.tts import configured_tts_cache_identity, create_tts_provider, prepare_spoken_answer, synthesize_speech
from src.tts_contract import TTSProvider


UPLOAD_DIR = PROJECT_ROOT / "cache" / "uploads"
AUDIO_DIR = PROJECT_ROOT / "cache" / "tts" / "workspace"
_DOC_ID = re.compile(r"[0-9a-f]{16}\Z")
LOGGER = logging.getLogger(__name__)


def _safe_filename(value: str) -> str:
    """Keep a user-supplied upload name inside the local uploads directory."""
    name = value.replace("\\", "/").split("/")[-1].strip()
    if not name or len(name) > 180 or not name.lower().endswith(".pdf"):
        raise ValueError("Choose a PDF with a valid filename.")
    if name in {".", ".."} or any(char in name for char in '<>:"|?*\x00'):
        raise ValueError("The PDF filename contains unsupported characters.")
    return name


def _sections(tree: dict[str, Any], page_count: int) -> list[dict[str, Any]]:
    """Expose only navigation metadata from the native PageIndex tree."""
    structure = tree.get("structure")
    if not isinstance(structure, list):
        return []
    sections: list[dict[str, Any]] = []
    for node in structure:
        if not isinstance(node, dict):
            continue
        start = node.get("start_index")
        if not isinstance(start, int) or not 1 <= start <= page_count:
            continue
        sections.append({
            "id": str(node.get("node_id", "")),
            "title": str(node.get("title") or f"Page {start}"),
            "page": start,
            "end_page": min(page_count, int(node.get("end_index") or start)),
        })
    return sections


@dataclass
class WorkspaceService:
    """Serve one local user's uploads, grounded questions, and narration audio."""

    upload_dir: Path = UPLOAD_DIR
    audio_dir: Path = AUDIO_DIR
    _audio_paths: dict[str, Path] = field(default_factory=dict, init=False)
    _tts_provider: TTSProvider | None = field(default=None, init=False, repr=False)
    _tts_lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def upload(self, filename: str, payload: bytes) -> dict[str, Any]:
        """Validate and retain an uploaded PDF, then reuse or build its tree."""
        name = _safe_filename(filename)
        if not payload:
            raise ValueError("The selected PDF is empty.")
        # On Windows a failed filename-based PyMuPDF open can keep its temporary
        # file locked. Reject malformed bytes in memory before creating a file.
        try:
            with pymupdf.open(stream=payload, filetype="pdf"):
                pass
        except (OSError, RuntimeError) as error:
            raise ValueError("Unable to open or read PDF upload.") from error
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=self.upload_dir) as temporary:
            temporary_path = Path(temporary) / name
            temporary_path.write_bytes(payload)
            document = extract_pdf(temporary_path)
            doc_id = generate_doc_id(temporary_path)
            target_dir = self.upload_dir / doc_id
            target = target_dir / name
            if target.exists():
                if target.read_bytes() != payload:
                    raise ValueError(
                        "A different PDF has the same filename and size. Rename it and upload again."
                    )
            else:
                target_dir.mkdir(parents=True, exist_ok=True)
                temporary_path.replace(target)
        tree = build_tree(target, doc_id)
        return self._document_response(doc_id, document, tree)

    def document(self, doc_id: str) -> dict[str, Any]:
        """Return extracted pages and cached chapter navigation for an upload."""
        path = self._document_path(doc_id)
        return self._document_response(doc_id, extract_pdf(path), build_tree(path, doc_id))

    def pdf_page_image(self, doc_id: str, page_number: int) -> bytes:
        """Render one uploaded PDF page exactly as a PNG for the reading view."""
        with pymupdf.open(self._document_path(doc_id)) as document:
            if not 1 <= page_number <= len(document):
                raise ValueError("PDF page is outside the document.")
            return document[page_number - 1].get_pixmap(
                matrix=pymupdf.Matrix(2, 2), alpha=False
            ).tobytes("png")

    def pdf_page_words(self, doc_id: str, page_number: int) -> dict[str, Any]:
        """Return PDF word rectangles aligned to the extracted narration text."""
        with pymupdf.open(self._document_path(doc_id)) as document:
            if not 1 <= page_number <= len(document):
                raise ValueError("PDF page is outside the document.")
            page = document[page_number - 1]
            source_text = page.get_text(sort=True)
            cursor = 0
            words: list[dict[str, Any]] = []
            for x0, y0, x1, y1, text, *_ in page.get_text("words", sort=True):
                start = source_text.find(text, cursor)
                if start < 0:
                    continue  # Some PDF glyph encodings cannot be mapped to extracted text.
                cursor = start + len(text)
                words.append({
                    "start": start, "end": cursor,
                    "x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0,
                })
            return {
                "width": page.rect.width,
                "height": page.rect.height,
                "words": words,
            }

    def ask(self, doc_id: str, query: str) -> dict[str, Any]:
        """Run the existing grounded-answer graph for one typed question."""
        if not query.strip():
            raise ValueError("Enter a question about the document.")
        path = self._document_path(doc_id)
        result = get_graph().invoke({
            "doc_id": doc_id,
            "file_path": path,
            "query": query.strip(),
        })
        return {
            "answer": result["answer"],
            "relevant_pages": list(result.get("relevant_pages", ())),
        }

    def ask_stream(self, doc_id: str, query: str) -> Iterator[dict[str, Any]]:
        """Yield source pages followed by grounded answer deltas for the browser."""
        if not query.strip():
            raise ValueError("Enter a question about the document.")
        path = self._document_path(doc_id)
        result = retrieve_document(query.strip(), doc_id, path)
        yield {"type": "sources", "relevant_pages": list(result.relevant_pages)}
        for delta in stream_answer(query.strip(), result.source_text, result.relevant_pages):
            yield {"type": "delta", "text": delta}
        yield {"type": "done"}

    def ask_live_stream(
        self, doc_id: str, query: str, history: object = None
    ) -> Iterator[dict[str, Any]]:
        """Retrieve once, then stream a grounded Gemini Live spoken answer."""
        if not query.strip():
            raise ValueError("Enter a question about the document.")
        turns: tuple[ConversationTurn, ...] = normalize_history(history)
        path = self._document_path(doc_id)
        started = perf_counter()
        retrieval_query = query.strip()
        if turns:
            retrieval_query = (
                "Previous question for context: " + turns[-1]["question"]
                + "\nCurrent question: " + retrieval_query
            )
        result = retrieve_document(retrieval_query, doc_id, path)
        LOGGER.info("Workspace source retrieval completed in %.2f s; pages=%s", perf_counter() - started, len(result.relevant_pages))
        if turns:
            yield from stream_live_answer(query.strip(), result, turns)
        else:
            yield from stream_live_answer(query.strip(), result)

    def sarah_speech(self, text: str) -> Iterator[dict[str, Any]]:
        """Stream and disk-cache short prompts in the same Gemini Live voice as answers."""
        spoken = text.strip()
        if not spoken or len(spoken) > 1_000:
            raise ValueError("Sarah's spoken prompt must be 1 to 1,000 characters.")
        key = sha256(
            f"{GEMINI_LIVE_MODEL}\0{GEMINI_LIVE_VOICE}\0{spoken}".encode("utf-8")
        ).hexdigest()[:24]
        cached = self.audio_dir / f"live-prompt-{key}.pcm"
        yield {"type": "sources", "relevant_pages": []}
        if cached.is_file():
            with cached.open("rb") as audio:
                while samples := audio.read(48_000):
                    yield {"type": "audio", "pcm_base64": base64.b64encode(samples).decode("ascii")}
            yield {"type": "done"}
            return
        chunks: list[bytes] = []
        completed = False
        for event in stream_live_speech(spoken):
            if event["type"] == "audio":
                chunks.append(base64.b64decode(event["pcm_base64"]))
            elif event["type"] == "done":
                completed = True
            yield event
        if completed and chunks:
            self.audio_dir.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(b"".join(chunks))

    def narrate(self, doc_id: str, page_number: int, text_offset: int) -> dict[str, Any]:
        """Synthesize or reuse a provider-specific segment at a text cursor."""
        document = extract_pdf(self._document_path(doc_id))
        if not 1 <= page_number <= document.num_pages:
            raise ValueError("Narration page is outside the document.")
        if not 0 <= text_offset <= len(document.pages[page_number - 1].text):
            raise ValueError("Narration position is outside the page.")
        state = advance_narration(
            NarrationState(document, page_number, text_offset), 0
        )
        if state.is_complete:
            return {"complete": True}
        provider_name, voice_key = configured_tts_cache_identity()
        # The same PDF may gain a corrected extraction order without changing doc_id.
        text_key = sha256(
            state.document.pages[state.page_number - 1].text.encode("utf-8")
        ).hexdigest()[:12]
        token = (
            f"segment-{provider_name}-{voice_key}-"
            f"{doc_id}-{state.page_number}-{state.text_offset}-{text_key}"
        )
        wav_path = self.audio_dir / f"{token}.wav"
        metadata_path = self.audio_dir / f"{token}.json"
        if wav_path.is_file() and metadata_path.is_file():
            response = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            with self._tts_lock:
                if wav_path.is_file() and metadata_path.is_file():
                    response = json.loads(metadata_path.read_text(encoding="utf-8"))
                else:
                    provider = self._provider()
                    self.audio_dir.mkdir(parents=True, exist_ok=True)
                    source_text, _ = synthesize_narration_segment(
                        state, wav_path, max_characters=500, provider=provider,
                    )
                    next_state = advance_narration(state, len(source_text))
                    response = {
                        "complete": False,
                        "page": state.page_number,
                        "offset": state.text_offset,
                        "text": source_text,
                        "next_page": next_state.page_number,
                        "next_offset": next_state.text_offset,
                        "next_complete": next_state.is_complete,
                        "audio_url": f"/api/audio?token={token}",
                    }
                    metadata_path.write_text(json.dumps(response), encoding="utf-8")
        self._audio_paths[token] = wav_path
        return response

    def transcribe(self, payload: bytes, extension: str, *, wake_word: bool = False) -> str:
        """Transcribe browser audio, giving Whisper spelling context for Sarah only."""
        if extension not in {"webm", "ogg", "wav", "mp4"}:
            raise ValueError("Unsupported microphone recording format.")
        if not payload:
            raise ValueError("The microphone recording is empty.")
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=self.audio_dir) as temporary:
            path = Path(temporary) / f"question.{extension}"
            path.write_bytes(payload)
            if wake_word:
                return transcribe_audio(path, prompt="Sarah. Hey Sarah.", language="en")
            return transcribe_audio(path)

    def answer_audio(self, answer: str) -> dict[str, str]:
        """Synthesize the spoken version of a grounded answer."""
        spoken = prepare_spoken_answer(answer)
        if not spoken:
            raise ValueError("There is no answer text to speak.")
        provider_name, voice_key = configured_tts_cache_identity()
        token = f"answer-{provider_name}-{voice_key}-" + sha256(
            spoken.encode("utf-8")
        ).hexdigest()[:24]
        wav_path = self.audio_dir / f"{token}.wav"
        if not wav_path.is_file():
            with self._tts_lock:
                if wav_path.is_file():
                    self._audio_paths[token] = wav_path
                    return {"audio_url": f"/api/audio?token={token}"}
                provider = self._provider()
                self.audio_dir.mkdir(parents=True, exist_ok=True)
                synthesize_speech(spoken, wav_path, provider=provider)
        self._audio_paths[token] = wav_path
        return {"audio_url": f"/api/audio?token={token}"}

    def audio_path(self, token: str) -> Path:
        """Resolve only audio that this service generated during this run."""
        path = self._audio_paths.get(token)
        if path is None or not path.is_file():
            raise FileNotFoundError("Audio is not available; request it again.")
        return path

    def _provider(self) -> TTSProvider:
        """Reuse one configured speech engine for prefetch and answers."""
        if self._tts_provider is None:
            with self._tts_lock:
                if self._tts_provider is None:
                    self._tts_provider = create_tts_provider()
        return self._tts_provider

    def _document_path(self, doc_id: str) -> Path:
        if not _DOC_ID.fullmatch(doc_id):
            raise ValueError("Invalid document identifier.")
        directory = self.upload_dir / doc_id
        paths = list(directory.glob("*.pdf")) if directory.is_dir() else []
        if len(paths) != 1:
            raise FileNotFoundError("Document not found. Upload the PDF again.")
        return paths[0]

    @staticmethod
    def _document_response(
        doc_id: str, document: ExtractedDocument, tree: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "doc_id": doc_id,
            "filename": document.filename,
            "num_pages": document.num_pages,
            "sections": _sections(tree, document.num_pages),
            "pages": [
                {"page_number": page.page_number, "text": page.text}
                for page in document.pages
            ],
        }
