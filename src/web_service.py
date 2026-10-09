"""Local workspace operations built from the existing ConversIQ layers."""

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Any

import pymupdf
from piper import PiperVoice

from src.config import PROJECT_ROOT
from src.extraction import ExtractedDocument, extract_pdf, generate_doc_id
from src.graph import get_graph
from src.indexer import build_tree
from src.narrator import NarrationState, advance_narration, synthesize_narration_segment
from src.stt import transcribe_audio
from src.tts import load_voice, prepare_spoken_answer, synthesize_speech


UPLOAD_DIR = PROJECT_ROOT / "cache" / "uploads"
AUDIO_DIR = PROJECT_ROOT / "cache" / "tts" / "workspace"
VOICE_PATH = PROJECT_ROOT / "cache" / "voices" / "en_US-lessac-medium.onnx"
_DOC_ID = re.compile(r"[0-9a-f]{16}\Z")


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
    voice_path: Path = VOICE_PATH
    _audio_paths: dict[str, Path] = field(default_factory=dict, init=False)
    _voice: PiperVoice | None = field(default=None, init=False, repr=False)
    _voice_lock: Lock = field(default_factory=Lock, init=False, repr=False)

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

    def narrate(self, doc_id: str, page_number: int, text_offset: int) -> dict[str, Any]:
        """Synthesize or reuse a real Piper segment beginning at a text cursor."""
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
        token = f"segment-{doc_id}-{state.page_number}-{state.text_offset}"
        wav_path = self.audio_dir / f"{token}.wav"
        metadata_path = self.audio_dir / f"{token}.json"
        if wav_path.is_file() and metadata_path.is_file():
            response = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            with self._voice_lock:
                if wav_path.is_file() and metadata_path.is_file():
                    response = json.loads(metadata_path.read_text(encoding="utf-8"))
                else:
                    voice = self._loaded_voice()
                    self.audio_dir.mkdir(parents=True, exist_ok=True)
                    source_text, _ = synthesize_narration_segment(
                        state, self.voice_path, wav_path, max_characters=500,
                        loaded_voice=voice,
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

    def transcribe(self, payload: bytes, extension: str) -> str:
        """Transcribe a browser recording using the existing Groq STT layer."""
        if extension not in {"webm", "ogg", "wav", "mp4"}:
            raise ValueError("Unsupported microphone recording format.")
        if not payload:
            raise ValueError("The microphone recording is empty.")
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=self.audio_dir) as temporary:
            path = Path(temporary) / f"question.{extension}"
            path.write_bytes(payload)
            return transcribe_audio(path)

    def answer_audio(self, answer: str) -> dict[str, str]:
        """Synthesize the spoken version of a grounded answer with Piper."""
        spoken = prepare_spoken_answer(answer)
        if not spoken:
            raise ValueError("There is no answer text to speak.")
        token = "answer-" + sha256(spoken.encode("utf-8")).hexdigest()[:24]
        wav_path = self.audio_dir / f"{token}.wav"
        if not wav_path.is_file():
            with self._voice_lock:
                if wav_path.is_file():
                    self._audio_paths[token] = wav_path
                    return {"audio_url": f"/api/audio?token={token}"}
                voice = self._loaded_voice()
                self.audio_dir.mkdir(parents=True, exist_ok=True)
                synthesize_speech(
                    spoken, self.voice_path, wav_path, loaded_voice=voice
                )
        self._audio_paths[token] = wav_path
        return {"audio_url": f"/api/audio?token={token}"}

    def audio_path(self, token: str) -> Path:
        """Resolve only audio that this service generated during this run."""
        path = self._audio_paths.get(token)
        if path is None or not path.is_file():
            raise FileNotFoundError("Audio is not available; request it again.")
        return path

    def _loaded_voice(self) -> PiperVoice:
        """Load Piper once per service; callers hold ``_voice_lock``."""
        if self._voice is None:
            self._voice = load_voice(self.voice_path)
        return self._voice

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
