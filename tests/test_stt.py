"""Local tests for the Groq Whisper file transcription boundary."""

from pathlib import Path
from types import SimpleNamespace
import wave

import pytest

from src import stt


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    """Create a small valid WAV file without committing recorded audio."""
    path = tmp_path / "question.wav"
    with wave.open(str(path), "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(16_000)
        recording.writeframes(b"\x00\x00" * 16_000)
    return path


def test_transcribe_audio_uses_groq_whisper(
    audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public API passes the audio stream to the configured model."""
    observed: dict[str, object] = {}

    def fake_groq(*, api_key: str) -> SimpleNamespace:
        observed["api_key_present"] = bool(api_key)

        def create(*, file: tuple[str, object], model: str) -> SimpleNamespace:
            observed["filename"] = file[0]
            observed["header"] = file[1].read(4)
            observed["model"] = model
            return SimpleNamespace(text="  What is a Portkey?  ")

        return SimpleNamespace(
            audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create))
        )

    monkeypatch.setattr(stt, "Groq", fake_groq)

    assert stt.transcribe_audio(audio_file) == "What is a Portkey?"
    assert observed == {
        "api_key_present": True,
        "filename": "question.wav",
        "header": b"RIFF",
        "model": stt.STT_MODEL,
    }


def test_transcribe_audio_can_hint_wake_name(
    audio_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Short wake recordings pass spelling and language context to Whisper."""
    observed: dict[str, object] = {}

    def fake_groq(*, api_key: str) -> SimpleNamespace:
        def create(**options: object) -> SimpleNamespace:
            observed.update(options)
            return SimpleNamespace(text="Sarah")

        return SimpleNamespace(
            audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create))
        )

    monkeypatch.setattr(stt, "Groq", fake_groq)
    assert stt.transcribe_audio(audio_file, prompt="Sarah. Hey Sarah.", language="en") == "Sarah"
    assert observed["prompt"] == "Sarah. Hey Sarah."
    assert observed["language"] == "en"


def test_missing_audio_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Audio file does not exist"):
        stt.transcribe_audio(tmp_path / "missing.wav")


def test_unsupported_audio_extension_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text("not audio", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported audio file extension"):
        stt.transcribe_audio(path)


def test_empty_audio_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.wav"
    path.touch()
    with pytest.raises(ValueError, match="Audio file is empty"):
        stt.transcribe_audio(path)
