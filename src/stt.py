"""Transcribe recorded audio with Groq Whisper."""

from pathlib import Path

from groq import Groq

from src.config import GROQ_API_KEY, STT_MODEL


_SUPPORTED_EXTENSIONS = frozenset(
    {".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".wav", ".webm"}
)


def transcribe_audio(
    file_path: str | Path, *, prompt: str | None = None, language: str | None = None
) -> str:
    """Return the transcript of a supported local audio file.

    The file is sent to Groq's transcription endpoint. An audio file with no
    recognized speech may produce an empty string; provider errors propagate.
    An optional prompt can supply spelling context for short utterances.
    """
    audio_path = Path(file_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
    if audio_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported audio file extension: {audio_path.suffix or '<none>'}")
    if audio_path.stat().st_size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")

    client = Groq(api_key=GROQ_API_KEY)
    with audio_path.open("rb") as audio_file:
        options = {"file": (audio_path.name, audio_file), "model": STT_MODEL}
        if prompt:
            options["prompt"] = prompt
        if language:
            options["language"] = language
        transcription = client.audio.transcriptions.create(**options)
    return transcription.text.strip()
