"""Transcribe recorded audio with Groq Whisper."""

from pathlib import Path

from groq import Groq

from src.config import GROQ_API_KEY, STT_MODEL


_SUPPORTED_EXTENSIONS = frozenset(
    {".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".wav", ".webm"}
)


def transcribe_audio(file_path: str | Path) -> str:
    """Return the transcript of a supported local audio file.

    The file is sent to Groq's transcription endpoint. An audio file with no
    recognized speech may produce an empty string; provider errors propagate.
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
        transcription = client.audio.transcriptions.create(
            file=(audio_path.name, audio_file),
            model=STT_MODEL,
        )
    return transcription.text.strip()
