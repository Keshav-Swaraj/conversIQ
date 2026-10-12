"""Optional Groq Orpheus speech adapter."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterator

from src.tts_audio import combine_wavs
from src.tts_contract import AudioChunk, TTSSetupError


def split_orpheus_text(text: str, limit: int = 200) -> list[str]:
    """Split at word boundaries when possible without dropping any character."""
    if limit <= 0:
        raise ValueError("Orpheus text limit must be positive.")
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            boundary = max(text.rfind(" ", start + 1, end + 1), text.rfind("\n", start + 1, end + 1))
            if boundary > start:
                end = boundary + 1
        parts.append(text[start:end])
        start = end
    return parts


class GroqAdapter:
    """Split Orpheus's 200-character requests and join WAVs in source order."""

    name = "groq"
    is_local = False
    supports_streaming = False

    def __init__(self, api_key: str, model: str, voice: str) -> None:
        if not api_key:
            raise TTSSetupError("Groq TTS is selected but GROQ_API_KEY is not configured in .env.")
        if not model.strip() or not voice.strip():
            raise ValueError("Groq TTS model and voice must not be empty.")
        self.model = model
        self.voice = voice
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from groq import Groq

            self._client = Groq(api_key=self._api_key)
        return self._client

    def synthesize(self, text: str, output_path: Path) -> None:
        """Produce one ordered WAV despite Orpheus's per-request text limit."""
        parts = split_orpheus_text(text)
        if not parts:
            raise ValueError("Text to synthesize must not be empty.")
        if len(parts) == 1:
            response = self._get_client().audio.speech.create(
                model=self.model, voice=self.voice, input=parts[0], response_format="wav",
            )
            response.write_to_file(output_path)
            return
        with TemporaryDirectory(dir=output_path.parent) as temporary:
            paths: list[Path] = []
            for index, part in enumerate(parts):
                path = Path(temporary) / f"{index:04d}.wav"
                response = self._get_client().audio.speech.create(
                    model=self.model, voice=self.voice, input=part, response_format="wav",
                )
                response.write_to_file(path)
                paths.append(path)
            combine_wavs(paths, output_path)

    def stream(self, text: str) -> Iterator[AudioChunk]:
        """The Orpheus adapter currently offers only complete WAV generation."""
        raise NotImplementedError("Groq Orpheus streaming is not available in this adapter.")

    def stop(self) -> None:
        """The current synchronous Orpheus request cannot be cancelled mid-call."""

    def close(self) -> None:
        """Close the Groq client if one was created."""
        if self._client is not None:
            self._client.close()
            self._client = None
