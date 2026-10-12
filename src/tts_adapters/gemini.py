"""Optional Google Gemini speech adapter."""

from base64 import b64decode
from pathlib import Path
from threading import Event
from typing import Any, Iterator
import wave

from src.tts_contract import AudioChunk, TTSSetupError


class GeminiAdapter:
    """Use the official Google Gen AI SDK for single-speaker TTS."""

    name = "gemini"
    is_local = False
    supports_streaming = True

    def __init__(self, api_key: str, model: str, voice: str) -> None:
        if not api_key:
            raise TTSSetupError("Gemini TTS is selected but GEMINI_API_KEY is not configured in .env.")
        if not model.strip() or not voice.strip():
            raise ValueError("Gemini TTS model and voice must not be empty.")
        self.model = model
        self.voice = voice
        self._api_key = api_key
        self._client: Any = None
        self._cancel = Event()

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai
            except ImportError as error:
                raise TTSSetupError("Gemini TTS needs google-genai. Run: uv sync --extra gemini --inexact") from error
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _request(self, text: str, *, streaming: bool) -> Any:
        return self._get_client().interactions.create(
            model=self.model,
            input=[{"type": "user_input", "content": [{"type": "text", "text": text}]}],
            response_format={"type": "audio"},
            generation_config={"speech_config": [{"voice": self.voice}]},
            stream=streaming,
        )

    @staticmethod
    def _audio(data: str | None) -> bytes:
        if not data:
            raise ValueError("Gemini TTS returned no audio data.")
        return b64decode(data, validate=True)

    def synthesize(self, text: str, output_path: Path) -> None:
        """Save Gemini's complete WAV response."""
        response = self._request(text, streaming=False)
        data = self._audio(getattr(getattr(response, "output_audio", None), "data", None))
        if data.startswith(b"RIFF"):
            output_path.write_bytes(data)
        else:
            # Explicitly requested raw PCM is still supported by the API.
            with wave.open(str(output_path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(24_000)
                audio.writeframes(data)

    def stream(self, text: str) -> Iterator[AudioChunk]:
        """Yield Gemini's documented 24 kHz mono PCM streaming chunks."""
        self._cancel.clear()
        has_audio = False
        for event in self._request(text, streaming=True):
            if self._cancel.is_set():
                break
            if event.event_type == "error":
                raise RuntimeError(f"Gemini TTS stream failed: {event.error}")
            if event.event_type != "step.delta" or event.delta.type != "audio":
                continue
            has_audio = True
            yield AudioChunk(self._audio(event.delta.data), 24_000)
        if not has_audio and not self._cancel.is_set():
            raise ValueError("Gemini TTS returned no audio data.")
        yield AudioChunk(b"", 24_000, is_final=True)

    def stop(self) -> None:
        """Stop yielding new audio chunks."""
        self._cancel.set()

    def close(self) -> None:
        """Close the Google client if one was created."""
        if self._client is not None:
            self._client.close()
            self._client = None
