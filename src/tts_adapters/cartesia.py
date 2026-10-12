"""Cartesia TTS adapter (the active default provider)."""

from pathlib import Path
from threading import Event
from typing import Any, Iterator

from src.tts_contract import AudioChunk, TTSSetupError


class CartesiaAdapter:
    """Reuse one Cartesia client and produce the existing 44.1 kHz WAV format."""

    name = "cartesia"
    is_local = False
    supports_streaming = True

    def __init__(self, api_key: str, model: str, voice: str, client: Any = None) -> None:
        if not api_key and client is None:
            raise TTSSetupError("Cartesia TTS is selected but CARTESIA_API_KEY is not configured in .env.")
        if not model.strip() or not voice.strip():
            raise ValueError("Cartesia model and voice ID must not be empty.")
        self.model = model
        self.voice = voice
        self._api_key = api_key
        self._client = client
        self._owns_client = client is None
        self._cancel = Event()

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from cartesia import Cartesia
            except ImportError as error:
                raise TTSSetupError("Cartesia is not installed. Run: uv add cartesia") from error
            self._client = Cartesia(api_key=self._api_key)
        return self._client

    def synthesize(self, text: str, output_path: Path) -> None:
        """Write one complete Cartesia WAV without changing the transcript."""
        response = self._get_client().tts.generate(
            model_id=self.model,
            transcript=text,
            voice=self.voice,
            language="en",
            output_format={
                "container": "wav", "encoding": "pcm_s16le", "sample_rate": 44_100,
            },
        )
        response.write_to_file(output_path)

    def stream(self, text: str) -> Iterator[AudioChunk]:
        """Yield Cartesia SSE audio as normalized 44.1 kHz mono PCM chunks."""
        self._cancel.clear()
        events = self._get_client().tts.generate_sse(
            model_id=self.model,
            transcript=text,
            voice=self.voice,
            language="en",
            output_format={
                "container": "raw", "encoding": "pcm_s16le", "sample_rate": 44_100,
            },
        )
        for event in events:
            if self._cancel.is_set():
                break
            if event.type == "chunk" and event.audio:
                yield AudioChunk(event.audio, 44_100)
            elif event.type == "error":
                raise RuntimeError(f"Cartesia streaming failed: {event.title}: {event.message}")
            elif event.type == "done":
                break
        yield AudioChunk(b"", 44_100, is_final=True)

    def stop(self) -> None:
        """Stop yielding new chunks from the current stream."""
        self._cancel.set()

    def close(self) -> None:
        """Close only a client owned by this adapter."""
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None
