"""Small provider contract for normalized ConversIQ speech output."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol


class TTSSetupError(RuntimeError):
    """A selected speech provider lacks its key, model, or optional package."""


@dataclass(frozen=True)
class AudioChunk:
    """A mono 16-bit PCM chunk; ``is_final`` marks the end of a stream."""

    audio: bytes
    sample_rate: int
    is_final: bool = False


class TTSProvider(Protocol):
    """A speech engine that writes browser-playable WAV files."""

    name: str
    model: str
    voice: str
    is_local: bool
    supports_streaming: bool

    def synthesize(self, text: str, output_path: Path) -> None:
        """Write a complete WAV to ``output_path``."""

    def stream(self, text: str) -> Iterator[AudioChunk]:
        """Yield PCM chunks when ``supports_streaming`` is true."""

    def stop(self) -> None:
        """Request cancellation of active streaming, if supported."""

    def close(self) -> None:
        """Release provider resources."""
