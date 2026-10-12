"""Optional local Kokoro ONNX speech adapter."""

from pathlib import Path
from typing import Any, Iterator
import wave

from src.tts_contract import AudioChunk, TTSSetupError


class KokoroAdapter:
    """Load a local English Kokoro model and matching voices only when selected."""

    name = "kokoro"
    is_local = True
    supports_streaming = False

    def __init__(self, model_path: str, voice: str) -> None:
        setup_message = (
            "Kokoro needs KOKORO_MODEL_PATH pointing to kokoro-v1.0.onnx "
            "with voices-v1.0.bin beside it."
        )
        path = Path(model_path)
        if not model_path or not path.is_file():
            raise TTSSetupError(setup_message)
        voices_path = path.with_name("voices-v1.0.bin")
        if not voices_path.is_file():
            raise TTSSetupError(setup_message)
        if not voice.strip():
            raise ValueError("KOKORO_VOICE must not be empty.")
        self.model = str(path.resolve())
        self.voice = voice
        self._voices_path = str(voices_path.resolve())
        self._engine: Any = None

    def _get_engine(self) -> Any:
        if self._engine is None:
            try:
                from kokoro_onnx import Kokoro
            except ImportError as error:
                raise TTSSetupError(
                    "Kokoro needs kokoro-onnx. Run: uv sync --extra kokoro --inexact"
                ) from error
            self._engine = Kokoro(self.model, self._voices_path)
        return self._engine

    def synthesize(self, text: str, output_path: Path) -> None:
        """Generate one complete mono 16-bit PCM WAV from the exact input text."""
        import numpy as np

        samples, sample_rate = self._get_engine().create(
            text, voice=self.voice, lang="en-us"
        )
        audio = np.asarray(samples)
        if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError("Kokoro returned invalid audio samples.")
        if sample_rate <= 0:
            raise ValueError("Kokoro returned an invalid sample rate.")
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
        with wave.open(str(output_path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(int(sample_rate))
            output.writeframes(pcm.tobytes())

    def stream(self, text: str) -> Iterator[AudioChunk]:
        """The current adapter exposes complete WAVs; streaming is not wired yet."""
        raise NotImplementedError("Kokoro streaming is not available in this adapter.")

    def stop(self) -> None:
        """Complete-WAV generation cannot currently be cancelled mid-call."""

    def close(self) -> None:
        """Release the loaded ONNX model reference."""
        self._engine = None
