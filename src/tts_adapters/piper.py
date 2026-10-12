"""Optional local Piper adapter."""

from pathlib import Path
from threading import Event
from typing import Any, Iterator
import wave

from src.tts_contract import AudioChunk, TTSSetupError


class PiperAdapter:
    """Load an existing local Piper voice once; never download it automatically."""

    name = "piper"
    is_local = True
    supports_streaming = True

    def __init__(self, model_path: str) -> None:
        path = Path(model_path)
        if not model_path or not path.is_file() or not Path(f"{path}.json").is_file():
            raise TTSSetupError(
                "Piper TTS needs PIPER_MODEL_PATH pointing to an existing .onnx "
                "voice with an adjacent .onnx.json file."
            )
        self.model = str(path.resolve())
        self.voice = path.stem
        self._voice: Any = None
        self._cancel = Event()

    def _get_voice(self) -> Any:
        if self._voice is None:
            try:
                from piper import PiperVoice
            except ImportError as error:
                raise TTSSetupError("Piper is not installed. Run: uv pip install piper-tts") from error
            self._voice = PiperVoice.load(self.model, use_cuda=False)
        return self._voice

    def synthesize(self, text: str, output_path: Path) -> None:
        """Generate the same WAV format as Piper's Python API."""
        voice = self._get_voice()
        with wave.open(str(output_path), "wb") as audio:
            voice.synthesize_wav(text, audio)

    def stream(self, text: str) -> Iterator[AudioChunk]:
        """Yield Piper's sentence audio as mono 16-bit PCM."""
        self._cancel.clear()
        for chunk in self._get_voice().synthesize(text):
            if self._cancel.is_set():
                break
            if chunk.sample_width != 2 or chunk.sample_channels != 1:
                raise ValueError("Piper returned an unsupported PCM format.")
            yield AudioChunk(chunk.audio_int16_bytes, chunk.sample_rate)
        yield AudioChunk(b"", self._get_voice().config.sample_rate, is_final=True)

    def stop(self) -> None:
        """Stop yielding new audio chunks."""
        self._cancel.set()

    def close(self) -> None:
        """Release the loaded voice reference."""
        self._voice = None
