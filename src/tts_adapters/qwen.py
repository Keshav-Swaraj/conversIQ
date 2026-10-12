"""Optional local Qwen3-TTS CustomVoice adapter."""

from pathlib import Path
from typing import Any, Iterator
import wave

from src.tts_contract import AudioChunk, TTSSetupError


class QwenAdapter:
    """Use a manually installed 0.6B CustomVoice model without auto-downloads."""

    name = "qwen"
    is_local = True
    supports_streaming = False  # The wrapper's non_streaming_mode flag is not audio streaming.

    def __init__(self, model_path: str, voice: str) -> None:
        path = Path(model_path)
        if not model_path or not path.is_dir():
            raise TTSSetupError(
                "Qwen TTS needs QWEN_MODEL_PATH pointing to an existing local "
                "Qwen3-TTS CustomVoice model directory. Download it manually first."
            )
        if not voice.strip():
            raise ValueError("QWEN_VOICE must not be empty.")
        self.model = str(path.resolve())
        self.voice = voice
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                import torch
                from qwen_tts import Qwen3TTSModel
            except ImportError as error:
                raise TTSSetupError(
                    "Qwen TTS needs qwen-tts and torch in this environment. "
                    "Run: uv sync --extra qwen --inexact"
                ) from error
            use_cuda = torch.cuda.is_available()
            self._model = Qwen3TTSModel.from_pretrained(
                self.model,
                device_map="cuda:0" if use_cuda else "cpu",
                dtype=torch.bfloat16 if use_cuda else torch.float32,
                attn_implementation="sdpa",
            )
        return self._model

    def synthesize(self, text: str, output_path: Path) -> None:
        """Generate a local voice sample and write mono 16-bit PCM WAV."""
        import numpy as np

        wavs, sample_rate = self._get_model().generate_custom_voice(
            text=text, language="English", speaker=self.voice,
        )
        if not wavs:
            raise ValueError("Qwen3-TTS returned no audio.")
        pcm = (np.clip(np.asarray(wavs[0]), -1.0, 1.0) * 32767).astype("<i2")
        with wave.open(str(output_path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(int(sample_rate))
            audio.writeframes(pcm.tobytes())

    def stream(self, text: str) -> Iterator[AudioChunk]:
        """Qwen's high-level CustomVoice wrapper does not expose PCM streaming."""
        raise NotImplementedError("Qwen3-TTS CustomVoice streaming is not available in this adapter.")

    def stop(self) -> None:
        """Full local generation cannot currently be cancelled mid-call."""

    def close(self) -> None:
        """Release the model reference and its GPU memory when possible."""
        self._model = None
