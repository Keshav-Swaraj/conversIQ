"""CPU-only Silero ONNX voice activity detection for 16 kHz audio frames."""

from pathlib import Path

import numpy as np
import onnxruntime as ort


class SileroVad:
    """Maintain Silero's recurrent state across 512-sample microphone frames."""

    def __init__(self, model_path: str | Path) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"Silero VAD model does not exist: {model_path}")
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)

    def speech_probability(self, pcm_frame: bytes) -> float:
        """Return speech probability for one 512-sample, mono 16-bit PCM frame."""
        if len(pcm_frame) != 1_024:
            raise ValueError("Silero VAD requires exactly 512 mono int16 samples.")
        samples = np.frombuffer(pcm_frame, dtype="<i2").astype(np.float32)
        samples /= 32_768.0
        audio = np.concatenate((self._context, samples.reshape(1, 512)), axis=1)
        probability, self._state = self._session.run(
            None,
            {
                "input": audio,
                "state": self._state,
                "sr": np.array(16_000, dtype=np.int64),
            },
        )
        self._context = audio[:, -64:]
        return float(probability[0, 0])
