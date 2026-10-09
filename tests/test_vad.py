"""Tests for Silero's stateful ONNX input adapter."""

from pathlib import Path

import numpy as np
import pytest

from src import vad


def test_silero_vad_preserves_model_state_and_audio_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "silero.onnx"
    model_path.write_bytes(b"model fixture")
    inputs: list[dict[str, np.ndarray]] = []

    class FakeSession:
        def __init__(self, path, *, sess_options, providers):
            assert path == str(model_path)
            assert providers == ["CPUExecutionProvider"]

        def run(self, output_names, values):
            inputs.append(values)
            return np.array([[0.75]], dtype=np.float32), values["state"] + 1

    monkeypatch.setattr(vad.ort, "InferenceSession", FakeSession)
    detector = vad.SileroVad(model_path)
    frame = (1_000).to_bytes(2, "little", signed=True) * 512

    assert detector.speech_probability(frame) == pytest.approx(0.75)
    assert detector.speech_probability(frame) == pytest.approx(0.75)
    assert inputs[0]["input"].shape == (1, 576)
    assert np.all(inputs[0]["input"][:, :64] == 0)
    assert np.all(inputs[1]["input"][:, :64] > 0)
    assert np.all(inputs[1]["state"] == 1)
    with pytest.raises(ValueError, match="512 mono int16 samples"):
        detector.speech_probability(b"\x00")


def test_silero_vad_requires_a_local_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Silero VAD model"):
        vad.SileroVad(tmp_path / "missing.onnx")
