"""Tests for local microphone WAV recording without a physical audio device."""

from pathlib import Path
from types import SimpleNamespace
import wave

import pytest

from src import audio


def test_record_microphone_writes_mono_pcm_wav(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[int, int, int, str, bool]] = []
    pcm = b"\x01\x00" * 160

    def fake_rec(frames, *, samplerate, channels, dtype, blocking):
        calls.append((frames, samplerate, channels, dtype, blocking))
        return SimpleNamespace(tobytes=lambda: pcm)

    monkeypatch.setattr(audio.sd, "rec", fake_rec)
    output = tmp_path / "nested" / "question.wav"

    assert audio.record_microphone(output, 0.01) == output
    assert calls == [(160, 16_000, 1, "int16", True)]
    with wave.open(str(output), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16_000
        assert wav_file.readframes(160) == pcm


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_record_microphone_rejects_bad_duration(
    tmp_path: Path, duration: float
) -> None:
    with pytest.raises(ValueError, match="duration_seconds"):
        audio.record_microphone(tmp_path / "question.wav", duration)


def test_record_microphone_reports_device_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(*args, **kwargs):
        raise audio.sd.PortAudioError("No input device")

    monkeypatch.setattr(audio.sd, "rec", unavailable)
    with pytest.raises(RuntimeError, match="default microphone"):
        audio.record_microphone(tmp_path / "question.wav", 1)


@pytest.mark.parametrize("interrupt", [False, True])
def test_play_wav_interruptible_stops_or_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt: bool
) -> None:
    wav_path = tmp_path / "narration.wav"
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(b"\x01\x00" * 4)

    events: list[str] = []
    written: list[bytes] = []

    class FakeStream:
        def __init__(self, **kwargs):
            assert kwargs == {
                "samplerate": 16_000, "channels": 1,
                "dtype": "int16", "blocksize": 2,
            }

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            events.append("close")

        def write(self, block):
            written.append(block)

        def abort(self):
            events.append("abort")

    monkeypatch.setattr(audio.sd, "RawOutputStream", FakeStream)
    completed = audio.play_wav_interruptible(
        wav_path, lambda: interrupt and bool(written), block_frames=2
    )

    assert completed is not interrupt
    assert written == ([b"\x01\x00" * 2] if interrupt else [b"\x01\x00" * 2] * 2)
    assert events == (["abort", "close"] if interrupt else ["close"])


def test_play_wav_interruptible_rejects_non_pcm_wav(tmp_path: Path) -> None:
    wav_path = tmp_path / "eight_bit.wav"
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(1)
        wav_file.setframerate(16_000)
        wav_file.writeframes(b"\x00")
    with pytest.raises(ValueError, match="16-bit PCM"):
        audio.play_wav_interruptible(wav_path, lambda: False)


@pytest.mark.parametrize("input_device", [None, 3])
def test_speech_monitor_detects_sustained_speech_and_keeps_preroll(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, input_device: int | None
) -> None:
    probabilities = iter([0.1] + [0.8] * 6)

    class FakeVad:
        def __init__(self, model_path):
            assert model_path == tmp_path / "silero.onnx"

        def speech_probability(self, frame):
            return next(probabilities)

    class FakeInputStream:
        def __init__(self, **kwargs):
            assert kwargs["samplerate"] == 16_000
            assert kwargs["blocksize"] == 512
            assert kwargs["device"] == input_device
            self.callback = kwargs["callback"]

        def __enter__(self):
            for _ in range(9):
                self.callback(memoryview(b"\x01\x00" * 512), 512, None, False)
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

    monkeypatch.setattr(audio, "SileroVad", FakeVad)
    monkeypatch.setattr(audio.sd, "RawInputStream", FakeInputStream)

    with audio.monitor_microphone(tmp_path / "silero.onnx", input_device) as monitor:
        assert monitor.speech_detected()
        output = monitor.capture_question(tmp_path / "question.wav", 0.064)

    with wave.open(str(output), "rb") as wav_file:
        assert wav_file.getnframes() == 9 * 512
        assert wav_file.getframerate() == 16_000
