"""Local tests for the Piper text-to-WAV boundary."""

from pathlib import Path
from types import SimpleNamespace
import sys
import wave

import pytest

from src import tts


@pytest.fixture
def voice_path(tmp_path: Path) -> Path:
    model = tmp_path / "test-voice.onnx"
    model.write_bytes(b"model fixture")
    model.with_suffix(".onnx.json").write_text("{}", encoding="utf-8")
    return model


def test_synthesize_speech_writes_wav(
    voice_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    def load(model_path: Path, *, use_cuda: bool) -> SimpleNamespace:
        observed["model_path"] = model_path
        observed["use_cuda"] = use_cuda

        def synthesize_wav(text: str, wav_file: wave.Wave_write) -> None:
            observed["text"] = text
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16_000)
            wav_file.writeframes(b"\x01\x00" * 160)

        return SimpleNamespace(synthesize_wav=synthesize_wav)

    monkeypatch.setattr(tts, "PiperVoice", SimpleNamespace(load=load))
    output = tmp_path / "nested" / "spoken.wav"

    assert tts.synthesize_speech("Hello, ConversIQ.", voice_path, output) == output
    assert observed == {
        "model_path": voice_path,
        "use_cuda": False,
        "text": "Hello, ConversIQ.",
    }
    with wave.open(str(output), "rb") as wav_file:
        assert wav_file.getnframes() == 160
        assert wav_file.getframerate() == 16_000


def test_explicit_voice_is_loaded_once_for_multiple_segments(
    voice_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loads: list[Path] = []
    spoken: list[str] = []

    def synthesize_wav(text: str, wav_file: wave.Wave_write) -> None:
        spoken.append(text)
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(b"\x00\x00" * 16)

    def load(model_path: Path, *, use_cuda: bool) -> SimpleNamespace:
        loads.append(model_path)
        assert not use_cuda
        return SimpleNamespace(synthesize_wav=synthesize_wav)

    monkeypatch.setattr(tts, "PiperVoice", SimpleNamespace(load=load))
    loaded_voice = tts.load_voice(voice_path)
    tts.synthesize_speech("First sentence.", voice_path, tmp_path / "first.wav", loaded_voice=loaded_voice)
    tts.synthesize_speech("Second sentence.", voice_path, tmp_path / "second.wav", loaded_voice=loaded_voice)

    assert loads == [voice_path]
    assert spoken == ["First sentence.", "Second sentence."]


def test_answer_speech_omits_markdown_and_citation_markers() -> None:
    answer = (
        "1. **The original tragedy** – Three people died. 【7†L9-L20】\n"
        "2. **Later** – Frank overheard a plot (pages 13-14). 【13†L1-L4】"
    )
    assert tts.prepare_spoken_answer(answer) == (
        "The original tragedy – Three people died. "
        "Later – Frank overheard a plot."
    )


def test_empty_text_is_rejected(voice_path: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        tts.synthesize_speech("  ", voice_path, tmp_path / "out.wav")


def test_missing_voice_or_configuration_is_rejected(tmp_path: Path) -> None:
    model = tmp_path / "missing.onnx"
    with pytest.raises(FileNotFoundError, match="voice model does not exist"):
        tts.synthesize_speech("Hello", model, tmp_path / "out.wav")

    model.write_bytes(b"model fixture")
    with pytest.raises(FileNotFoundError, match="voice configuration does not exist"):
        tts.synthesize_speech("Hello", model, tmp_path / "out.wav")


def test_output_must_be_wav(voice_path: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Output must be a .wav file"):
        tts.synthesize_speech("Hello", voice_path, tmp_path / "out.mp3")


def test_play_wav_uses_synchronous_windows_playback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_path = tmp_path / "answer.wav"
    with wave.open(str(audio_path), "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(16_000)
        recording.writeframes(b"\x00\x00" * 160)

    calls: list[tuple[str, int]] = []
    fake_winsound = SimpleNamespace(
        SND_FILENAME=1,
        PlaySound=lambda path, flags: calls.append((path, flags)),
    )
    monkeypatch.setitem(sys.modules, "winsound", fake_winsound)

    tts.play_wav(audio_path)
    assert calls == [(str(audio_path), 1)]
