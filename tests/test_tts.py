"""Offline tests for the Cartesia text-to-WAV boundary."""

from pathlib import Path
from types import SimpleNamespace
import sys
import wave

import pytest

from src import tts


def _write_wav(path: Path, *, frames: int = 160, rate: int = 44_100) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x01\x00" * frames)


def test_synthesize_speech_writes_valid_wav_with_selected_voice(tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def generate(**kwargs: object) -> SimpleNamespace:
        observed.update(kwargs)
        return SimpleNamespace(write_to_file=lambda path: _write_wav(Path(path)))

    client = SimpleNamespace(tts=SimpleNamespace(generate=generate))
    output = tmp_path / "nested" / "spoken.wav"
    assert tts.synthesize_speech(
        "Hello, ConversIQ.", output, client=client,
        voice_id="voice-1", model_id="sonic-latest",
    ) == output
    assert observed == {
        "model_id": "sonic-latest",
        "transcript": "Hello, ConversIQ.",
        "voice": "voice-1",
        "language": "en",
        "output_format": {
            "container": "wav", "encoding": "pcm_s16le", "sample_rate": 44_100,
        },
    }
    with wave.open(str(output), "rb") as wav:
        assert wav.getnframes() == 160
        assert wav.getframerate() == 44_100


def test_owned_client_is_closed_after_synthesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[bool] = []
    fake = SimpleNamespace(
        name="cartesia", model="sonic-latest", voice="voice-1",
        synthesize=lambda text, path: _write_wav(path),
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr(tts, "create_tts_provider", lambda: fake)
    tts.synthesize_speech("Hello.", tmp_path / "speech.wav")
    assert closed == [True]


def test_missing_cartesia_key_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tts.config, "CARTESIA_API_KEY", "")
    with pytest.raises(RuntimeError, match=r"CARTESIA_API_KEY.*\.env"):
        tts.create_tts_provider("cartesia")


def test_failed_response_does_not_replace_existing_wav(tmp_path: Path) -> None:
    output = tmp_path / "spoken.wav"
    _write_wav(output)
    before = output.read_bytes()
    fake = SimpleNamespace(tts=SimpleNamespace(generate=lambda **kwargs: SimpleNamespace(
        write_to_file=lambda path: Path(path).write_bytes(b"invalid wav")
    )))
    with pytest.raises((ValueError, wave.Error)):
        tts.synthesize_speech("Hello.", output, client=fake)
    assert output.read_bytes() == before


def test_answer_speech_omits_markdown_and_citation_markers() -> None:
    answer = (
        "1. **The original tragedy** – Three people died. 【7†L9-L20】\n"
        "2. **Later** – Frank overheard a plot (pages 13-14). 【13†L1-L4】"
    )
    assert tts.prepare_spoken_answer(answer) == (
        "The original tragedy – Three people died. "
        "Later – Frank overheard a plot."
    )


def test_invalid_inputs_are_rejected_before_api_call(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        tts.synthesize_speech("  ", tmp_path / "out.wav")
    with pytest.raises(ValueError, match="voice ID"):
        tts.synthesize_speech("Hello", tmp_path / "out.wav", voice_id=" ")
    with pytest.raises(ValueError, match=r"Output must be a \.wav"):
        tts.synthesize_speech("Hello", tmp_path / "out.mp3")


def test_play_wav_uses_synchronous_windows_playback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_path = tmp_path / "answer.wav"
    _write_wav(audio_path)
    calls: list[tuple[str, int]] = []
    monkeypatch.setitem(sys.modules, "winsound", SimpleNamespace(
        SND_FILENAME=1,
        PlaySound=lambda path, flags: calls.append((path, flags)),
    ))
    tts.play_wav(audio_path)
    assert calls == [(str(audio_path), 1)]
