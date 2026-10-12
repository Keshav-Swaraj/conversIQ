"""Offline contract tests for optional speech engines and provider selection."""

from base64 import b64encode
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import sys
import wave

import pytest

from src import config, tts
from src.tts_adapters.cartesia import CartesiaAdapter
from src.tts_adapters.gemini import GeminiAdapter
from src.tts_adapters.groq import GroqAdapter, split_orpheus_text
from src.tts_adapters.kokoro import KokoroAdapter
from src.tts_adapters.piper import PiperAdapter
from src.tts_adapters.qwen import QwenAdapter
from src.tts_contract import TTSSetupError


def _wav_bytes(rate: int = 24_000) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b"\x01\x00" * 120)
    return output.getvalue()


def test_cartesia_remains_selected_and_cache_key_is_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "TTS_PROVIDER", "cartesia")
    monkeypatch.setattr(config, "CARTESIA_API_KEY", "test-only")
    provider = tts.create_tts_provider()
    assert isinstance(provider, CartesiaAdapter)
    assert provider.name == "cartesia"
    assert tts.configured_tts_cache_identity() == (
        "cartesia", tts.provider_cache_key(provider),
    )


def test_factory_rejects_unknown_provider_and_missing_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="Unknown TTS provider"):
        tts.create_tts_provider("unknown")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    with pytest.raises(TTSSetupError, match="GEMINI_API_KEY"):
        tts.create_tts_provider("gemini")
    monkeypatch.setattr(config, "CARTESIA_API_KEY", "")
    with pytest.raises(TTSSetupError, match="CARTESIA_API_KEY"):
        tts.create_tts_provider("cartesia")


def test_local_providers_fail_clearly_without_model_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "PIPER_MODEL_PATH", "")
    monkeypatch.setattr(config, "QWEN_MODEL_PATH", "")
    monkeypatch.setattr(config, "KOKORO_MODEL_PATH", "")
    with pytest.raises(TTSSetupError, match="PIPER_MODEL_PATH"):
        tts.create_tts_provider("piper")
    with pytest.raises(TTSSetupError, match="QWEN_MODEL_PATH"):
        tts.create_tts_provider("qwen")
    with pytest.raises(TTSSetupError, match="KOKORO_MODEL_PATH"):
        tts.create_tts_provider("kokoro")


def test_provider_registration_and_capabilities(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"fixture")
    Path(f"{model}.json").write_text("{}", encoding="utf-8")
    qwen_model = tmp_path / "qwen-model"
    qwen_model.mkdir()
    kokoro_model = tmp_path / "kokoro-v1.0.onnx"
    kokoro_model.write_bytes(b"fixture")
    (tmp_path / "voices-v1.0.bin").write_bytes(b"fixture")
    monkeypatch.setattr(config, "PIPER_MODEL_PATH", str(model))
    monkeypatch.setattr(config, "QWEN_MODEL_PATH", str(qwen_model))
    monkeypatch.setattr(config, "KOKORO_MODEL_PATH", str(kokoro_model))
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-only")
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-only")
    assert isinstance(tts.create_tts_provider("piper"), PiperAdapter)
    assert isinstance(tts.create_tts_provider("qwen"), QwenAdapter)
    assert isinstance(tts.create_tts_provider("kokoro"), KokoroAdapter)
    assert isinstance(tts.create_tts_provider("gemini"), GeminiAdapter)
    assert isinstance(tts.create_tts_provider("groq"), GroqAdapter)
    assert PiperAdapter.supports_streaming
    assert not QwenAdapter.supports_streaming
    assert not KokoroAdapter.supports_streaming


def test_piper_adapter_passes_exact_text_and_produces_wav(tmp_path: Path) -> None:
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"fixture")
    Path(f"{model}.json").write_text("{}", encoding="utf-8")
    provider = PiperAdapter(str(model))
    seen: list[str] = []

    def synthesize(text: str, audio: wave.Wave_write) -> None:
        seen.append(text)
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(22_050)
        audio.writeframes(b"\x01\x00" * 120)

    provider._voice = SimpleNamespace(synthesize_wav=synthesize)
    output = tts.synthesize_speech("  Keep exact words.  ", tmp_path / "out.wav", provider=provider)
    assert seen == ["  Keep exact words.  "]
    assert output.is_file()


def test_piper_missing_package_error_is_not_hidden_by_wave_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"fixture")
    Path(f"{model}.json").write_text("{}", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "piper", SimpleNamespace())
    with pytest.raises(TTSSetupError, match="piper-tts"):
        tts.synthesize_speech("Hello.", tmp_path / "out.wav", provider=PiperAdapter(str(model)))


def test_qwen_adapter_passes_exact_text_to_custom_voice(tmp_path: Path) -> None:
    import numpy as np

    model_path = tmp_path / "qwen-model"
    model_path.mkdir()
    provider = QwenAdapter(str(model_path), "Ryan")
    calls: list[dict] = []

    def generate_custom_voice(**kwargs):
        calls.append(kwargs)
        return [np.array([0.0, 0.1, -0.1], dtype=np.float32)], 24_000

    provider._model = SimpleNamespace(generate_custom_voice=generate_custom_voice)
    tts.synthesize_speech("Do not paraphrase me.", tmp_path / "qwen.wav", provider=provider)
    assert calls == [{
        "text": "Do not paraphrase me.", "language": "English", "speaker": "Ryan",
    }]


def test_kokoro_adapter_passes_exact_text_and_produces_wav(tmp_path: Path) -> None:
    import numpy as np

    model = tmp_path / "kokoro-v1.0.onnx"
    model.write_bytes(b"fixture")
    (tmp_path / "voices-v1.0.bin").write_bytes(b"fixture")
    provider = KokoroAdapter(str(model), "af_nicole")
    calls: list[tuple[str, str, str]] = []

    def create(text: str, *, voice: str, lang: str):
        calls.append((text, voice, lang))
        return np.array([0.0, 0.1, -0.1], dtype=np.float32), 24_000

    provider._engine = SimpleNamespace(create=create)
    output = tts.synthesize_speech("Do not paraphrase me.", tmp_path / "kokoro.wav", provider=provider)
    assert calls == [("Do not paraphrase me.", "af_nicole", "en-us")]
    with wave.open(str(output), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        assert audio.getframerate() == 24_000


def test_gemini_adapter_uses_audio_model_and_normalizes_wav(tmp_path: Path) -> None:
    provider = GeminiAdapter("test-only", "test-model", "Kore")
    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_audio=SimpleNamespace(data=b64encode(_wav_bytes()).decode()))

    provider._client = SimpleNamespace(interactions=SimpleNamespace(create=create))
    tts.synthesize_speech("Exact input.", tmp_path / "gemini.wav", provider=provider)
    assert calls[0]["input"][0]["content"][0]["text"] == "Exact input."
    assert calls[0]["model"] == "test-model"
    assert calls[0]["generation_config"] == {"speech_config": [{"voice": "Kore"}]}
    assert calls[0]["stream"] is False


def test_gemini_stream_keeps_audio_and_skips_only_metadata() -> None:
    provider = GeminiAdapter("test-only", "test-model", "Kore")
    audio_event = SimpleNamespace(
        event_type="step.delta",
        delta=SimpleNamespace(type="audio", data=b64encode(b"\x01\x00").decode()),
    )
    provider._client = SimpleNamespace(interactions=SimpleNamespace(create=lambda **kwargs: [
        SimpleNamespace(event_type="interaction.created"), audio_event,
    ]))
    chunks = list(provider.stream("Exact input."))
    assert chunks[0].audio == b"\x01\x00"
    assert chunks[-1].is_final


def test_groq_splits_without_losing_text_and_joins_in_order(tmp_path: Path) -> None:
    text = "First sentence. " * 20
    assert "".join(split_orpheus_text(text)) == text
    assert all(len(part) <= 200 for part in split_orpheus_text(text))
    provider = GroqAdapter("test-only", "orpheus", "troy")
    calls: list[str] = []

    def create(**kwargs):
        calls.append(kwargs["input"])
        return SimpleNamespace(write_to_file=lambda path: Path(path).write_bytes(_wav_bytes()))

    provider._client = SimpleNamespace(audio=SimpleNamespace(speech=SimpleNamespace(create=create)))
    tts.synthesize_speech(text, tmp_path / "groq.wav", provider=provider)
    assert calls == split_orpheus_text(text)
    with wave.open(str(tmp_path / "groq.wav"), "rb") as audio:
        assert audio.getnframes() == 120 * len(calls)


def test_failed_provider_preserves_existing_audio_and_records_failure(tmp_path: Path) -> None:
    output = tmp_path / "existing.wav"
    output.write_bytes(_wav_bytes())
    old = output.read_bytes()
    provider = SimpleNamespace(
        name="fake", model="test", voice="test", close=lambda: None,
        synthesize=lambda text, path: (_ for _ in ()).throw(RuntimeError("failed")),
    )
    metrics = []
    with pytest.raises(RuntimeError, match="failed"):
        tts.synthesize_speech("Keep the cursor.", output, provider=provider, on_metrics=metrics.append)
    assert output.read_bytes() == old
    assert not metrics[0].success
    assert metrics[0].characters == len("Keep the cursor.")


def test_cartesia_stream_exposes_pcm_chunks_and_stop() -> None:
    events = [
        SimpleNamespace(type="chunk", audio=b"\x01\x00"),
        SimpleNamespace(type="done"),
    ]
    client = SimpleNamespace(tts=SimpleNamespace(generate_sse=lambda **kwargs: events))
    provider = CartesiaAdapter("", "sonic-latest", "voice", client=client)
    chunks = list(provider.stream("Hello"))
    assert chunks[0].audio == b"\x01\x00"
    assert chunks[-1].is_final
