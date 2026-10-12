"""Provider-neutral TTS selection, normalized WAV output, and speech cleanup."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import logging
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Callable

from src import config
from src.tts_audio import wav_duration
from src.tts_contract import TTSProvider


LOGGER = logging.getLogger("conversiq.tts")
PROVIDER_NAMES = ("cartesia", "piper", "qwen", "kokoro", "gemini", "groq")


@dataclass(frozen=True)
class TTSMetrics:
    """One synthesis attempt, including failures and optional streaming latency."""

    provider: str
    model: str
    characters: int
    words: int
    started_at: str
    first_audio_seconds: float | None
    total_seconds: float
    audio_seconds: float | None
    real_time_factor: float | None
    success: bool
    error: str | None = None


def create_tts_client() -> Any:
    """Create a Cartesia client for callers of the legacy client API."""
    if not config.CARTESIA_API_KEY:
        raise RuntimeError("CARTESIA_API_KEY is missing or empty. Add it to the project-root .env file.")
    from cartesia import Cartesia

    return Cartesia(api_key=config.CARTESIA_API_KEY)


def _configured_voice(name: str) -> tuple[str, str]:
    options = {
        "cartesia": (config.CARTESIA_MODEL, config.CARTESIA_VOICE_ID),
        "piper": (config.PIPER_MODEL_PATH, config.PIPER_MODEL_PATH),
        "qwen": (config.QWEN_MODEL_PATH, config.QWEN_VOICE),
        "kokoro": (config.KOKORO_MODEL_PATH, config.KOKORO_VOICE),
        "gemini": (config.GEMINI_TTS_MODEL, config.GEMINI_TTS_VOICE),
        "groq": (config.GROQ_TTS_MODEL, config.GROQ_TTS_VOICE),
    }
    try:
        return options[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown TTS provider '{name}'. Choose from: {', '.join(PROVIDER_NAMES)}."
        ) from error


def create_tts_provider(name: str | None = None) -> TTSProvider:
    """Select one configured adapter without loading unused model packages."""
    selected = (name or config.TTS_PROVIDER).strip().lower()
    model, voice = _configured_voice(selected)
    if selected == "cartesia":
        from src.tts_adapters.cartesia import CartesiaAdapter

        return CartesiaAdapter(config.CARTESIA_API_KEY, model, voice)
    if selected == "piper":
        from src.tts_adapters.piper import PiperAdapter

        return PiperAdapter(model)
    if selected == "qwen":
        from src.tts_adapters.qwen import QwenAdapter

        return QwenAdapter(model, voice)
    if selected == "kokoro":
        from src.tts_adapters.kokoro import KokoroAdapter

        return KokoroAdapter(model, voice)
    if selected == "gemini":
        from src.tts_adapters.gemini import GeminiAdapter

        return GeminiAdapter(config.GEMINI_API_KEY, model, voice)
    if selected == "groq":
        from src.tts_adapters.groq import GroqAdapter

        return GroqAdapter(config.GROQ_API_KEY, model, voice)
    raise AssertionError("Unregistered TTS provider configuration.")


def provider_cache_key(provider: TTSProvider) -> str:
    """Separate generated audio by provider, model and voice without exposing secrets."""
    return sha256(
        f"{provider.name}\0{provider.model}\0{provider.voice}".encode("utf-8")
    ).hexdigest()[:12]


def configured_tts_cache_identity() -> tuple[str, str]:
    """Return the configured provider and voice key without loading a client."""
    selected = config.TTS_PROVIDER
    model, voice = _configured_voice(selected)
    key = sha256(f"{selected}\0{model}\0{voice}".encode("utf-8")).hexdigest()[:12]
    return selected, key


def prepare_spoken_answer(answer: str) -> str:
    """Remove display-only formatting and citations before speaking an answer."""
    spoken = re.sub(r"【[^】]*】", "", answer)
    spoken = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", spoken)
    spoken = re.sub(r"\((?:pages?|pp?\.)\s+\d[\d,\s–-]*\)", "", spoken, flags=re.I)
    spoken = re.sub(r"(?m)^\s*(?:\d+[.)]|[-*•])\s+", "", spoken)
    spoken = spoken.replace("*", "").replace("_", "").replace("`", "")
    spoken = spoken.replace("≈", "about ")
    spoken = re.sub(r"\s+([.,!?;:])", r"\1", spoken)
    return " ".join(spoken.split())


def synthesize_speech(
    text: str,
    output_path: str | Path,
    *,
    provider: TTSProvider | None = None,
    client: Any = None,
    voice_id: str | None = None,
    model_id: str | None = None,
    on_metrics: Callable[[TTSMetrics], None] | None = None,
) -> Path:
    """Synthesize atomically to a validated WAV using the selected TTS engine.

    ``client``, ``voice_id`` and ``model_id`` retain the older Cartesia-only
    call pattern. New callers should pass a ``provider`` or use TTS_PROVIDER.
    The source output is not replaced if synthesis or validation fails.
    """
    if not text.strip():
        raise ValueError("Text to synthesize must not be empty.")
    wav_path = Path(output_path)
    if wav_path.suffix.lower() != ".wav":
        raise ValueError(f"Output must be a .wav file: {wav_path}")
    if provider is not None and (client is not None or voice_id is not None or model_id is not None):
        raise ValueError("Pass either a provider or Cartesia-specific overrides, not both.")
    if voice_id is not None and not voice_id.strip():
        raise ValueError("Cartesia voice ID must not be empty.")
    if model_id is not None and not model_id.strip():
        raise ValueError("Cartesia model ID must not be empty.")

    owned = provider is None
    if provider is None and (client is not None or voice_id is not None or model_id is not None):
        from src.tts_adapters.cartesia import CartesiaAdapter

        provider = CartesiaAdapter(
            config.CARTESIA_API_KEY,
            model_id or config.CARTESIA_MODEL,
            voice_id or config.CARTESIA_VOICE_ID,
            client=client,
        )
    elif provider is None:
        provider = create_tts_provider()
    started = perf_counter()
    timestamp = datetime.now(timezone.utc).isoformat()
    duration: float | None = None
    failure: str | None = None
    try:
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=wav_path.parent) as temporary:
            candidate = Path(temporary) / "speech.wav"
            provider.synthesize(text, candidate)
            duration = wav_duration(candidate)
            candidate.replace(wav_path)
        return wav_path
    except Exception as error:
        failure = type(error).__name__
        raise
    finally:
        elapsed = perf_counter() - started
        metrics = TTSMetrics(
            provider=provider.name,
            model=provider.model,
            characters=len(text),
            words=len(text.split()),
            started_at=timestamp,
            first_audio_seconds=None,  # Complete-WAV path; not a streamed response.
            total_seconds=elapsed,
            audio_seconds=duration,
            real_time_factor=elapsed / duration if duration else None,
            success=failure is None,
            error=failure,
        )
        LOGGER.info("tts_synthesis %s", json.dumps(asdict(metrics)))
        if on_metrics is not None:
            on_metrics(metrics)
        if owned:
            provider.close()


def play_wav(file_path: str | Path) -> None:
    """Play a local WAV synchronously on Windows, returning when it finishes."""
    wav_path = Path(file_path)
    if not wav_path.is_file():
        raise FileNotFoundError(f"WAV file does not exist: {wav_path}")
    if wav_path.suffix.lower() != ".wav":
        raise ValueError(f"Expected a .wav file: {wav_path}")
    try:
        import winsound
    except ModuleNotFoundError as error:
        raise RuntimeError("Synchronous WAV playback requires Windows.") from error
    winsound.PlaySound(str(wav_path), winsound.SND_FILENAME)
