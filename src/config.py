"""Project configuration loaded from the repository-root ``.env`` file."""

from os import getenv
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
"""Absolute path to the ConversIQ project root."""

ENV_FILE: Path = PROJECT_ROOT / ".env"
"""Location of the project-local environment file."""

load_dotenv(ENV_FILE)


def _read_groq_api_key() -> str:
    """Return the required Groq API key or raise a clear configuration error."""
    api_key = getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing or empty. Add it to the project-root .env file."
        )
    return api_key


GROQ_API_KEY: str = _read_groq_api_key()
"""Groq API key loaded from the project-root ``.env`` file."""

PAGEINDEX_MODEL: str = "groq/openai/gpt-oss-20b"
"""Provider-prefixed model for PageIndex-style retrieval reasoning."""

ANSWER_MODEL: str = "openai/gpt-oss-120b"
"""Groq model reserved for future grounded answer generation."""

GROQ_MODEL: str = ANSWER_MODEL
"""Backward-compatible alias for ``ANSWER_MODEL`` used by Notebook 01."""

STT_MODEL: str = "whisper-large-v3-turbo"
"""Groq Whisper model used for Layer 6 speech transcription."""

CARTESIA_API_KEY: str = getenv("CARTESIA_API_KEY", "").strip()
"""Cartesia credential; required only when speech is synthesized."""

CARTESIA_MODEL: str = getenv("CARTESIA_MODEL", "sonic-latest").strip() or "sonic-latest"
"""Cartesia TTS model used for narration and spoken answers."""

CARTESIA_VOICE_ID: str = (
    getenv("CARTESIA_VOICE_ID", "6ccbfb76-1fc6-48f7-b71d-91ac6298247b").strip()
    or "6ccbfb76-1fc6-48f7-b71d-91ac6298247b"
)
"""Default Cartesia voice ID; override in the project-root ``.env`` file."""

TTS_PROVIDER: str = getenv("TTS_PROVIDER", "cartesia").strip().lower() or "cartesia"
"""Active speech provider; Cartesia remains the default."""

PIPER_MODEL_PATH: str = getenv("PIPER_MODEL_PATH", "").strip()
"""Local Piper ``.onnx`` voice path; required only when Piper is selected."""

QWEN_MODEL_PATH: str = getenv("QWEN_MODEL_PATH", "").strip()
"""Existing local Qwen3-TTS CustomVoice model directory (no auto-download)."""

QWEN_VOICE: str = getenv("QWEN_VOICE", "Ryan").strip() or "Ryan"
"""Speaker name for the local Qwen3-TTS CustomVoice model."""

KOKORO_MODEL_PATH: str = getenv("KOKORO_MODEL_PATH", "").strip()
"""Local Kokoro ONNX model path; matching voices file must be adjacent."""

KOKORO_VOICE: str = getenv("KOKORO_VOICE", "af_nicole").strip() or "af_nicole"
"""Voice name for the local Kokoro model."""

GEMINI_API_KEY: str = getenv("GEMINI_API_KEY", "").strip()
"""Gemini credential; required for workspace conversation or Gemini TTS."""

GEMINI_LIVE_MODEL: str = getenv("GEMINI_LIVE_MODEL", "gemini-3.8-live").strip() or "gemini-3.8-live"
"""Gemini Live model used for the workspace's grounded spoken answers."""

GEMINI_LIVE_VOICE: str = getenv("GEMINI_LIVE_VOICE", "").strip()
"""Optional shared Live voice; empty preserves the existing answer voice default."""

GEMINI_TTS_MODEL: str = getenv("GEMINI_TTS_MODEL", "gemini-3.8-flash-lite-tts").strip()
"""Gemini speech model."""

GEMINI_TTS_VOICE: str = getenv("GEMINI_TTS_VOICE", "Kore").strip()
"""Gemini prebuilt speech voice."""

GROQ_TTS_MODEL: str = getenv("GROQ_TTS_MODEL", "canopylabs/orpheus-v1-english").strip()
"""Groq Orpheus speech model, separate from the answer model."""

GROQ_TTS_VOICE: str = getenv("GROQ_TTS_VOICE", "troy").strip()
"""Groq Orpheus speech voice."""

CACHE_DIR: Path = PROJECT_ROOT / "cache" / "trees"
"""Directory for cached document trees."""

SAMPLE_DOCS_DIR: Path = PROJECT_ROOT / "data" / "sample_docs"
"""Directory for project sample documents."""

CACHE_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
