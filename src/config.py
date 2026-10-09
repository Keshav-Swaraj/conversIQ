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

CACHE_DIR: Path = PROJECT_ROOT / "cache" / "trees"
"""Directory for cached document trees."""

SAMPLE_DOCS_DIR: Path = PROJECT_ROOT / "data" / "sample_docs"
"""Directory for project sample documents."""

CACHE_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
