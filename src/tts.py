"""Synthesize local WAV audio with Piper."""

from pathlib import Path
import re
import wave

from piper import PiperVoice


def _validated_voice_path(voice_path: str | Path) -> Path:
    """Check that a Piper voice and its adjacent configuration are present."""
    model_path = Path(voice_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Piper voice model does not exist: {model_path}")
    if model_path.suffix.lower() != ".onnx":
        raise ValueError(f"Piper voice model must be an .onnx file: {model_path}")
    config_path = model_path.with_suffix(model_path.suffix + ".json")
    if not config_path.is_file():
        raise FileNotFoundError(f"Piper voice configuration does not exist: {config_path}")
    return model_path


def load_voice(voice_path: str | Path) -> PiperVoice:
    """Load one Piper voice for explicit reuse within a narration session."""
    return PiperVoice.load(_validated_voice_path(voice_path), use_cuda=False)


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
    voice_path: str | Path,
    output_path: str | Path,
    *,
    loaded_voice: PiperVoice | None = None,
) -> Path:
    """Write ``text`` to a WAV file using a local Piper voice model.

    Args:
        text: Non-empty text to speak.
        voice_path: Path to a Piper ``.onnx`` model and adjacent ``.onnx.json``.
        output_path: Destination WAV path; its parent directory is created.
        loaded_voice: Optional voice already loaded for this session.

    Returns:
        Path to the generated WAV file.
    """
    if not text.strip():
        raise ValueError("Text to synthesize must not be empty.")

    model_path = _validated_voice_path(voice_path)

    wav_path = Path(output_path)
    if wav_path.suffix.lower() != ".wav":
        raise ValueError(f"Output must be a .wav file: {wav_path}")

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    voice = loaded_voice if loaded_voice is not None else load_voice(model_path)
    with wave.open(str(wav_path), "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
    return wav_path


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
