"""WAV validation and combination shared by optional TTS adapters."""

from pathlib import Path
import wave


def wav_duration(path: Path) -> float:
    """Validate a mono 16-bit PCM WAV and return its duration in seconds."""
    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2 or audio.getnframes() == 0:
            raise ValueError("TTS provider returned an empty or unsupported WAV file.")
        if audio.getcomptype() != "NONE" or audio.getframerate() <= 0:
            raise ValueError("TTS provider returned an unsupported WAV encoding.")
        return audio.getnframes() / audio.getframerate()


def combine_wavs(parts: list[Path], output: Path) -> None:
    """Join same-format WAV parts without altering their order or adding silence."""
    if not parts:
        raise ValueError("There are no WAV parts to combine.")
    expected: tuple[int, int, int] | None = None
    with wave.open(str(output), "wb") as joined:
        for part in parts:
            with wave.open(str(part), "rb") as audio:
                fmt = (audio.getnchannels(), audio.getsampwidth(), audio.getframerate())
                if audio.getcomptype() != "NONE" or fmt[0:2] != (1, 2):
                    raise ValueError("TTS provider returned an unsupported WAV format.")
                if expected is None:
                    expected = fmt
                    joined.setnchannels(fmt[0])
                    joined.setsampwidth(fmt[1])
                    joined.setframerate(fmt[2])
                elif fmt != expected:
                    raise ValueError("Groq returned WAV parts with inconsistent audio formats.")
                joined.writeframes(audio.readframes(audio.getnframes()))
