"""Manual, opt-in comparison of configured TTS providers on identical text."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import wave

from src.tts import PROVIDER_NAMES, TTSMetrics, create_tts_provider, synthesize_speech
from src.tts_audio import wav_duration
from src.tts_contract import TTSSetupError


SAMPLES = (
    "The Riddle House stood on a hill above the village.",
    "Frank paused at the doorway, listening carefully before he stepped inside.",
)


def _stream_once(text: str, output: Path, provider) -> TTSMetrics:
    """Measure one streaming request without duplicating the provider call."""
    from datetime import datetime, timezone

    output.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    started = perf_counter()
    first: float | None = None
    rate: int | None = None
    try:
        with wave.open(str(output), "wb") as audio:
            for chunk in provider.stream(text):
                if chunk.is_final:
                    break
                if not chunk.audio:
                    continue
                if first is None:
                    first = perf_counter() - started
                    rate = chunk.sample_rate
                    audio.setnchannels(1)
                    audio.setsampwidth(2)
                    audio.setframerate(rate)
                if chunk.sample_rate != rate or len(chunk.audio) % 2:
                    raise ValueError("Provider changed audio format during streaming.")
                audio.writeframes(chunk.audio)
        duration = wav_duration(output)
        elapsed = perf_counter() - started
        return TTSMetrics(
            provider.name, provider.model, len(text), len(text.split()), timestamp,
            first, elapsed, duration, elapsed / duration, True,
        )
    except Exception:
        output.unlink(missing_ok=True)
        raise


def main() -> None:
    """Run explicit TTS comparisons; missing optional providers are skipped."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--providers", nargs="+", choices=PROVIDER_NAMES, default=PROVIDER_NAMES)
    parser.add_argument("--text", action="append", help="Use custom text instead of built-in samples")
    parser.add_argument("--output-dir", type=Path, default=Path("cache/tts/benchmark"))
    parser.add_argument("--stream", action="store_true", help="Measure first audio for streaming-capable adapters")
    parser.add_argument("--list", action="store_true", help="Show providers without generating audio")
    args = parser.parse_args()

    if args.list:
        for name in args.providers:
            try:
                provider = create_tts_provider(name)
            except TTSSetupError as error:
                print(f"{name}: unavailable ({error})")
            else:
                print(f"{name}: configured; streaming={provider.supports_streaming}")
                provider.close()
        return

    print("Benchmarking can make paid cloud requests. No fallback provider is used.")
    for name in args.providers:
        try:
            provider = create_tts_provider(name)
        except TTSSetupError as error:
            print(f"{name}: skipped ({error})")
            continue
        try:
            for index, text in enumerate(args.text or SAMPLES, 1):
                output = args.output_dir / f"{name}-{index}.wav"
                try:
                    if args.stream and provider.supports_streaming:
                        result = _stream_once(text, output, provider)
                    else:
                        measurements: list[TTSMetrics] = []
                        synthesize_speech(text, output, provider=provider, on_metrics=measurements.append)
                        result = measurements[0]
                    print(json.dumps({**asdict(result), "output": str(output)}))
                except TTSSetupError as error:
                    print(f"{name}: skipped ({error})")
                    break
                except Exception as error:
                    print(f"{name}: failed ({type(error).__name__}: {error})")
                    break
        finally:
            provider.close()


if __name__ == "__main__":
    main()
