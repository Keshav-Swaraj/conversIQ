"""ConversIQ narration and spoken-question entry point."""

import argparse
from math import isfinite
from pathlib import Path

from src.audio import monitor_microphone, play_wav_interruptible, record_microphone
from src.extraction import extract_pdf, generate_doc_id
from src.graph import InterruptState, get_interrupt_graph
from src.indexer import build_tree
from src.narrator import advance_narration, start_narration, synthesize_narration_segment
from src.session import run_continuous_session
from src.tts import play_wav


def run_once(
    pdf_path: str | Path,
    question_audio_path: str | Path | None,
    output_dir: str | Path,
    segment_characters: int = 140,
    record_seconds: float | None = None,
    interrupt_on_enter: bool = False,
    interrupt_on_speech: bool = False,
    vad_model_path: str | Path | None = None,
) -> InterruptState:
    """Narrate a segment, answer one recorded or microphone question, then continue.

    Each narration cursor advances only after its WAV has finished playing.
    In either interrupt mode, stopping a segment leaves its cursor unchanged.
    Narration therefore replays that segment after the answer rather than
    risking skipped text.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    if (question_audio_path is None) == (record_seconds is None):
        raise ValueError("Provide either question_audio_path or record_seconds.")
    if question_audio_path is not None:
        question_audio_path = Path(question_audio_path)
        if not question_audio_path.is_file():
            raise FileNotFoundError(f"Question audio does not exist: {question_audio_path}")
    elif record_seconds is not None and (
        not isfinite(record_seconds) or record_seconds <= 0
    ):
        raise ValueError("record_seconds must be a finite positive number.")
    if segment_characters <= 0:
        raise ValueError("segment_characters must be positive.")
    if interrupt_on_speech and record_seconds is None:
        raise ValueError("Speech interruption requires record_seconds.")
    if interrupt_on_speech and interrupt_on_enter:
        raise ValueError("Choose one interruption mode.")
    model_path = None
    if interrupt_on_speech:
        model_path = (
            Path(vad_model_path)
            if vad_model_path is not None
            else Path(__file__).resolve().parent / "cache" / "vad" / "silero_vad.onnx"
        )
        if not model_path.is_file():
            raise FileNotFoundError(f"Silero VAD model does not exist: {model_path}")

    document = extract_pdf(pdf_path)
    state = start_narration(document)
    if state.is_complete:
        raise ValueError("The PDF has no extractable text to narrate.")

    doc_id = generate_doc_id(pdf_path)
    build_tree(pdf_path, doc_id)

    first_text, first_audio = synthesize_narration_segment(
        state, output_dir / "narration_before.wav", segment_characters
    )
    if interrupt_on_speech:
        with monitor_microphone(model_path) as microphone:
            completed = play_wav_interruptible(first_audio, microphone.speech_detected)
            if not completed:
                question_audio_path = microphone.capture_question(
                    output_dir / "question.wav", record_seconds
                )
    elif interrupt_on_enter:
        completed = play_wav_interruptible(first_audio, _enter_pressed)
    else:
        play_wav(first_audio)
        completed = True
    if completed:
        state = advance_narration(state, len(first_text))

    if interrupt_on_speech and completed:
        return {
            "narration_state": state,
            "doc_id": doc_id,
            "file_path": pdf_path,
            "error": "No speech detected during the narration segment.",
        }

    if record_seconds is not None and not interrupt_on_speech:
        question_audio_path = record_microphone(
            output_dir / "question.wav", record_seconds
        )

    result: InterruptState = get_interrupt_graph().invoke(
        {
            "narration_state": state,
            "doc_id": doc_id,
            "file_path": pdf_path,
            "audio_path": question_audio_path,
            "answer_audio_path": output_dir / "answer.wav",
        }
    )
    state = result["narration_state"]
    if not state.is_complete:
        next_text, next_audio = synthesize_narration_segment(
            state, output_dir / "narration_after.wav", segment_characters
        )
        play_wav(next_audio)
        result["narration_state"] = advance_narration(state, len(next_text))
    return result


def _enter_pressed() -> bool:
    """Consume a pending console key and signal interruption for Enter only."""
    import msvcrt

    return msvcrt.kbhit() and msvcrt.getwch() == "\r"


def main() -> None:
    """Parse CLI paths and run one cycle or a continuous speech session."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True, help="PDF to narrate")
    question_source = parser.add_mutually_exclusive_group(required=True)
    question_source.add_argument(
        "--question-audio", type=Path, help="Prerecorded spoken question"
    )
    question_source.add_argument(
        "--record-seconds", type=float, help="Question recording length in microphone modes"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("cache/tts"), help="Generated WAV directory"
    )
    parser.add_argument("--segment-characters", type=int, default=140)
    interruption = parser.add_mutually_exclusive_group()
    interruption.add_argument(
        "--interrupt-on-enter", action="store_true",
        help="Press Enter during first narration segment to stop it early",
    )
    interruption.add_argument(
        "--interrupt-on-speech", action="store_true",
        help="Use Silero VAD to stop narration when you speak",
    )
    parser.add_argument(
        "--vad-model", type=Path,
        default=Path(__file__).resolve().parent / "cache" / "vad" / "silero_vad.onnx",
        help="Silero ONNX model for --interrupt-on-speech",
    )
    parser.add_argument(
        "--continuous", action="store_true",
        help="Continue through the PDF and allow repeated speech interruptions",
    )
    parser.add_argument(
        "--max-segments", type=int,
        help="Stop continuous mode after this many narration attempts",
    )
    args = parser.parse_args()

    if args.interrupt_on_speech and args.record_seconds is None:
        parser.error("--interrupt-on-speech requires --record-seconds")
    if args.continuous and not args.interrupt_on_speech:
        parser.error("--continuous requires --interrupt-on-speech")
    if args.max_segments is not None and not args.continuous:
        parser.error("--max-segments requires --continuous")

    if args.interrupt_on_enter:
        print(
            "Press Enter during the first narration segment to interrupt it. "
            "An interrupted segment will replay from its start after the answer."
        )
    if args.interrupt_on_speech:
        print(
            "Speech mode: wear headphones to reduce narrator audio leaking into "
            "the microphone. Speak while narration plays."
        )
    if args.record_seconds is not None and not args.interrupt_on_speech:
        print(
            "Microphone mode: speak immediately after narration stops; "
            f"recording lasts {args.record_seconds:g} seconds."
        )

    if args.continuous:
        result = run_continuous_session(
            args.pdf, args.vad_model, args.output_dir,
            args.record_seconds, args.segment_characters, args.max_segments,
        )
        state = result.narration_state
        print(f"Narration complete: {state.is_complete}")
        print(f"Segments attempted: {result.segments_attempted}")
        print(f"Speech interruptions: {result.interruptions}")
        print(f"Narration cursor: page {state.page_number}, character {state.text_offset}")
        if result.last_answer:
            print(f"Last answer: {result.last_answer}")
        return

    result = run_once(
        args.pdf,
        args.question_audio,
        args.output_dir,
        args.segment_characters,
        args.record_seconds,
        args.interrupt_on_enter,
        args.interrupt_on_speech,
        args.vad_model,
    )
    state = result["narration_state"]
    print(f'Question: {result.get("query", "") or "<no speech recognized>"}')
    print(f'Pages: {list(result.get("relevant_pages", ()))}')
    print(f'Answer: {result.get("answer", "<none>")}')
    print(f"Narration cursor: page {state.page_number}, character {state.text_offset}")
    if result.get("error"):
        print(f'Status: {result["error"]}')


if __name__ == "__main__":
    main()
