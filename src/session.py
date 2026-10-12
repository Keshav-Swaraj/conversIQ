"""Continuous narration with spoken interruptions over one document."""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from src.audio import monitor_microphone, play_wav_interruptible
from src.extraction import extract_pdf, generate_doc_id
from src.graph import get_interrupt_graph
from src.indexer import build_tree
from src.narrator import NarrationState, advance_narration, start_narration, synthesize_narration_segment
from src.tts import create_tts_provider


@dataclass(frozen=True)
class SessionResult:
    """Cursor and work completed by one continuous narration session."""

    narration_state: NarrationState
    segments_attempted: int
    interruptions: int
    last_answer: str | None


def run_continuous_session(
    pdf_path: str | Path,
    vad_model_path: str | Path,
    output_dir: str | Path,
    record_seconds: float = 5.0,
    segment_characters: int = 140,
    max_segments: int | None = None,
    stop_requested: Callable[[], bool] | None = None,
    on_update: Callable[[str], None] | None = None,
    input_device: int | None = None,
) -> SessionResult:
    """Narrate until complete, allowing one spoken interruption per segment.

    An interrupted segment is replayed from its beginning after the answer;
    the text cursor advances only after complete playback. ``max_segments``
    bounds manual demonstrations without changing the document cursor.
    A stop request ends playback at the next safe checkpoint without advancing
    an unfinished segment. Updates are delivered to the caller's thread.
    ``input_device`` selects the local microphone; None uses the default.
    """
    if not isfinite(record_seconds) or record_seconds <= 0:
        raise ValueError("record_seconds must be a finite positive number.")
    if segment_characters <= 0:
        raise ValueError("segment_characters must be positive.")
    if max_segments is not None and max_segments <= 0:
        raise ValueError("max_segments must be positive.")

    def stopping() -> bool:
        return stop_requested is not None and stop_requested()

    def update(message: str) -> None:
        if on_update is not None:
            on_update(message)

    pdf_path = Path(pdf_path)
    vad_model_path = Path(vad_model_path)
    output_dir = Path(output_dir)
    if not vad_model_path.is_file():
        raise FileNotFoundError(f"Silero VAD model does not exist: {vad_model_path}")

    update(f"Opening {pdf_path.name}...")
    document = extract_pdf(pdf_path)
    narration = start_narration(document)
    if narration.is_complete:
        raise ValueError("The PDF has no extractable text to narrate.")

    doc_id = generate_doc_id(pdf_path)
    update(f"Loading or building the document tree for {document.num_pages} pages...")
    build_tree(pdf_path, doc_id)
    if stopping():
        update("Session stopped before narration.")
        return SessionResult(narration, 0, 0, None)
    workflow = get_interrupt_graph()
    update("Preparing speech provider...")
    tts_provider = create_tts_provider()
    attempts = 0
    interruptions = 0
    last_answer: str | None = None
    interrupted_segment: tuple[str, Path] | None = None

    try:
        while (
            not stopping()
            and not narration.is_complete
            and (max_segments is None or attempts < max_segments)
        ):
            update(f"Narrating page {narration.page_number}...")
            if interrupted_segment is None:
                text, wav_path = synthesize_narration_segment(
                    narration, output_dir / "narration.wav", segment_characters,
                    provider=tts_provider,
                )
            else:
                text, wav_path = interrupted_segment
                interrupted_segment = None
            if stopping():
                break
            attempts += 1
            with monitor_microphone(vad_model_path, input_device) as microphone:
                completed = play_wav_interruptible(
                    wav_path, lambda: stopping() or microphone.speech_detected()
                )
                if stopping():
                    break
                if not completed:
                    update("Speech detected; recording your question...")
                question_audio = (
                    None
                    if completed
                    else microphone.capture_question(output_dir / "question.wav", record_seconds)
                )

            if stopping():
                break

            if completed:
                narration = advance_narration(narration, len(text))
                continue

            interruptions += 1
            interrupted_segment = (text, wav_path)
            result = workflow.invoke(
                {
                    "narration_state": narration,
                    "doc_id": doc_id,
                    "file_path": pdf_path,
                    "audio_path": question_audio,
                    "tts_provider": tts_provider,
                    "answer_audio_path": output_dir / "answer.wav",
                }
            )
            narration = result["narration_state"]
            last_answer = result.get("answer")
            if result.get("query"):
                update(f"Question: {result['query']}")
            if last_answer:
                pages = list(result.get("relevant_pages", ()))
                update(f"Answer (pages {pages}): {last_answer}")
            elif result.get("error"):
                update(f"Question status: {result['error']}")
    finally:
        tts_provider.close()

    update("Narration complete." if narration.is_complete else "Session stopped.")
    return SessionResult(narration, attempts, interruptions, last_answer)
