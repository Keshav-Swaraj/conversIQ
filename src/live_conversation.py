"""Grounded, one-turn Gemini Live speech for the local workspace."""

import asyncio
import base64
from collections.abc import AsyncIterator, Iterator
import logging
from time import perf_counter
from typing import Any, TypedDict

from src.config import GEMINI_API_KEY, GEMINI_LIVE_MODEL, GEMINI_LIVE_VOICE
from src.retriever import RetrievalResult


MAX_EVIDENCE_CHARACTERS = 16_000
FIRST_OUTPUT_TIMEOUT_SECONDS = 25
LOGGER = logging.getLogger(__name__)


class ConversationTurn(TypedDict):
    """One prior question and its spoken answer, scoped to the current document."""

    question: str
    answer: str


def normalize_history(value: object) -> tuple[ConversationTurn, ...]:
    """Validate and bound browser-supplied context before using it in a prompt."""
    if value is None:
        return ()
    if not isinstance(value, list) or len(value) > 8:
        raise ValueError("Conversation history must contain at most eight turns.")
    turns: list[ConversationTurn] = []
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get("question"), str) or not isinstance(item.get("answer"), str):
            raise ValueError("Conversation history must contain question/answer text pairs.")
        question, answer = item["question"].strip(), item["answer"].strip()
        if not question or not answer or len(question) > 500 or len(answer) > 2_000:
            raise ValueError("Conversation history contains an empty or oversized turn.")
        turns.append({"question": question, "answer": answer})
    return tuple(turns)
_SYSTEM_INSTRUCTION = (
    "Answer the user's question only from the supplied document excerpts. "
    "Use prior conversation only to understand references; it is not evidence. "
    "If the excerpts do not support an answer, say so plainly. Do not use "
    "outside knowledge or invent facts. Speak naturally and briefly. "
    "Mention page numbers only when useful."
)
_SPEECH_INSTRUCTION = (
    "You are Sarah, the voice of a document conversation assistant. "
    "Say only the supplied text, verbatim, in a warm natural voice. "
    "Do not add a preface, explanation, or extra words."
)


class LiveConversationError(RuntimeError):
    """A safe, user-facing Gemini Live setup or response failure."""


def source_excerpts(result: RetrievalResult) -> tuple[str, tuple[int, ...]]:
    """Bound selected page text while retaining ordered coverage of every nonempty page."""
    pages = [page for page in result.source_pages if page.text.strip()]
    if not pages:
        return "", ()
    per_page = max(1, (MAX_EVIDENCE_CHARACTERS - len(pages) * 24) // len(pages))
    excerpts = [f"[Page {page.page_number}]\n{page.text[:per_page]}" for page in pages]
    return "\n\n".join(excerpts), tuple(page.page_number for page in pages)


async def _session_events(
    prompt: str, instruction: str, purpose: str,
) -> AsyncIterator[dict[str, Any]]:
    if not GEMINI_API_KEY:
        raise LiveConversationError("GEMINI_API_KEY is missing. Add it to the project-root .env file.")
    try:
        from google import genai
    except ImportError as error:
        raise LiveConversationError("Install the Gemini dependency with uv sync.") from error

    client = genai.Client(api_key=GEMINI_API_KEY)
    audio_received = False
    output_started = False
    started = perf_counter()
    LOGGER.info("Gemini Live %s: model=%s voice=%s prompt_chars=%s", purpose, GEMINI_LIVE_MODEL, GEMINI_LIVE_VOICE, len(prompt))
    config: dict[str, Any] = {
        "response_modalities": ["AUDIO"],
        "output_audio_transcription": {},
        "system_instruction": instruction,
    }
    if GEMINI_LIVE_VOICE:
        config["speech_config"] = {
            "voice_config": {
                "prebuilt_voice_config": {"voice_name": GEMINI_LIVE_VOICE}
            }
        }
    try:
        async with asyncio.timeout(90):
            async with asyncio.timeout(FIRST_OUTPUT_TIMEOUT_SECONDS) as first_output:
                async with client.aio.live.connect(
                    model=GEMINI_LIVE_MODEL,
                    config=config,
                ) as session:
                    await session.send_client_content(
                        turns={"role": "user", "parts": [{"text": prompt}]},
                        turn_complete=True,
                    )
                    async for response in session.receive():
                        content = response.server_content
                        if content is None:
                            continue
                        transcription = content.output_transcription
                        if transcription is not None and transcription.text:
                            if not output_started:
                                first_output.reschedule(None)
                                output_started = True
                                LOGGER.info("Gemini Live first output after %.2f s", perf_counter() - started)
                            yield {"type": "delta", "text": transcription.text}
                        if content.model_turn is not None:
                            for part in content.model_turn.parts or []:
                                blob = part.inline_data
                                if blob is None or not (blob.mime_type or "").startswith("audio/pcm"):
                                    continue
                                samples = blob.data
                                if not isinstance(samples, bytes) or len(samples) % 2:
                                    raise LiveConversationError("Gemini Live returned invalid audio.")
                                if samples:
                                    if not output_started:
                                        first_output.reschedule(None)
                                        output_started = True
                                        LOGGER.info("Gemini Live first output after %.2f s", perf_counter() - started)
                                    audio_received = True
                                    yield {"type": "audio", "pcm_base64": base64.b64encode(samples).decode("ascii")}
                        if content.turn_complete:
                            break
        if not audio_received:
            raise LiveConversationError("Gemini Live returned no spoken answer. Check model access and quota.")
        LOGGER.info("Gemini Live answer complete after %.2f s", perf_counter() - started)
        yield {"type": "done"}
    except TimeoutError as error:
        if output_started:
            message = "Gemini Live stalled before finishing the answer. Please try again."
        else:
            message = (
                f"Gemini Live did not start answering within {FIRST_OUTPUT_TIMEOUT_SECONDS} seconds. "
                "Check its quota or connection."
            )
        LOGGER.warning("Gemini Live timed out after %.2f s", perf_counter() - started)
        raise LiveConversationError(message) from error
    except LiveConversationError:
        raise
    except Exception as error:
        if getattr(error, "code", None) == 429:
            raise LiveConversationError("Gemini Live rate limit reached. Wait and try again.") from error
        raise
    finally:
        client.close()


async def _live_events(
    query: str, evidence: str, pages: tuple[int, ...],
    history: tuple[ConversationTurn, ...] = (),
) -> AsyncIterator[dict[str, Any]]:
    previous = "".join(
        f"User: {turn['question']}\nSarah: {turn['answer']}\n" for turn in history
    )
    prompt = (
        ("Prior conversation for reference only, not evidence:\n" + previous + "\n" if previous else "")
        + f"Current question:\n{query}\n\nDocument excerpts:\n{evidence}\n\n"
        f"Available source pages: {', '.join(map(str, pages))}"
    )
    async for event in _session_events(prompt, _SYSTEM_INSTRUCTION, "answering"):
        yield event


def _sync_events(events: AsyncIterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Bridge one Gemini Live session to the synchronous workspace server."""
    with asyncio.Runner() as runner:
        try:
            while True:
                try:
                    yield runner.run(anext(events))
                except StopAsyncIteration:
                    break
        finally:
            runner.run(events.aclose())


def stream_live_speech(text: str) -> Iterator[dict[str, Any]]:
    """Speak a short Sarah prompt using the same Gemini Live model and voice as answers."""
    prompt = f"Speak exactly this text, without adding anything:\n{text}"
    yield from _sync_events(_session_events(prompt, _SPEECH_INSTRUCTION, "speaking prompt"))


def stream_live_answer(
    query: str, result: RetrievalResult,
    history: tuple[ConversationTurn, ...] = (),
) -> Iterator[dict[str, Any]]:
    """Yield selected sources, Gemini Live transcript deltas, and 24 kHz PCM chunks."""
    evidence, pages = source_excerpts(result)
    yield {"type": "sources", "relevant_pages": list(pages)}
    if not evidence:
        yield {"type": "delta", "text": "I couldn't find relevant text in this document to answer that."}
        yield {"type": "done"}
        return
    events = _live_events(query, evidence, pages, history) if history else _live_events(query, evidence, pages)
    yield from _sync_events(events)
