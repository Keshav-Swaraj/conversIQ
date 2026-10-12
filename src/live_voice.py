"""Local WebSocket bridge for one grounded Gemini Live voice conversation."""

import asyncio
import base64
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import re
from threading import Event
from typing import Any
from urllib.parse import parse_qs, urlsplit

from google import genai
from google.genai import types
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from src.config import GEMINI_API_KEY, GEMINI_LIVE_MODEL, GEMINI_LIVE_VOICE
from src.live_conversation import source_excerpts
from src.retriever import retrieve_document
from src.web_service import WorkspaceService


LOGGER = logging.getLogger(__name__)
_CONTINUE = re.compile(
    r"^(?:yes(?: please)?|continue|resume|continue (?:reading|narration)|"
    r"resume narration|keep reading|go back to (?:the )?(?:book|narration))[.!?\s]*$",
    re.IGNORECASE,
)
_SMALLTALK = re.compile(
    r"^(?:hi|hello|hey|thanks|thank you|who are you|what(?:'s| is) your name|"
    r"how are you|can you hear me)[.!?\s]*$",
    re.IGNORECASE,
)


@dataclass
class _TurnGate:
    """Withhold document answers until the PageIndex tool has supplied evidence."""

    greeting: bool = True
    pending: bool = False
    lookup_for_turn: bool = False
    input_text: str = ""
    last_query: str = ""
    awaiting_transcript: bool = False
    selected_pages: tuple[int, ...] = ()
    answer_text: str = ""
    buffered: list[dict[str, Any]] = field(default_factory=list)


_INSTRUCTION = (
    "You are Sarah, Keshav's warm, concise voice companion while he reads a PDF. "
    "Greet him briefly when the session begins. Listen naturally and let him interrupt. "
    "Before answering ANY factual question about the PDF, call lookup_document with "
    "the user's current question (include references to earlier turns when needed). "
    "Use ONLY the returned page excerpts or server-provided document excerpts as evidence. "
    "When server-provided excerpts are already in the current turn, do not call lookup_document again. "
    "If the excerpts do not support an "
    "answer, say you could not find it; never use outside knowledge for PDF facts. "
    "In a document answer, mention the specific supporting page numbers naturally, such as "
    "'On page 7...' and 'On page 21...'. Do not cite an entire section range or invent page numbers. "
    "If the user asks to continue or resume narration, call resume_narration. "
    "For small talk about yourself, respond naturally without a document lookup. "
    "Speak in short, conversational sentences without Markdown."
)


def _config() -> types.LiveConnectConfig:
    """Configure audio, transcripts, and blocking document tools for one session."""
    value: dict[str, Any] = {
        "response_modalities": ["AUDIO"],
        "input_audio_transcription": {},
        "output_audio_transcription": {},
        "context_window_compression": {"sliding_window": {}},
        "system_instruction": _INSTRUCTION,
        "tools": [{"function_declarations": [
            {
                "name": "lookup_document",
                "description": "Find authoritative PDF pages before answering any document question. Always use this for document facts.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {"query": {"type": "STRING", "description": "The user's current document question."}},
                    "required": ["query"],
                },
                "behavior": "BLOCKING",
            },
            {
                "name": "resume_narration",
                "description": "End the voice conversation and resume PDF narration at the saved position when the user requests it.",
                "behavior": "BLOCKING",
            },
        ]}],
    }
    if GEMINI_LIVE_VOICE:
        value["speech_config"] = {
            "voice_config": {"prebuilt_voice_config": {"voice_name": GEMINI_LIVE_VOICE}}
        }
    return types.LiveConnectConfig.model_validate(value)


async def _send(connection: ServerConnection, event: dict[str, Any]) -> None:
    await connection.send(json.dumps(event, ensure_ascii=False))


async def _read_browser(
    connection: ServerConnection, session: Any, gate: _TurnGate | None = None
) -> None:
    """Forward browser PCM or typed turns to the active Gemini session."""
    if gate is None:
        gate = _TurnGate()
    async for message in connection:
        if isinstance(message, bytes):
            if not message or len(message) > 65_536 or len(message) % 2:
                raise ValueError("Microphone chunks must be small, nonempty 16-bit PCM frames.")
            await session.send_realtime_input(
                audio=types.Blob(data=message, mime_type="audio/pcm;rate=16000")
            )
            continue
        event = json.loads(message)
        if not isinstance(event, dict):
            raise ValueError("Invalid voice-session message.")
        kind = event.get("type")
        if kind == "speech_start":
            gate.greeting = False
            gate.pending = True
            gate.lookup_for_turn = False
            gate.input_text = ""
            gate.last_query = ""
            gate.awaiting_transcript = True
            gate.selected_pages = ()
            gate.answer_text = ""
            gate.buffered.clear()
            await _send(connection, {"type": "input_start"})
        elif kind == "audio_end":
            await session.send_realtime_input(audio_stream_end=True)
        elif kind == "text":
            text = event.get("text")
            if not isinstance(text, str) or not text.strip() or len(text) > 500:
                raise ValueError("A typed voice question must be 1 to 500 characters.")
            if _CONTINUE.fullmatch(text.strip()):
                await _send(connection, {"type": "resume"})
                return
            gate.greeting = False
            gate.pending = _SMALLTALK.fullmatch(text.strip()) is None
            gate.lookup_for_turn = False
            gate.input_text = ""
            gate.last_query = text.strip()
            gate.awaiting_transcript = False
            gate.selected_pages = ()
            gate.answer_text = ""
            gate.buffered.clear()
            await session.send_client_content(
                turns={"role": "user", "parts": [{"text": text.strip()}]},
                turn_complete=True,
            )
        elif kind == "close":
            return
        else:
            raise ValueError("Unknown voice-session message.")


async def _tool_response(
    connection: ServerConnection, session: Any, service: WorkspaceService,
    doc_id: str, pdf_path: Path, call: Any, gate: _TurnGate | None = None,
) -> bool:
    """Resolve one Gemini tool call; return true if narration should resume."""
    if call.name == "resume_narration":
        await session.send_tool_response(function_responses=types.FunctionResponse(
            id=call.id, name=call.name, response={"status": "resuming"},
        ))
        await _send(connection, {"type": "resume"})
        return True
    if call.name != "lookup_document":
        raise ValueError(f"Unexpected Gemini tool: {call.name}")
    query = (call.args or {}).get("query")
    if not isinstance(query, str) or not query.strip() or len(query) > 500:
        raise ValueError("Gemini requested document lookup without a valid question.")
    result = await asyncio.to_thread(retrieve_document, query.strip(), doc_id, pdf_path)
    evidence, pages = source_excerpts(result)
    await _send(connection, {"type": "sources", "relevant_pages": list(pages)})
    await session.send_tool_response(function_responses=types.FunctionResponse(
        id=call.id,
        name=call.name,
        response={
            "pages": list(pages),
            "source_text": evidence,
            "status": "found" if evidence else "no_relevant_document_text",
        },
    ))
    if gate is not None:
        gate.pending = False
        gate.lookup_for_turn = True
        gate.selected_pages = pages
        gate.answer_text = ""
        gate.buffered.clear()  # Anything produced before evidence was not grounded.
    return False


async def _provide_fallback_evidence(
    connection: ServerConnection, session: Any, doc_id: str,
    pdf_path: Path, query: str, gate: _TurnGate,
) -> None:
    """Ground a missed tool call in PageIndex evidence before requesting spoken output."""
    result = await asyncio.to_thread(retrieve_document, query, doc_id, pdf_path)
    evidence, pages = source_excerpts(result)
    await _send(connection, {"type": "sources", "relevant_pages": list(pages)})
    gate.pending = False
    gate.lookup_for_turn = True
    gate.selected_pages = pages
    gate.answer_text = ""
    gate.buffered.clear()
    await session.send_client_content(
        turns={"role": "user", "parts": [{"text": (
            f"Answer the just-spoken question: {query}\n\n"
            f"Authoritative document excerpts:\n{evidence or '[No relevant document text found]'}\n\n"
            "Speak a brief answer using ONLY these excerpts. Say the specific page numbers "
            "that support each claim, not the whole page range. If the excerpts do not "
            "answer the question, say that the document does not provide the answer. "
            "Do not call lookup_document again for this turn."
        )}]},
        turn_complete=True,
    )


async def _read_gemini(
    connection: ServerConnection, session: Any, service: WorkspaceService,
    doc_id: str, pdf_path: Path, gate: _TurnGate | None = None,
) -> None:
    """Forward spoken output and resolve PageIndex requests inside the same session."""
    if gate is None:
        gate = _TurnGate()
    while True:
        received = False
        # The installed SDK's receive() stops after each model turn. Re-enter it
        # so this one connection survives greeting, tool calls, and follow-ups.
        async for response in session.receive():
            received = True
            if await _forward_response(
                connection, session, service, doc_id, pdf_path, response, gate,
            ):
                return
        if not received:
            return


async def _forward_response(
    connection: ServerConnection, session: Any, service: WorkspaceService,
    doc_id: str, pdf_path: Path, response: Any, gate: _TurnGate,
) -> bool:
    """Forward one Gemini event, returning true for a resume-narration command."""
    if response.tool_call:
        for call in response.tool_call.function_calls or []:
            if await _tool_response(connection, session, service, doc_id, pdf_path, call, gate):
                return True
    content = response.server_content
    if content is None:
        return False
    if content.interrupted:
        await _send(connection, {"type": "interrupted"})
    transcript = content.input_transcription
    if transcript and transcript.text:
        gate.awaiting_transcript = False
        if gate.greeting or (not gate.input_text and not gate.lookup_for_turn):
            gate.greeting = False
            gate.pending = True
        gate.input_text = (
            transcript.text if transcript.text.startswith(gate.input_text)
            else gate.input_text + transcript.text
        )
        await _send(connection, {
            "type": "input", "text": transcript.text,
            "finished": bool(transcript.finished),
        })
        if transcript.finished:
            command = gate.input_text.strip()
            gate.input_text = ""
            gate.last_query = command
            if _CONTINUE.fullmatch(command):
                await _send(connection, {"type": "resume"})
                return True
            if _SMALLTALK.fullmatch(command):
                gate.pending = False
                for event in gate.buffered:
                    await _send(connection, event)
                gate.buffered.clear()
    spoken = content.output_transcription
    if spoken and spoken.text:
        await _output(connection, gate, {"type": "delta", "text": spoken.text})
        if not gate.pending:
            gate.answer_text += spoken.text
    if content.model_turn:
        for part in content.model_turn.parts or []:
            blob = part.inline_data
            if blob and (blob.mime_type or "").startswith("audio/pcm"):
                samples = blob.data
                if not isinstance(samples, bytes) or len(samples) % 2:
                    raise ValueError("Gemini returned malformed PCM audio.")
                if samples:
                    await _output(connection, gate, {
                        "type": "audio", "pcm_base64": base64.b64encode(samples).decode("ascii"),
                    })
    if content.turn_complete:
        if gate.pending:
            gate.buffered.clear()
            query = gate.last_query or gate.input_text.strip()
            if query:
                gate.input_text = ""
                await _provide_fallback_evidence(
                    connection, session, doc_id, pdf_path, query, gate,
                )
                return False
            if gate.awaiting_transcript:
                # A prior model turn may finish while the new utterance is still
                # being transcribed. Keep its grounding gate closed.
                return False
            gate.pending = False
            await _send(connection, {"type": "not_grounded"})
        else:
            cited = tuple(dict.fromkeys(
                int(match.group(1)) for match in re.finditer(
                    r"\bpages?\s+(\d+)\b", gate.answer_text, re.IGNORECASE,
                ) if int(match.group(1)) in gate.selected_pages
            ))
            if cited:
                await _send(connection, {"type": "cited_pages", "relevant_pages": list(cited)})
            await _send(connection, {"type": "done"})
        gate.greeting = False
        gate.lookup_for_turn = False
        gate.input_text = ""
        gate.last_query = ""
        gate.awaiting_transcript = False
        gate.selected_pages = ()
        gate.answer_text = ""
    return False


async def _output(connection: ServerConnection, gate: _TurnGate, event: dict[str, Any]) -> None:
    if gate.pending:
        if len(gate.buffered) < 256:
            gate.buffered.append(event)
    else:
        await _send(connection, event)


async def handle_voice_connection(connection: ServerConnection, service: WorkspaceService) -> None:
    """Run one browser-bound Gemini Live session, closed when the browser leaves."""
    origin = connection.request.headers.get("Origin") if connection.request else None
    if origin and urlsplit(origin).hostname not in {"localhost", "127.0.0.1"}:
        await connection.close(code=1008, reason="Local origin required")
        return
    path = urlsplit(connection.request.path if connection.request else "")
    parameters = parse_qs(path.query)
    try:
        if path.path != "/api/live-voice" or len(parameters.get("doc_id", [])) != 1:
            raise ValueError("A document ID is required for live voice.")
        doc_id = parameters["doc_id"][0]
        pdf_path = service._document_path(doc_id)
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is missing from the project-root .env file.")
        client = genai.Client(api_key=GEMINI_API_KEY)
        try:
            async with client.aio.live.connect(model=GEMINI_LIVE_MODEL, config=_config()) as session:
                await _send(connection, {"type": "ready"})
                await session.send_client_content(
                    turns={"role": "user", "parts": [{"text": "Sarah was just called. Briefly greet Keshav, then listen."}]},
                    turn_complete=True,
                )
                gate = _TurnGate()
                incoming = asyncio.create_task(_read_browser(connection, session, gate))
                outgoing = asyncio.create_task(_read_gemini(connection, session, service, doc_id, pdf_path, gate))
                try:
                    done, pending = await asyncio.wait(
                        {incoming, outgoing}, return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        task.result()
                finally:
                    for task in (incoming, outgoing):
                        task.cancel()
                    await asyncio.gather(incoming, outgoing, return_exceptions=True)
        finally:
            client.close()
    except ConnectionClosed:
        return
    except Exception as error:
        LOGGER.exception("Live voice session failed")
        if connection.state.name == "OPEN":
            await _send(connection, {"type": "error", "message": str(error)})
    finally:
        await connection.close()


async def serve_live_voice(port: int, service: WorkspaceService, ready: Event | None = None) -> None:
    """Serve browser voice sessions on a loopback-only WebSocket port."""
    async with serve(
        lambda connection: handle_voice_connection(connection, service),
        "127.0.0.1", port, max_size=65_536,
    ):
        if ready:
            ready.set()
        await asyncio.Future()
