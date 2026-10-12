"""Explicit page-and-character state for document narration."""

from dataclasses import dataclass, replace
from pathlib import Path
import re

from typing import Any

from src.extraction import ExtractedDocument
from src.tts import synthesize_speech
from src.tts_contract import TTSProvider


_TITLES = {"mr", "mrs", "ms", "dr", "prof", "jr", "sr", "st"}
_SENTENCE_END = re.compile(r"[.!?][\"'”’)]*(?=\s|$)")
_INITIALS = re.compile(r"\b(?:[A-Z]\.\s*){2,}(?=[A-Z])")


def _is_abbreviation(text: str, punctuation_index: int, match_end: int) -> bool:
    """Avoid treating initials and common titles as sentence endings."""
    if text[punctuation_index] != ".":
        return False
    token_match = re.search(r"[A-Za-z]+$", text[:punctuation_index])
    if token_match is None:
        return False
    token = token_match.group()
    if token.lower() in _TITLES:
        return True
    following = text[match_end:].lstrip()
    return len(token) == 1 and token.isupper() and bool(following[:1].isupper())


def _speech_text(source_text: str) -> str:
    """Smooth PDF whitespace and initials without changing source offsets."""
    normalized = " ".join(source_text.split())
    return _INITIALS.sub(
        lambda match: re.sub(r"([A-Z])\.", r"\1 ", match.group()).strip() + " ",
        normalized,
    ).replace("  ", " ").strip()


@dataclass(frozen=True)
class NarrationState:
    """Current position in extracted text and whether narration is paused.

    ``page_number`` is 1-based. When narration is complete, it is one past
    ``document.num_pages`` and ``text_offset`` is zero.
    """

    document: ExtractedDocument
    page_number: int = 1
    text_offset: int = 0
    is_paused: bool = False

    @property
    def is_complete(self) -> bool:
        """Whether the cursor has passed the last document page."""
        return self.page_number > self.document.num_pages


def start_narration(document: ExtractedDocument) -> NarrationState:
    """Begin narrating an extracted document from its first page."""
    if document.num_pages != len(document.pages):
        raise ValueError("Document page count does not match its page records.")
    return advance_narration(NarrationState(document=document), 0)


def remaining_page_text(state: NarrationState) -> str:
    """Return exact text following the cursor on its current page."""
    if state.is_complete:
        return ""
    return state.document.pages[state.page_number - 1].text[state.text_offset :]


def advance_narration(state: NarrationState, characters: int) -> NarrationState:
    """Move forward by text characters already played to the listener.

    Empty pages are skipped. Calling with zero characters only skips pages
    with no text. An advance beyond the document or while paused is rejected.
    """
    if state.is_paused:
        raise RuntimeError("Resume narration before advancing its position.")
    if characters < 0:
        raise ValueError("Character count must not be negative.")

    page_index = state.page_number - 1
    text_offset = state.text_offset
    remaining = characters
    while page_index < state.document.num_pages:
        page_text = state.document.pages[page_index].text
        available = len(page_text) - text_offset
        if available <= 0:
            page_index += 1
            text_offset = 0
            continue
        if remaining < available:
            text_offset += remaining
            remaining = 0
            break
        remaining -= available
        page_index += 1
        text_offset = 0

    if remaining:
        raise ValueError("Cannot advance beyond the end of the document.")
    return replace(state, page_number=page_index + 1, text_offset=text_offset)


def pause_narration(state: NarrationState) -> NarrationState:
    """Pause without changing the current text position."""
    return replace(state, is_paused=True)


def resume_narration(state: NarrationState) -> NarrationState:
    """Resume at the exact page and character offset where narration paused."""
    return replace(state, is_paused=False)


def synthesize_narration_segment(
    state: NarrationState,
    output_path: str | Path,
    max_characters: int = 500,
    *,
    provider: TTSProvider | None = None,
    client: Any = None,
) -> tuple[str, Path]:
    """Render a short group of complete sentences without advancing the cursor.

    A long sentence falls back to a word boundary or the character limit.
    Advance by the returned text length only after the entire WAV has played.
    """
    if state.is_paused:
        raise RuntimeError("Resume narration before synthesizing audio.")
    if max_characters <= 0:
        raise ValueError("max_characters must be positive.")
    remaining = remaining_page_text(state)
    text = remaining[:max_characters]
    if not text:
        raise ValueError("No text remains at the current narration position.")
    sentence_end = None
    for match in _SENTENCE_END.finditer(remaining):
        if match.end() > max_characters:
            break
        if not _is_abbreviation(remaining, match.start(), match.end()):
            sentence_end = match
    if sentence_end is not None:
        end = sentence_end.end()
        while end < len(text) and text[end].isspace():
            end += 1
        text = text[:end]
    elif len(remaining) > max_characters:
        word_end = max(text.rfind(" "), text.rfind("\n"), text.rfind("\t"))
        if word_end > 0:
            text = text[: word_end + 1]
    spoken_text = _speech_text(text)
    if provider is not None:
        return text, synthesize_speech(spoken_text, output_path, provider=provider)
    if client is not None:
        return text, synthesize_speech(spoken_text, output_path, client=client)
    return text, synthesize_speech(spoken_text, output_path)
