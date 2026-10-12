"""Tests for explicit narration position and pause/resume behavior."""

from pathlib import Path

import pytest

from src.extraction import ExtractedDocument, ExtractedPage
from src.narrator import (
    advance_narration,
    pause_narration,
    remaining_page_text,
    resume_narration,
    start_narration,
    synthesize_narration_segment,
)


@pytest.fixture
def document() -> ExtractedDocument:
    return ExtractedDocument(
        filename="example.pdf",
        num_pages=3,
        pages=(
            ExtractedPage(1, ""),
            ExtractedPage(2, "Hello"),
            ExtractedPage(3, " world"),
        ),
    )


def test_cursor_skips_blank_pages_and_advances_across_pages(
    document: ExtractedDocument,
) -> None:
    state = start_narration(document)
    assert (state.page_number, state.text_offset) == (2, 0)
    assert remaining_page_text(state) == "Hello"

    state = advance_narration(state, 3)
    assert (state.page_number, state.text_offset) == (2, 3)
    assert remaining_page_text(state) == "lo"

    state = advance_narration(state, 4)
    assert (state.page_number, state.text_offset) == (3, 2)
    assert remaining_page_text(state) == "orld"

    state = advance_narration(state, 4)
    assert state.is_complete
    assert (state.page_number, state.text_offset) == (4, 0)
    assert remaining_page_text(state) == ""


def test_pause_and_resume_keep_exact_position(document: ExtractedDocument) -> None:
    state = advance_narration(start_narration(document), 2)
    paused = pause_narration(state)
    assert paused.is_paused
    assert (paused.page_number, paused.text_offset) == (2, 2)
    assert remaining_page_text(paused) == "llo"
    with pytest.raises(RuntimeError, match="Resume narration"):
        advance_narration(paused, 1)

    resumed = resume_narration(paused)
    assert not resumed.is_paused
    assert (resumed.page_number, resumed.text_offset) == (2, 2)
    assert remaining_page_text(resumed) == "llo"


def test_advance_cannot_leave_document(document: ExtractedDocument) -> None:
    state = start_narration(document)
    with pytest.raises(ValueError, match="beyond the end"):
        advance_narration(state, 12)
    with pytest.raises(ValueError, match="must not be negative"):
        advance_narration(state, -1)


def test_synthesis_preserves_cursor(
    document: ExtractedDocument,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = start_narration(document)
    audio_path = tmp_path / "spoken.wav"
    captured: list[str] = []

    def fake_synthesize(text: str, output_path: str | Path) -> Path:
        captured.append(text)
        return Path(output_path)

    monkeypatch.setattr("src.narrator.synthesize_speech", fake_synthesize)
    spoken_text, path = synthesize_narration_segment(
        state, audio_path, max_characters=3
    )
    assert (spoken_text, path, captured) == ("Hel", audio_path, ["Hel"])
    assert (state.page_number, state.text_offset) == (2, 0)

    with pytest.raises(RuntimeError, match="Resume narration"):
        synthesize_narration_segment(pause_narration(state), audio_path)


def test_page_count_must_match_page_records() -> None:
    invalid = ExtractedDocument("bad.pdf", 2, (ExtractedPage(1, "text"),))
    with pytest.raises(ValueError, match="page count"):
        start_narration(invalid)


@pytest.mark.parametrize(
    ("page_text", "limit", "first_segment", "first_spoken"),
    [
        (
            "One sentence. Another sentence!", 140,
            "One sentence. Another sentence!", "One sentence. Another sentence!",
        ),
        ('He said "Stop!" Then left.', 140, 'He said "Stop!" Then left.', 'He said "Stop!" Then left.'),
        ("One. Two. Three. Four.", 12, "One. Two. ", "One. Two."),
        ("A long sentence without punctuation here", 14, "A long ", "A long"),
        ("Longwordwithoutspaces", 6, "Longwo", "Longwo"),
        (
            "J. K. Rowling wrote it. Mr. Dursley read it.", 140,
            "J. K. Rowling wrote it. Mr. Dursley read it.",
            "J K Rowling wrote it. Mr. Dursley read it.",
        ),
        (
            "J.K. Rowling wrote it. More text.", 140,
            "J.K. Rowling wrote it. More text.", "J K Rowling wrote it. More text.",
        ),
        (
            "Mr. Dursley was home. Then he left.", 140,
            "Mr. Dursley was home. Then he left.",
            "Mr. Dursley was home. Then he left.",
        ),
        (
            "by\nJ. K. Rowling\nIllustrations by Mary", 140,
            "by\nJ. K. Rowling\nIllustrations by Mary",
            "by J K Rowling Illustrations by Mary",
        ),
    ],
)
def test_segments_preserve_every_character_and_advance_only_after_playback(
    page_text: str,
    limit: int,
    first_segment: str,
    first_spoken: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = ExtractedDocument("example.pdf", 1, (ExtractedPage(1, page_text),))
    state = start_narration(document)
    spoken: list[str] = []

    def fake_synthesize(text: str, output_path: str | Path) -> Path:
        spoken.append(text)
        return Path(output_path)

    monkeypatch.setattr("src.narrator.synthesize_speech", fake_synthesize)
    segments: list[str] = []
    while not state.is_complete:
        cursor_before = (state.page_number, state.text_offset)
        segment, _ = synthesize_narration_segment(
            state, tmp_path / "segment.wav", limit
        )
        assert (state.page_number, state.text_offset) == cursor_before
        assert 0 < len(segment) <= limit
        segments.append(segment)
        state = advance_narration(state, len(segment))

    assert segments[0] == first_segment
    assert "".join(segments) == page_text
    assert spoken[0] == first_spoken
    assert len(spoken) == len(segments)
