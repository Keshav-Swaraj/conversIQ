"""Local workspace integration checks without LLM, STT, or TTS calls."""

from unittest.mock import patch

import pymupdf
import pytest

from src.web_service import WorkspaceService


@pytest.fixture
def pdf_bytes() -> bytes:
    """Build a small real PDF without committing a binary fixture."""
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "A document sentence for the reader.")
        return pdf.tobytes()


def test_upload_returns_real_pages_and_cached_tree_navigation(tmp_path, pdf_bytes):
    service = WorkspaceService(upload_dir=tmp_path / "uploads")
    native_tree = {
        "structure": [{
            "node_id": "0001", "title": "Introduction", "start_index": 1,
            "end_index": 1, "nodes": [],
        }]
    }
    with patch("src.web_service.build_tree", return_value=native_tree) as build:
        response = service.upload("reader.pdf", pdf_bytes)
    assert response["filename"] == "reader.pdf"
    assert response["num_pages"] == 1
    assert response["pages"][0]["page_number"] == 1
    assert "A document sentence" in response["pages"][0]["text"]
    assert response["sections"] == [
        {"id": "0001", "title": "Introduction", "page": 1, "end_page": 1}
    ]
    build.assert_called_once()
    with patch("src.web_service.build_tree", return_value=native_tree):
        assert service.document(response["doc_id"])["pages"] == response["pages"]


def test_upload_rejects_corrupt_pdf_and_non_pdf_name(tmp_path):
    service = WorkspaceService(upload_dir=tmp_path / "uploads")
    with pytest.raises(ValueError, match="Choose a PDF"):
        service.upload("notes.txt", b"not a pdf")
    with pytest.raises(ValueError, match="Unable to open or read PDF"):
        service.upload("broken.pdf", b"not a pdf")


def test_ask_uses_existing_graph_without_duplicating_retrieval(tmp_path, pdf_bytes):
    service = WorkspaceService(upload_dir=tmp_path / "uploads")
    with patch("src.web_service.build_tree", return_value={"structure": []}):
        uploaded = service.upload("reader.pdf", pdf_bytes)
    with patch("src.web_service.get_graph") as get_graph:
        get_graph.return_value.invoke.return_value = {
            "answer": "The document says so.", "relevant_pages": (1,)
        }
        result = service.ask(uploaded["doc_id"], "What does it say?")
    assert result == {"answer": "The document says so.", "relevant_pages": [1]}
    get_graph.return_value.invoke.assert_called_once()


def test_narration_prefetch_reuses_one_loaded_voice(tmp_path, pdf_bytes):
    service = WorkspaceService(
        upload_dir=tmp_path / "uploads",
        audio_dir=tmp_path / "audio",
        voice_path=tmp_path / "voice.onnx",
    )
    with patch("src.web_service.build_tree", return_value={"structure": []}):
        doc_id = service.upload("reader.pdf", pdf_bytes)["doc_id"]

    voice = object()

    def render_segment(state, _voice_path, output_path, _limit=None, **kwargs):
        assert kwargs["loaded_voice"] is voice
        output_path.write_bytes(b"WAV")
        text = state.document.pages[state.page_number - 1].text[state.text_offset:][:10]
        return text, output_path

    def render_answer(_text, _voice_path, output_path, **kwargs):
        assert kwargs["loaded_voice"] is voice
        output_path.write_bytes(b"WAV")
        return output_path

    with (
        patch("src.web_service.load_voice", return_value=voice) as load,
        patch("src.web_service.synthesize_narration_segment", side_effect=render_segment) as render,
        patch("src.web_service.synthesize_speech", side_effect=render_answer),
    ):
        first = service.narrate(doc_id, 1, 0)
        assert service.narrate(doc_id, 1, 0) == first  # cache hit
        service.narrate(doc_id, 1, 10)
        service.answer_audio("A grounded answer.")
    assert render.call_count == 2
    load.assert_called_once_with(service.voice_path)
