"""Local workspace integration checks without LLM, STT, or TTS calls."""

from unittest.mock import patch

import pymupdf
import pytest

from src.web_service import WorkspaceService


def test_wake_transcription_gets_name_context_without_changing_questions(tmp_path):
    service = WorkspaceService(audio_dir=tmp_path / "audio")
    with patch("src.web_service.transcribe_audio", return_value="Sarah") as transcribe:
        assert service.transcribe(b"wake audio", "webm", wake_word=True) == "Sarah"
        assert transcribe.call_args.kwargs == {
            "prompt": "Sarah. Hey Sarah.", "language": "en",
        }
        assert service.transcribe(b"question audio", "webm") == "Sarah"
        assert transcribe.call_args.kwargs == {}


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


def test_pdf_reading_page_matches_uploaded_page_and_text(tmp_path, pdf_bytes):
    service = WorkspaceService(upload_dir=tmp_path / "uploads")
    with patch("src.web_service.build_tree", return_value={"structure": []}):
        doc_id = service.upload("reader.pdf", pdf_bytes)["doc_id"]
    assert service.pdf_page_image(doc_id, 1).startswith(b"\x89PNG\r\n\x1a\n")
    page = service.pdf_page_words(doc_id, 1)
    assert page["width"] > 0 and page["height"] > 0
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
        extracted_text = pdf[0].get_text(sort=True)
        expected_words = [word[4] for word in pdf[0].get_text("words", sort=True)]
    assert page["words"]
    assert [extracted_text[word["start"]:word["end"]] for word in page["words"]] == expected_words
    with pytest.raises(ValueError, match="outside the document"):
        service.pdf_page_image(doc_id, 2)


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


def test_streamed_answer_reuses_retrieved_source_once(tmp_path, pdf_bytes):
    service = WorkspaceService(upload_dir=tmp_path / "uploads")
    with patch("src.web_service.build_tree", return_value={"structure": []}):
        doc_id = service.upload("reader.pdf", pdf_bytes)["doc_id"]
    retrieved = type("Retrieved", (), {
        "relevant_pages": (1,), "source_text": "[Page 1]\nA document sentence.",
    })()
    with (
        patch("src.web_service.retrieve_document", return_value=retrieved) as retrieve,
        patch("src.web_service.stream_answer", return_value=iter(["A document", " sentence."])) as stream,
    ):
        events = list(service.ask_stream(doc_id, "What does it say?"))
    assert events == [
        {"type": "sources", "relevant_pages": [1]},
        {"type": "delta", "text": "A document"},
        {"type": "delta", "text": " sentence."},
        {"type": "done"},
    ]
    retrieve.assert_called_once()
    stream.assert_called_once_with("What does it say?", retrieved.source_text, (1,))


def test_narration_prefetch_reuses_one_tts_provider(tmp_path, pdf_bytes, monkeypatch):
    monkeypatch.setattr(
        "src.web_service.configured_tts_cache_identity",
        lambda: ("cartesia", "voice-key"),
    )
    service = WorkspaceService(
        upload_dir=tmp_path / "uploads",
        audio_dir=tmp_path / "audio",
    )
    with patch("src.web_service.build_tree", return_value={"structure": []}):
        doc_id = service.upload("reader.pdf", pdf_bytes)["doc_id"]

    provider = type("FakeProvider", (), {
        "name": "cartesia", "model": "sonic-latest", "voice": "voice-1",
    })()

    def render_segment(state, output_path, **kwargs):
        assert kwargs["provider"] is provider
        output_path.write_bytes(b"WAV")
        text = state.document.pages[state.page_number - 1].text[state.text_offset:][:10]
        return text, output_path

    def render_answer(_text, output_path, **kwargs):
        assert kwargs["provider"] is provider
        output_path.write_bytes(b"WAV")
        return output_path

    with (
        patch("src.web_service.create_tts_provider", return_value=provider) as create,
        patch("src.web_service.synthesize_narration_segment", side_effect=render_segment) as render,
        patch("src.web_service.synthesize_speech", side_effect=render_answer),
    ):
        first = service.narrate(doc_id, 1, 0)
        assert service.narrate(doc_id, 1, 0) == first  # cache hit
        service.narrate(doc_id, 1, 10)
        service.answer_audio("A grounded answer.")
    assert render.call_count == 2
    create.assert_called_once_with()
    assert "cartesia" in first["audio_url"]

    # A fresh service instance must read the disk cache without creating a provider.
    restarted = WorkspaceService(upload_dir=service.upload_dir, audio_dir=service.audio_dir)
    with patch("src.web_service.create_tts_provider", side_effect=AssertionError("cache bypass")):
        assert restarted.narrate(doc_id, 1, 0) == first
