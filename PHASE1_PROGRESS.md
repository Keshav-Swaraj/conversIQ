# ConverseIQ — Phase 1 Progress & Architecture Documentation

## 1. Overview

**ConverseIQ** is an offline-first, voice-enabled document intelligence platform built on a vectorless RAG architecture optimized for **maximum efficiency and minimal latency**. Phase 1 completes the desktop core document loading using the C-compiled **PyMuPDF** (`fitz`) engine, ultra-fast **RecursiveCharacterTextSplitter**, BM25 sparse candidate retrieval, cross-encoder semantic reranking, grounded Ollama generation, sliding window conversation memory, and LangGraph orchestration.

All components were developed and verified sequentially across 6 Jupyter notebooks before being converted one-to-one into production Python modules in `core/`.

---

## 2. Current Folder Structure

```
updatedConversIQ/
│
├── notebooks/                        # Development & verification notebooks
│   ├── 01_document_processing.ipynb  # Fast-path PDF loading (PyMuPDF) & RecursiveCharacterTextSplitter
│   ├── 02_bm25_retrieval.ipynb       # BM25 sparse retriever setup & query scoring
│   ├── 03_reranking.ipynb            # Cross-encoder (MiniLM) reranking & CPU latency testing
│   ├── 04_llm_generation.ipynb       # Ollama Llama 3 grounded prompts & token streaming
│   ├── 05_memory.ipynb               # Sliding window conversation memory (k=5)
│   └── 06_full_pipeline.ipynb        # LangGraph StateGraph end-to-end pipeline demo
│
├── core/                             # Converted production modules (1:1 mapping)
│   ├── __init__.py                   # Core package exports
│   ├── document.py                   # PDFDocumentProcessor & HierarchicalChunk (PyMuPDF + Recursive Splitter)
│   ├── retriever.py                  # BM25RerankRetriever (BM25 + CrossEncoder)
│   ├── llm.py                        # LLMGenerator (ChatOllama + LCEL chain)
│   ├── memory.py                     # SlidingWindowMemoryManager (k=5)
│   └── pipeline.py                   # ConverseIQPipeline (LangGraph StateGraph)
│
├── tests/                            # Automated test suite
│   ├── test_document.py              # PDF parsing & chunking unit tests
│   ├── test_retriever.py             # BM25 + Reranker unit tests
│   └── test_pipeline.py              # LangGraph pipeline integration tests
│
├── scripts/                          # Utility & helper scripts
│   └── create_sample_pdf.py          # Synthetic PDF report generator for tests
│
├── data/                             # Test data directory
│   └── sample_ai_report.pdf          # Generated sample PDF document
│
├── main.py                           # CLI entry point for Phase 1 demo
├── pyproject.toml                    # UV project configuration & dependencies
├── uv.lock                           # UV lockfile
├── .env                              # Environment variables (OLLAMA_HOST, OLLAMA_MODEL)
├── CONVERSIQ.md                      # Project source-of-truth specification
└── PHASE1_PROGRESS.md                # Phase 1 completion report
```

---

## 3. Completed Features & Modern Best Practices

- ✅ **Ultra-Fast PDF Loading & Extraction**: C-compiled `PyMuPDFLoader` (`pymupdf`) parsing pages in sub-20ms.
- ✅ **High-Efficiency Text Chunking**: `langchain-text-splitters` `RecursiveCharacterTextSplitter` separating text on paragraph/sentence boundaries while maintaining target word limits.
- ✅ **Vectorless BM25 Retrieval**: In-memory sparse indexing via `BM25Retriever`.
- ✅ **Cross-Encoder Reranking**: Re-scoring top-10 sparse candidates using `ms-marco-MiniLM-L-6-v2` down to top-3 semantically aligned context chunks on CPU (<500ms).
- ✅ **Grounded LLM Generation**: ChatOllama integration with Llama 3 using strict negative constraints to prevent hallucinations.
- ✅ **Sliding Window Memory**: Conversation state manager maintaining a 5-turn history window ($k=5$) for multi-turn follow-up queries.
- ✅ **LangGraph StateGraph Pipeline**: Typed node graph (`load` → `retrieve_rerank` → `generate`) orchestrating the RAG execution.
- ✅ **Automated Test Battery**: Full unit and integration test suite passing with `pytest`.

---

## 4. Notebook Summary

| Notebook | Purpose | Implemented Functionality | Verification Status |
| :--- | :--- | :--- | :--- |
| **`01_document_processing.ipynb`** | Document Processing | C-compiled PDF loading (`PyMuPDFLoader`), text cleaning, `RecursiveCharacterTextSplitter`. | ✅ Passed (sub-20ms warm ingestion, no mid-sentence cuts). |
| **`02_bm25_retrieval.ipynb`** | BM25 Retrieval | In-memory `BM25Retriever`, exact vs. paraphrased keyword queries, zero-match handling. | ✅ Passed (exact match rank 1, zero match graceful fallback). |
| **`03_reranking.ipynb`** | Cross-Encoder Reranking | `CrossEncoder` (`ms-marco-MiniLM-L-6-v2`), candidate re-scoring, CPU latency measurement. | ✅ Passed (CPU latency <500ms, candidate score sorting verified). |
| **`04_llm_generation.ipynb`** | LLM Generation | `ChatOllama` Llama 3 connection, grounded prompt templates, off-topic fallback, token streaming. | ✅ Passed (factual answers grounded, off-topic fallback triggered, streaming verified). |
| **`05_memory.ipynb`** | Conversation Memory | `SlidingWindowMemory` ($k=5$), follow-up context resolution, turn eviction, memory reset. | ✅ Passed (turn eviction at $k=5$, clean memory reset). |
| **`06_full_pipeline.ipynb`** | Full Pipeline Demo | LangGraph `StateGraph` pipeline connecting all nodes, 5-question battery execution. | ✅ Passed (all 5 verification queries answered correctly). |

---

## 5. Python Modules

Each notebook mapped directly into a production module in `core/`:

1. **`core/document.py`**: Exports `PDFDocumentProcessor` and `HierarchicalChunk`. Uses C-compiled `PyMuPDFLoader` and `RecursiveCharacterTextSplitter` for maximum CPU throughput and minimal latency.
2. **`core/retriever.py`**: Exports `BM25RerankRetriever`. Combines sparse candidate retrieval (`BM25Retriever`) with deep semantic cross-attention reranking (`CrossEncoder`).
3. **`core/llm.py`**: Exports `LLMGenerator`. Manages `ChatOllama` connections, grounded prompt templates, synchronous generation, and token streaming.
4. **`core/memory.py`**: Exports `SlidingWindowMemoryManager`. Tracks conversation history with automatic $k=5$ turn eviction and prompt formatting.
5. **`core/pipeline.py`**: Exports `ConverseIQPipeline`. Builds and runs the LangGraph `StateGraph` state machine for end-to-end question answering.

---

## 6. Dependencies

| Package | Purpose | Used In |
| :--- | :--- | :--- |
| `uv` | Exclusive Python package manager & virtual environment manager. | Project-wide |
| `pymupdf` (`fitz`) | Ultra-fast C-compiled PDF text parsing (`PyMuPDFLoader`). | Notebook 01, `core/document.py` |
| `langchain-text-splitters` | Fast sentence-boundary aware recursive text chunking (`RecursiveCharacterTextSplitter`). | Notebook 01, `core/document.py` |
| `rank-bm25` / `langchain-community` | In-memory sparse BM25 retriever index. | Notebook 02, `core/retriever.py` |
| `sentence-transformers` | Cross-encoder model (`ms-marco-MiniLM-L-6-v2`) for semantic candidate reranking. | Notebook 03, `core/retriever.py` |
| `langchain-ollama` / `ollama` | Connecting to local Ollama instance running Llama 3. | Notebook 04, `core/llm.py` |
| `langgraph` | Building node-based `StateGraph` RAG pipeline. | Notebook 06, `core/pipeline.py` |
| `pytest` | Automated unit and integration testing. | `tests/` directory |
| `reportlab` | Generating synthetic PDF reports for test suites. | `scripts/create_sample_pdf.py` |

---

## 7. How to Run

### Prerequisites
Ensure **Ollama** is installed and running with model `llama3` pulled:
```bash
ollama pull llama3
ollama serve
```

### Installation & Environment Setup
Using **UV** (no manual virtualenv required):
```bash
# Navigate to workspace
cd updatedConversIQ

# Synchronize dependencies with uv
uv sync
```

### Running Jupyter Notebooks
```bash
uv run jupyter notebook notebooks/01_document_processing.ipynb
```

### Running CLI Demonstration
```bash
uv run python main.py --pdf data/sample_ai_report.pdf
```

### Running Test Suite
```bash
uv run pytest
```

---

## 8. Verification Results

All automated pytest suites passed:
- `tests/test_document.py`: PyMuPDF extraction and `RecursiveCharacterTextSplitter` chunking assertions verified.
- `tests/test_retriever.py`: BM25 candidate retrieval and CrossEncoder reranking latency (<1000ms) verified.
- `tests/test_pipeline.py`: LangGraph end-to-end execution, grounded answers, and off-topic fallback verified.

---

## 9. Current Limitations

As specified in the project roadmap, the following features are intentionally **not implemented** in Phase 1:
- **Voice Pipeline**: Silero VAD, Whisper Speech-to-Text (STT), and Piper Text-to-Speech (TTS) are reserved for Phase 2.
- **REST API**: FastAPI server endpoints (`/upload`, `/query`, `/narrate`) are reserved for Phase 3.
- **Mobile Application**: Flutter mobile application and on-device MediaPipe Gemma 4 integration are reserved for Phase 4.

---

## 10. Next Phase (Phase 2 Preview)

Phase 2 will implement the **Voice Layer** and concurrent processing threads:
- Integration of **Silero VAD** for real-time micro-second voice activity detection.
- Integration of **Whisper STT** for low-latency speech transcription.
- Integration of **Piper TTS** for real-time document narration and answer vocalization.
- Multi-threaded narration & listener workers enabling interruption-based Q&A during document narration.
