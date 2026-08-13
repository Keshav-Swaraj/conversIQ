# ConverseIQ

> **AI-powered, voice-first document learning system.** Upload a PDF. Listen to it narrated. Interrupt with a question. Get a grounded answer. Resume narration.

---

## What This Is

ConverseIQ transforms static PDF documents into interactive conversational experiences. No cloud. No vector database. No GPU required. Runs entirely offline on consumer hardware.

**Core research contributions:**

1. **Hierarchical Vectorless RAG** — BM25 sparse retrieval + cross-encoder reranking across section → paragraph → sentence levels. No FAISS. No embeddings. No vector DB.
2. **Fully Offline + Mobile-First** — Complete pipeline on-device. Whisper STT + Gemma 4 / Llama 3 LLM + Piper TTS. Works in airplane mode.

---

## How It Works

```
User uploads PDF
      ↓
Text extracted → Hierarchical chunks (Section → Paragraph → Sentence)
      ↓
BM25 index built in-memory (no embeddings, no vectors)
      ↓
┌─────────────────────────────────┐
│  RUNTIME — Two concurrent threads │
│                                   │
│  Narration Thread                 │
│  TTS reads document aloud ──────► │──► Speaker
│  Tracks char-level position       │
│                                   │
│  Interaction Thread               │
│  Silero VAD listens ──────────── │◄── Microphone
│  → Whisper STT transcribes        │
│  → BERT classifies intent         │
│  → BM25 retrieves candidates      │
│  → Cross-encoder reranks top 3    │
│  → LLM generates grounded answer  │
│  → TTS speaks answer              │
│  → Narration resumes              │
└─────────────────────────────────┘

```

---

## Development Approach

> **This project is developed in Jupyter notebooks first, then converted to Python modules.**

### Why notebooks for development

- Models load once per kernel session — no cold start on every test
- Each pipeline step is inspectable inline — see chunks, scores, answers immediately
- Faster iteration — change a prompt, re-run one cell, not the whole file
- BM25 index and LLM connection stay alive between cells

### When to convert to .py

- Each notebook phase is fully working and verified
- All verification checks pass
- Convert notebook cells → Python module functions one-to-one
- The final .py structure mirrors the notebook structure exactly

---

## Tech Stack

| Layer Technology Why  |                                                                   |                                                              |
| --------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------ |
| **PDF Parsing**       | `pymupdf` (PyMuPDFLoader / `fitz`)                                 | C-compiled engine for sub-20ms ultra-fast PDF extraction     |
| **Text Splitting**    | `langchain-text-splitters` RecursiveCharacterTextSplitter          | Sentence-boundary aware, hierarchical                        |
| **Retrieval**         | `langchain-community` BM25Retriever                               | No embeddings, no vector DB, CPU only                        |
| **Reranking**         | `langchain` ContextualCompressionRetriever + CrossEncoderReranker | Semantic scoring on BM25 candidates                          |
| **LLM**               | `langchain-ollama` ChatOllama + Llama 3                           | Local inference, offline, no API key                         |
| **Pipeline**          | `langgraph` StateGraph                                            | Node-based pipeline with typed state and conditional routing |
| **Memory**            | `langchain` ConversationBufferWindowMemory                        | 5-turn sliding window, injected into LLM                     |
| **STT**               | `openai-whisper` base model                                       | CPU-only, near-human accuracy                                |
| **VAD**               | `silero-vad`                                                      | Sub-100ms voice detection, 2MB model                         |
| **TTS**               | `piper-tts` via Sherpa-ONNX                                       | CPU-only, real-time factor 5x                                |
| **Intent**            | BERT fine-tuned classifier                                        | Question / Command / Noise classification                    |
| **API**               | `fastapi`                                                         | REST layer for mobile client                                 |
| **Mobile**            | `flutter` + MediaPipe + Gemma 4                                   | Phase 4                                                      |

---

## Project Structure

```
converseiq/
│
├── notebooks/                        # Development phase — build here first
│   ├── 01_document_processing.ipynb  # PDF load, clean, chunk, inspect
│   ├── 02_bm25_retrieval.ipynb       # BM25 index, query, inspect scores
│   ├── 03_reranking.ipynb            # Cross-encoder reranking, inspect
│   ├── 04_llm_generation.ipynb       # LLM prompt, answer, grounding test
│   ├── 05_memory.ipynb               # Sliding window memory, multi-turn
│   ├── 06_full_pipeline.ipynb        # End-to-end RAG pipeline validation & demo
│
├── core/                             # Converted from notebooks — production modules
│   ├── document.py                   # PDF loading, cleaning, chunking
│   ├── retriever.py                  # BM25Retriever + CrossEncoderReranker
│   ├── llm.py                        # ChatOllama + LCEL chain + prompt
│   ├── memory.py                     # ConversationBufferWindowMemory
│   └── pipeline.py                   # LangGraph StateGraph orchestrator
│
├── threads/                          # Phase 2 — voice concurrency
│   ├── narration.py                  # TTS narration thread
│   └── listener.py                   # VAD + STT + trigger thread
│
├── api/                              # Phase 3 — REST API
│   └── server.py                     # FastAPI endpoints
│
├── tests/
│   ├── test_document.py
│   ├── test_retriever.py
│   └── test_pipeline.py
│
├── models/
│   └── tts/                          # Piper TTS voice model files
│
├── main.py                           # Entry point
├── pyproject.toml
├── uv.lock
├── .env
└── README.md

```

---

## Notebook Development Order

Build and verify each notebook fully before moving to the next. Do not skip ahead. Do not convert to .py until the notebook works end-to-end.

### Notebook 01 — Document Processing

**Goal:** PDF → clean text → hierarchical chunks

```
Cell 1: Install and import dependencies
Cell 2: Load a PDF using pdfplumber
Cell 3: Clean text — strip page numbers, normalize whitespace
Cell 4: Hierarchical chunking:
         - Split into sections (by heading detection)
         - Split sections into paragraphs (~300 words)
         - Split paragraphs into sentences (NLTK)
Cell 5: Print and inspect chunks at each level
Cell 6: Verify — no mid-sentence cuts, overlap working

```

**Verified when:**

- 50-page PDF → 40–80 section-level chunks
- No chunk ends mid-sentence
- Section titles correctly detected
- Empty/scanned PDF handled gracefully

---

### Notebook 02 — BM25 Retrieval

**Goal:** chunks → BM25 index → ranked candidates

```
Cell 1: Import LangChain BM25Retriever
Cell 2: Build retriever from notebook 01 chunks
Cell 3: Query with exact keyword match — inspect scores
Cell 4: Query with paraphrased keywords — inspect scores
Cell 5: Verify top-10 candidates returned correctly
Cell 6: Inspect score distribution

```

**Verified when:**

- Exact keyword query returns correct chunk at rank 1
- Paraphrased query returns lower but non-zero score
- Zero-match query returns empty, no crash

---

### Notebook 03 — Cross-Encoder Reranking

**Goal:** top-10 BM25 candidates → top-3 semantically reranked

```
Cell 1: Import LangChain CrossEncoderReranker
Cell 2: Load ms-marco-MiniLM-L-6-v2 model
Cell 3: Take BM25 top-10 from notebook 02
Cell 4: Rerank and inspect score changes
Cell 5: Test paraphrased query — reranker should fix BM25 misses
Cell 6: Measure latency on CPU

```

**Verified when:**

- Paraphrased query: correct chunk moves to rank 1 after reranking
- Latency under 500ms for 10 candidates on CPU
- Combined retriever (BM25 → reranker) works as single LangChain chain

---

### Notebook 04 — LLM Answer Generation

**Goal:** top-3 chunks + query → grounded answer

```
Cell 1: Connect to Ollama (verify ollama serve is running)
Cell 2: Build ChatOllama + LCEL chain
Cell 3: Build system prompt with strict grounding rules
Cell 4: Test factual question — answer should cite document
Cell 5: Test off-topic question — answer should say not in document
Cell 6: Test temperature and num_predict settings
Cell 7: Test streaming response

```

**Verified when:**

- Factual query → grounded answer in 2–4 sentences
- Off-topic query → "I couldn't find that in the document"
- No hallucinated content in any response
- Streaming works cell by cell

---

### Notebook 05 — Conversation Memory

**Goal:** multi-turn Q&A with sliding window history

```
Cell 1: Initialize ConversationBufferWindowMemory (k=5)
Cell 2: Ask question 1, save to memory
Cell 3: Ask follow-up referencing question 1
Cell 4: Verify LLM uses history correctly
Cell 5: Ask 6 questions — verify oldest turn evicted
Cell 6: Clear memory — verify clean state

```

**Verified when:**

- Follow-up question answered with awareness of prior exchange
- After 6 turns, only last 5 in memory
- Memory clears correctly on new document

---

### Notebook 06 — Full End-to-End Pipeline

**Goal:** PDF upload → query → grounded answer in one flow

```
Cell 1: Import all modules from notebooks 01–05
Cell 2: Build LangGraph StateGraph pipeline
Cell 3: Define nodes: load → retrieve → rerank → generate
Cell 4: Define conditional edges (error routing)
Cell 5: Run full pipeline on test PDF
Cell 6: Run 5 verification questions:
         - Clearly answered by document
         - References previous answer
         - Completely off-topic
         - Uses different words (tests reranker)
         - Asks for summary
Cell 7: Inspect state at each node

```

**Verified when:**

- All 5 verification questions pass
- State flows correctly through graph
- Error node catches failures gracefully

---

### Notebook 07 — Optional Gradio Demo

**Goal:** Optional demo UI after Phase 1 is complete

```
Cell 1: Import pipeline from notebook 06
Cell 2: Build Gradio Blocks interface
Cell 3: PDF upload panel + status indicator
Cell 4: Chat panel with streaming responses
Cell 5: Source display (collapsible)
Cell 6: Launch inline in notebook

```

**Verified when:**

- Upload PDF → chunk count shown
- Ask question → streamed answer appears
- Sources shown below answer
- Clear chat works

---

## Converting Notebooks to Python Modules

Once notebooks 01–06 are verified, convert in this order:

```
notebook 01 → core/document.py
notebook 02 + 03 → core/retriever.py
notebook 04 → core/llm.py
notebook 05 → core/memory.py
notebook 06 → core/pipeline.py

```

**Rule:** One notebook cell = one function in the module. Keep the same logic, same variable names, same verification checks.

---

## Phase Roadmap

```
Phase 1 (Current) — Notebooks → Desktop RAG prototype
  ✅ Document processing
  ✅ Hierarchical BM25 retrieval
  ✅ Cross-encoder reranking
  ✅ LLM grounded answer generation
  ✅ Conversation memory
  ✅ End-to-end notebook validation

Phase 2 — Voice layer
  Voice Activity Detection (Silero VAD)
  Speech-to-Text (Whisper)
  Text-to-Speech (Piper TTS)
  Concurrent narration + listening threads
  Narration resumption from interruption point

Phase 3 — FastAPI REST backend
  POST /upload
  POST /query
  POST /voice-query
  GET  /narrate/{chunk_index}
  POST /session/clear

Phase 4 — Mobile (Flutter + Gemma 4)
  Flutter app (Android + iOS)
  Replace Llama 3 → Gemma 4 4B INT4 via MediaPipe
  Replace Whisper → Whisper.cpp
  Replace Piper → Sherpa-ONNX
  Full offline on-device deployment

```

---

## Setup

```bash
# 1. Install UV
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
# curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Initialize project
uv init converseiq
cd converseiq

# 3. Add dependencies
uv add langchain langchain-core langchain-community langchain-ollama
uv add langgraph
uv add pdfplumber pypdf
uv add nltk spacy
uv add rank-bm25 sentence-transformers
uv add ollama
uv add jupyter
uv add numpy python-dotenv pytest

# Phase 2
uv add openai-whisper torch torchaudio silero-vad piper-tts sounddevice pydub

# Phase 3
uv add fastapi uvicorn

# 4. Start Ollama
ollama pull llama3
ollama serve

# 5. Start notebook
uv run jupyter notebook notebooks/01_document_processing.ipynb
```


---

## Dependency Management (UV)

```text
pyproject.toml   # Project dependencies
uv.lock          # Locked versions (commit this)
.venv/           # Auto-managed by UV
```

UV automatically manages the virtual environment. No manual `venv` activation or `pip install -r requirements.txt` is required.


---

## Environment Variables (.env)

```
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3
PIPER_MODEL=models/tts/en_US-amy-medium.onnx

```

---

## Key Design Decisions

| Decision Choice Reason         |                                                |                                               |
| ------------------------------ | ---------------------------------------------- | --------------------------------------------- |
| No vector DB                   | BM25 + CrossEncoder                            | 60–70% less memory, no GPU, mobile viable     |
| LangGraph over manual pipeline | StateGraph                                     | Typed state, conditional routing, debuggable  |
| LangChain retrievers           | BM25Retriever + ContextualCompressionRetriever | Standardised interface, composable            |
| Notebooks first                | .ipynb → .py                                   | Faster iteration, models stay loaded          |
| Hierarchical chunking          | Section → Paragraph → Sentence                 | Respects document structure, better precision |
| Offline first                  | Ollama + Piper + Whisper                       | Privacy, no API cost, works anywhere          |

---

*ConverseIQ — Major Project Phase-I — 22CBP69* *NITTE Meenakshi Institute of Technology — AY 2025–2026*