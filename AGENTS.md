# ConversIQ — Codex Development Guidelines

## 1. Project Overview

ConversIQ is a voice-first document RAG system.

The long-term goal is:

```text
PDF
  ↓
Document understanding / indexing
  ↓
Narration
  ↓
User asks a spoken question
  ↓
Retrieve relevant document content
  ↓
Generate a grounded answer
  ↓
Resume narration from where it stopped
```

The project is intentionally built in layers.

The current implementation should prioritize correctness, simplicity, testability, and understanding over premature abstraction.

---

# 2. Current Development Strategy

Build ConversIQ one layer at a time.

Do NOT implement the entire application in a single task.

Each layer must:

1. Have a clearly defined responsibility.
2. Have a small public API.
3. Be tested independently.
4. Have a corresponding notebook checkpoint.
5. Work before the next layer is started.

Development order:

```text
Layer 0 — Project foundation
        ↓
Layer 1 — Configuration
        ↓
Layer 2 — PDF extraction
        ↓
Layer 3 — PageIndex tree/indexing
        ↓
Layer 4 — Retrieval + grounded answer
        ↓
Layer 5 — LangGraph orchestration
        ↓
Layer 6 — Speech-to-text
        ↓
Layer 7 — Text-to-speech
        ↓
Layer 8 — Narration state
        ↓
Layer 9 — Interrupt/resume workflow
        ↓
Layer 10 — Full application integration
```

Do not skip ahead unless explicitly instructed.

---

# 3. Project Structure

The target structure is:

```text
conversiq/
├── notebooks/
│   ├── 01_pdf_extraction.ipynb
│   ├── 02_pageindex_tree.ipynb
│   ├── 03_retrieval_query.ipynb
│   ├── 04_stt_groq_whisper.ipynb
│   ├── 05_tts_cartesia.ipynb
│   └── 06_interrupt_resume_logic.ipynb
│
├── src/
│   ├── __init__.py
│   ├── extraction.py
│   ├── indexer.py
│   ├── retriever.py
│   ├── stt.py
│   ├── tts.py
│   ├── narrator.py
│   ├── graph.py
│   └── config.py
│
├── cache/
│   └── trees/
│
├── data/
│   └── sample_docs/
│
├── main.py
├── AGENTS.md
├── requirements.txt
├── .env
├── .env.example
└── .gitignore
```

Do not introduce additional directories or frameworks unless there is a clear architectural reason.

---

# 4. Technology Responsibilities

Use each technology for the responsibility it is best suited for.

## Python

Use normal Python for deterministic infrastructure:

* filesystem operations
* PDF validation
* PDF parsing
* hashing
* cache management
* path management
* audio file management
* playback position tracking
* state persistence
* utility functions

Do not wrap ordinary deterministic code in LangChain just to use LangChain.

---

## PyMuPDF

Use PyMuPDF for PDF processing.

Responsibilities:

```text
PDF → pages → text
```

Do not use an LLM for basic PDF extraction.

---

## PageIndex

PageIndex is the document indexing/retrieval mechanism.

Responsibilities:

* building the hierarchical document tree
* reasoning over document structure
* locating relevant sections/pages
* supporting document-grounded retrieval

PageIndex should remain the core document retrieval mechanism.

---

## LangChain

Use LangChain for LLM-facing functionality.

Responsibilities may include:

* Groq chat model integration
* prompt templates
* chat messages
* structured output
* model invocation
* LLM tools where genuinely useful
* reusable LLM chains

Do not use LangChain merely as a wrapper around deterministic Python functions.

---

## LangGraph

Use LangGraph for application orchestration.

Responsibilities:

* shared application state
* workflow orchestration
* sequential execution
* conditional branching
* loops
* interruptions
* resumable workflows
* future narration/question/answer state transitions

LangGraph should represent application behavior, not replace every normal Python function.

---

## Groq

Groq is the LLM/STT provider.

LLM model:

```text
openai/gpt-oss-120b
```

All credentials must come from `.env`.

Never hardcode API keys.

---

## Whisper

Use Groq Whisper for speech-to-text.

Speech-to-text responsibility:

```text
audio → transcript
```

Keep this integration isolated in:

```text
src/stt.py
```

---

## Cartesia

Use Cartesia for text-to-speech. Load credentials from `.env` and make no
Cartesia API calls at import time.

Text-to-speech responsibility:

```text
text → audio
```

Keep this integration isolated in:

```text
src/tts.py
```

---

# 5. RAG Architecture

ConversIQ is NOT a traditional embedding/vector-search RAG system.

Do NOT introduce:

* embeddings
* FAISS
* Chroma
* Pinecone
* Milvus
* Weaviate
* Qdrant
* other vector databases

Unless explicitly instructed otherwise, the retrieval architecture is:

```text
PDF
 ↓
PyMuPDF / document extraction
 ↓
PageIndex hierarchical tree
 ↓
PageIndex reasoning/retrieval
 ↓
relevant pages / sections
 ↓
source text
 ↓
LangChain ChatGroq
 ↓
grounded answer
```

The answer-generation model must be grounded in retrieved document content.

Do not invent facts that are not supported by the retrieved source.

---

# 6. LangGraph Architecture

The initial LangGraph should remain intentionally small.

Phase 1 graph:

```text
START
  ↓
load_tree
  ↓
retrieve
  ↓
generate_answer
  ↓
END
```

Expected state should contain at least:

```python
doc_id
file_path
query
tree
relevant_pages
source_text
answer
error
```

LangGraph owns the workflow/state.

Individual modules continue to own their domain logic.

For example:

```text
graph.py
    ↓
indexer.py
    ↓
PageIndex
```

and:

```text
graph.py
    ↓
retriever.py
    ↓
PageIndex + LangChain
```

Do not put the complete application implementation directly into one giant graph file.

---

# 7. Layer Responsibilities

## Layer 1 — Configuration

`src/config.py`

Responsibilities:

* load `.env`
* expose `GROQ_API_KEY`
* expose model names
* expose project paths
* fail clearly when required credentials are missing

Do not make API calls when importing configuration.

---

## Layer 2 — Extraction

`src/extraction.py`

Responsibilities:

```text
PDF → structured pages
```

Expected information should include:

* page number
* extracted text
* document-level metadata where useful

Extraction must be deterministic and independently testable.

Do not call Groq, LangChain, or LangGraph from the extraction layer.

---

## Layer 3 — Indexing

`src/indexer.py`

Responsibilities:

```text
document → PageIndex tree
```

Implement caching.

Expected cache location:

```text
cache/trees/{doc_id}.json
```

Behavior:

```text
cache exists?
    │
    ├── yes → load cache
    │
    └── no → build PageIndex tree → save cache
```

A cache hit must not unnecessarily call Groq/PageIndex again.

Provide a cache-only loading function.

---

## Layer 4 — Retrieval

`src/retriever.py`

Responsibilities:

```text
query + tree
    ↓
PageIndex reasoning
    ↓
relevant pages
    ↓
source text
    ↓
grounded answer
```

Keep retrieval and answer generation logically separable inside the module.

Return structured results rather than unstructured print statements.

At minimum:

```text
answer
relevant_pages
source_text
```

---

## Layer 5 — LangGraph

`src/graph.py`

Responsibilities:

* define graph state
* define nodes
* connect nodes
* compile graph
* expose graph invocation API

Do not duplicate retrieval or indexing logic already implemented in `indexer.py` and `retriever.py`.

---

## Layer 6 — STT

`src/stt.py`

Responsibilities:

```text
audio → text
```

Keep provider-specific API logic isolated.

---

## Layer 7 — TTS

`src/tts.py`

Responsibilities:

```text
text → audio
```

Keep Cartesia-specific logic isolated.

---

## Layer 8 — Narrator

`src/narrator.py`

Responsibilities include:

* narration state
* current page
* current text position
* pause state
* resume position
* interaction with TTS

Do not mix retrieval logic into narrator.py.

---

## Layer 9 — Interrupt/Resume

This layer combines:

```text
narration
    ↓
listen
    ↓
question?
    ↓
retrieve
    ↓
answer
    ↓
restore narration state
    ↓
resume
```

This is where LangGraph should become substantially more useful.

Use explicit state and conditional transitions rather than scattered global variables.

---

# 8. Notebook Strategy

Every notebook is a checkpoint.

## 01_pdf_extraction.ipynb

Must prove:

```text
PDF → extracted pages/text
```

Test:

* valid PDF
* invalid PDF
* page numbering
* text extraction
* empty/malformed pages where relevant

---

## 02_pageindex_tree.ipynb

Must prove:

```text
PDF → PageIndex tree
```

Test:

* tree creation
* tree structure
* cache creation
* cache hit
* optional forced rebuild

Make cache behavior visible.

---

## 03_retrieval_query.ipynb

Must prove:

```text
query → relevant pages → grounded answer
```

Test:

* answerable question
* relevant page selection
* source text
* unanswerable question
* no confident hallucination

This notebook should eventually demonstrate the LangGraph workflow.

---

## 04_stt_groq_whisper.ipynb

Must prove:

```text
audio → transcript
```

---

## 05_tts_cartesia.ipynb

Must prove:

```text
text → audio
```

---

## 06_interrupt_resume_logic.ipynb

Must prove:

```text
narration
 → interruption
 → question
 → retrieval
 → answer
 → resume from previous position
```

This notebook is the main behavioral test for the voice interaction layer.

---

# 9. Development Rules

## Rule 1 — One layer at a time

When asked to implement a layer:

* implement only that layer
* test only that layer
* update only the relevant notebook
* do not prematurely implement later layers

---

## Rule 2 — Inspect before modifying

Before changing an existing file:

1. read the file
2. understand its current API
3. identify dependencies
4. preserve working behavior
5. make the smallest necessary change

Do not rewrite functioning code without a concrete reason.

---

## Rule 3 — Reuse existing functionality

Do not duplicate:

* configuration logic
* Groq client creation
* PageIndex configuration
* cache logic
* PDF parsing
* retrieval logic

Reuse existing modules.

---

## Rule 4 — Keep modules small

Avoid giant files.

Prefer:

```text
small module
    ↓
small functions
    ↓
clear interfaces
```

over:

```text
one giant orchestration function
```

---

## Rule 5 — No hidden state

Avoid:

* unexpected global mutable state
* hidden caches
* implicit singleton objects
* module-level API calls

State should be explicit.

---

## Rule 6 — Type everything

Public functions must have type hints.

Prefer typed result structures for non-trivial return values.

Use clear docstrings for public functions.

---

## Rule 7 — Errors should be explicit

Raise meaningful exceptions or return explicit error state.

Do not silently swallow failures.

Do not use:

```python
except Exception:
    pass
```

unless there is a very strong reason and the behavior is documented.

---

## Rule 8 — No debugging prints in library code

Do not leave:

```python
print(...)
```

inside production modules for debugging.

Use logging where appropriate.

Notebook output is allowed for demonstrations/tests.

---

# 10. Dependency Rules

Keep `requirements.txt` minimal.

Before adding a dependency:

1. Determine whether the standard library or an existing dependency can solve the problem.
2. Check whether the package is actually required.
3. Avoid adding multiple libraries that solve the same problem.
4. Add the dependency only when justified.

Do not introduce frameworks simply because they are popular.

---

# 11. Environment and Secrets

Never hardcode:

```text
GROQ_API_KEY
```

or any other secret.

Use:

```text
.env
```

and keep `.env` in `.gitignore`.

Maintain:

```text
.env.example
```

with placeholder values only.

Never print API keys.

---

# 12. Testing Philosophy

Every layer must have a working test before moving forward.

Prefer testing real behavior over checking that functions merely execute.

Example:

Bad:

```python
assert retrieve(...) is not None
```

Better:

```text
query
 → expected relevant page
 → grounded answer
```

Tests should catch regressions.

---

# 13. Notebook Rules

Notebooks are for:

* experimentation
* demonstrations
* integration tests
* visual inspection
* learning/debugging

Core reusable logic belongs in `src/`.

Do not make notebooks the only place where important application logic exists.

Do not duplicate major implementation logic inside notebooks.

Notebook cells should import and use functions from `src/`.

---

# 14. Main Application

`main.py` should remain thin.

It should orchestrate existing modules rather than contain business logic.

Conceptually:

```text
main.py
    ↓
src modules
    ↓
actual implementation
```

Do not turn `main.py` into a giant script.

---

# 15. Codex Task Discipline

When implementing a requested task:

### First

Inspect:

* current files
* AGENTS.md
* relevant notebook
* relevant dependencies

### Then

State internally what the smallest required change is.

### Then

Implement only that change.

### Then

Run appropriate tests/checks.

### Finally

Report:

```text
Files changed:
...

Tests/checks run:
...

Notebook updated:
...

Result:
...

Known issues:
...
```

Do not modify unrelated files.

---

# 16. Do Not Over-Engineer

ConversIQ is being developed incrementally.

Avoid premature:

* agent systems
* multi-agent architectures
* autonomous tool loops
* complex memory systems
* event buses
* distributed systems
* unnecessary abstractions
* complicated class hierarchies

Start simple.

Introduce complexity only when the actual project requirement demands it.

---

# 17. Important Architectural Principle

Use the simplest technology that correctly solves the problem.

The desired division is:

```text
PyMuPDF
    ↓
document extraction

PageIndex
    ↓
document structure + retrieval

LangChain
    ↓
LLM interaction

LangGraph
    ↓
workflow + state + branching

Groq
    ↓
LLM + STT

Cartesia
    ↓
TTS

Normal Python
    ↓
files + caching + deterministic infrastructure
```

Do not blur these responsibilities without a concrete reason.

---

# 18. Phase 1 Definition of Done

Phase 1 is complete when this works reliably:

```text
PDF
 ↓
PageIndex tree
 ↓
cached tree
 ↓
user text query
 ↓
PageIndex retrieval
 ↓
relevant source pages/text
 ↓
LangChain ChatGroq
 ↓
grounded answer
```

The system must return:

```text
answer
relevant_pages
source_text
```

No voice functionality is required for Phase 1.

---

# 19. Future Voice Architecture

The eventual system should support:

```text
Document narration
       ↓
user interruption
       ↓
speech-to-text
       ↓
LangGraph state transition
       ↓
PageIndex retrieval
       ↓
LangChain answer
       ↓
text-to-speech
       ↓
restore narration state
       ↓
resume narration
```

The architecture should be designed so that these future capabilities can be added without rewriting the Phase 1 retrieval system.

---

# 20. Golden Rule

Do not optimize for the amount of code produced.

Optimize for:

```text
correctness
+
clarity
+
testability
+
small interfaces
+
understandable architecture
```

A small working layer is better than a large incomplete implementation.

When uncertain about architecture, prefer the simplest design that preserves the project's core requirements.

Never implement future functionality merely because it might eventually be useful.
