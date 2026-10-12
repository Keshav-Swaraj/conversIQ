# ConversIQ

ConversIQ is a local, voice-first PDF reader. Upload a document, follow its original pages while it is narrated, say **Sarah** to start a conversation, ask questions grounded in the document, and say **continue narration** to resume where you stopped.

The landing page is a visual demo; `/workspace` is the working application. The app is still a local prototype, not a hosted multi-user service.

## How it works

- **Reading and navigation:** PyMuPDF extracts page text and renders the original PDF pages. The reader highlights the current sentence and word and lets you select where narration starts.
- **Document structure and retrieval:** PageIndex Flash builds an LLM-free hierarchical tree (`summary=False`, `optimize=False`) and caches it under `cache/trees/`. A Groq-powered selector uses that tree to find relevant document sections; source text comes from the PDF, not a vector database.
- **Narration:** `TTS_PROVIDER` selects the voice backend. Cartesia is the default; Piper, Qwen, Kokoro, Gemini TTS, and Groq TTS have optional adapters. Generated audio and uploaded PDFs stay in ignored local cache directories.
- **Conversation:** Groq Whisper detects the Sarah wake word during narration. The browser then opens one Gemini Live voice session for follow-up questions and spoken answers. Document questions are grounded with PageIndex-selected page excerpts. Saying “continue narration” returns to the saved reading position.
- **Other entry points:** The Python CLI, notebooks, and LangGraph workflow remain available for layered development and testing.

PageIndex can select a whole chapter, so a section range such as “pages 7–22” is not itself a precise citation. The live answer is asked to name the specific supporting pages; the UI shows those cited pages when they are present in the answer transcript. Always check the source pages for important claims.

## Requirements

- Python 3.11 and [uv](https://docs.astral.sh/uv/)
- Node.js and npm for the React/Vite frontend
- A modern browser with microphone permission (localhost is supported)
- API access for Groq and Gemini Live; Cartesia access if using the default narration voice

Cloud speech and model calls may incur charges or hit provider rate limits. The server binds to `127.0.0.1`; it is not configured for public deployment.

## Local setup

From the repository root in PowerShell:

```powershell
uv sync
Copy-Item .env.example .env   # first setup only; do not overwrite an existing .env
```

Edit `.env` and set `GROQ_API_KEY` and `GEMINI_API_KEY`. For default narration, also set `CARTESIA_API_KEY` and keep `TTS_PROVIDER=cartesia`. `src.config` requires the Groq key even when only using a different feature. Never commit `.env` or model files.

Build the frontend and start the local server:

```powershell
cd landing_page
npm ci
npm run build
cd ..
uv run python web_server.py
```

Open [http://127.0.0.1:8765/workspace](http://127.0.0.1:8765/workspace) for the reader or [http://127.0.0.1:8765/](http://127.0.0.1:8765/) for the landing page. The voice WebSocket uses the next local port (`8766` by default). If you change the HTTP port with `--port`, allow both local ports.

## Use the workspace

1. Upload a PDF. The first upload builds its PageIndex tree; later uploads can reuse the cached tree.
2. Press Play or click a sentence on the PDF page to begin narration there.
3. Allow microphone access and keep Sarah wake-word listening enabled. Say “Sarah” during narration; narration pauses and the live conversation begins. Headphones help prevent speaker audio from reaching the mic.
4. Ask about the document. Sarah should speak a grounded answer and show its source pages. You can interrupt her with another question.
5. Say “continue narration” or use the on-screen control to return to your saved place.

The manual question control and typed question box are available if you do not want hands-free input. A PDF must contain extractable text for text-based narration and answers; scanned image-only PDFs need OCR, which is not part of this pipeline.

## TTS choices

Set `TTS_PROVIDER` in `.env` and restart the backend. The optional providers need their own credentials or locally installed model files:

| Value | Narration backend | Setup |
| --- | --- | --- |
| `cartesia` | Cloud, default | `CARTESIA_API_KEY` |
| `piper` | Local | Install `piper-tts`; set `PIPER_MODEL_PATH` to an existing voice `.onnx` |
| `qwen` | Local | `uv sync --extra qwen`; set `QWEN_MODEL_PATH` to an existing CustomVoice model directory |
| `kokoro` | Local | `uv sync --extra kokoro`; set `KOKORO_MODEL_PATH` to an existing ONNX model with its voices file |
| `gemini` | Cloud | `GEMINI_API_KEY` and Gemini TTS access |
| `groq` | Cloud | `GROQ_API_KEY` and Groq TTS access |

These settings change narration, **not** Sarah’s Gemini Live conversation voice. Local model weights, generated WAVs, and uploaded documents are not stored in Git. For manual provider comparisons, run `uv run python benchmark_tts.py --list` and then select providers with `--providers`.

## Development checks

```powershell
uv run pytest tests -q
cd landing_page
npm run build
npm run test:interrupt
```

With the backend running and a Playwright browser available, `npm run test:interrupt-browser` exercises the mocked wake/interruption/conversation/resume flow without cloud calls. Real microphone behavior and cloud-model responses still need manual testing. The `notebooks/` directory contains layer-by-layer checkpoints; select the project `.venv` kernel when running them.

`pyproject.toml` and `uv.lock` are the Python dependency source of truth. `requirements.txt` is an older minimal bootstrap list and does not represent the full app. Keep `data/`, `experiments/`, `research_paper/`, generated cache files, `.env`, and local voice models out of commits.
