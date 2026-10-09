# ConversIQ frontend

The landing page and its embedded product demo retain the original visual concept. The separate `/workspace` route is the local working reader: it uploads a real PDF, shows extracted pages and PageIndex sections, narrates with Piper, and sends questions through the existing grounded-answer workflow. Its microphone button records until clicked again, then uses Groq Whisper for transcription.

## Run the local workspace

From the project root:

```sh
cd landing_page
npm install
npm run build
cd ..
uv run python web_server.py
```

Open `http://127.0.0.1:8765/` for the landing page or `http://127.0.0.1:8765/workspace` for the real reader. This server binds to the local computer only. A project-root `.env` with `GROQ_API_KEY` is required by the existing configuration layer; the Piper voice model must be at `cache/voices/en_US-lessac-medium.onnx` with its `.onnx.json` file.

For frontend development, run `npm run dev` in `landing_page` alongside `uv run python web_server.py`. Vite proxies `/api` to the local Python server.

The embedded landing-page demo remains simulated and uses original summary passages, not book text. The real reader uses extracted PDF text. Narration reuses one Piper voice, prepares the next audio segment while the current one plays, and supports seeking within the current WAV. Precise word-level highlighting, PDF-page visual overlays, and immediate seeks to uncached audio are not implemented yet. `scripts/verify.mjs` checks the landing-page demo and needs a local Playwright Chromium executable.
