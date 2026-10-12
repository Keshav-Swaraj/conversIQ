# ConversIQ frontend

The landing page at `/` is a visual demo. `/workspace` is the working local PDF reader, with original PDF pages, narration, Sarah wake-word listening, and a grounded Gemini Live voice conversation. See the [project README](../README.md) for setup, required keys, supported narration providers, and usage.

From this directory, `npm ci` installs the locked frontend dependencies and `npm run build` creates `dist/` for the Python server. For frontend development, run `npm run dev` alongside `uv run python web_server.py` from the project root; Vite proxies HTTP `/api` requests to the local server.

`npm run test:interrupt` checks the speech gate. With the backend running and Playwright Chromium available, `npm run test:interrupt-browser` checks the mocked wake/interruption/conversation/resume UI flow without cloud calls. `scripts/verify.mjs` checks the landing-page demo.
