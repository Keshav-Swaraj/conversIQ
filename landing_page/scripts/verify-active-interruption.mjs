/** Offline browser test: Whisper wake, one Live session, document turns, resume. */
import assert from "node:assert/strict";
import { chromium } from "@playwright/test";

const browser = await chromium.launch({
  executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE || undefined,
  headless: true,
});
const page = await browser.newPage();
const errors = [];
const transcriptionRequests = [];
let transcript = "Background speech.";
page.on("pageerror", (error) => errors.push(error.message));
await page.addInitScript(() => {
  window.__mockMicLevel = 0;
  window.__mockTranscript = "";
  window.__liveTurns = [];
  window.__trackStops = 0;
  window.__audioSourceStops = 0;
  window.__pauseCount = 0;
  class FakeAudio {
    constructor(src = "") {
      this.src = src; this.paused = true; this.ended = false;
      this.currentTime = 2.5; this.duration = 30; this.playbackRate = 1;
    }
    load() { this.onloadedmetadata?.(); }
    play() { this.paused = false; return Promise.resolve(); }
    pause() { this.paused = true; window.__pauseCount += 1; }
    removeAttribute(name) { if (name === "src") this.src = ""; }
  }
  window.Audio = FakeAudio;
  const track = { stop() { window.__trackStops += 1; }, addEventListener() {} };
  navigator.mediaDevices.getUserMedia = async () => ({
    getTracks: () => [track], getAudioTracks: () => [track],
  });
  window.AudioContext = class {
    constructor() { this.destination = {}; this.currentTime = 0; this.sampleRate = 48_000; }
    createMediaStreamSource() { return { connect() {}, disconnect() {} }; }
    createAnalyser() { return {
      fftSize: 2048, connect() {},
      getFloatTimeDomainData(buffer) { buffer.fill(window.__mockMicLevel); },
    }; }
    createGain() { return { gain: { value: 0 }, connect() {}, disconnect() {} }; }
    createScriptProcessor() {
      const processor = { onaudioprocess: null, connect() {}, disconnect() {} };
      this.timer = setInterval(() => processor.onaudioprocess?.({
        inputBuffer: { getChannelData: () => new Float32Array(2048).fill(window.__mockMicLevel) },
      }), 40);
      return processor;
    }
    createBuffer(_channels, length, rate) { return {
      duration: length / rate, getChannelData: () => new Float32Array(length),
    }; }
    createBufferSource() { return {
      connect() {}, start() { setTimeout(() => this.onended?.({}), 10_000); },
      stop() { window.__audioSourceStops += 1; this.onended?.({}); },
    }; }
    resume() { return Promise.resolve(); }
    close() { clearInterval(this.timer); return Promise.resolve(); }
  };
  window.MediaRecorder = class {
    static isTypeSupported(type) { return type === "audio/webm"; }
    constructor() { this.state = "inactive"; }
    start() { this.state = "recording"; }
    stop() {
      if (this.state === "inactive") return;
      this.state = "inactive";
      setTimeout(() => {
        this.ondataavailable?.({ data: new Blob(["speech"], { type: "audio/webm" }) });
        this.onstop?.();
      }, 0);
    }
  };
  window.WebSocket = class {
    static OPEN = 1;
    constructor() {
      this.readyState = 1;
      setTimeout(() => {
        this.emit({ type: "ready" });
        this.emit({ type: "delta", text: "Hey Keshav." });
        this.emit({ type: "audio", pcm_base64: "AAAAAA==" });
        this.emit({ type: "done" });
      }, 0);
    }
    emit(event) { this.onmessage?.({ data: JSON.stringify(event) }); }
    send(data) {
      if (typeof data !== "string") return;
      const event = JSON.parse(data);
      if (event.type === "speech_start") {
        this.emit({ type: "input_start" });
        return;
      }
      if (event.type !== "audio_end" && event.type !== "text") return;
      const words = event.type === "text" ? event.text : window.__mockTranscript;
      window.__liveTurns.push(words);
      this.emit({ type: "input", text: words, finished: false });
      if (/continue narration/i.test(words)) {
        this.emit({ type: "resume" });
        return;
      }
      this.emit({ type: "sources", relevant_pages: [1] });
      this.emit({ type: "delta", text: "A Portkey transports people to a destination." });
      this.emit({ type: "audio", pcm_base64: "AAAAAA==" });
      this.emit({ type: "done" });
    }
    close() { this.readyState = 3; this.onclose?.(); }
  };
});
await page.route("**/api/**", async (route) => {
  const path = new URL(route.request().url()).pathname;
  let body = {};
  if (path === "/api/upload") body = {
    doc_id: "testdoc", filename: "sample.pdf", num_pages: 1,
    pages: [{ page_number: 1, text: "A Portkey transports people to a destination." }],
    sections: [{ id: "1", title: "Sample", page: 1, end_page: 1 }],
  };
  if (path === "/api/narrate") body = {
    complete: false, page: 1, offset: 0, text: "A Portkey transports people to a destination.",
    next_page: 2, next_offset: 0, next_complete: true, audio_url: "/api/audio?token=narration",
  };
  if (path === "/api/transcribe") {
    transcriptionRequests.push(new URL(route.request().url()).searchParams.get("wake"));
    body = { transcript };
  }
  if (path === "/api/live-voice-config") body = {
    url: "ws://127.0.0.1:8767/api/live-voice?doc_id=testdoc",
  };
  await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
});

async function speak(words, level = 0.12, milliseconds = 350) {
  transcript = words;
  await page.evaluate(({ words, level }) => {
    window.__mockTranscript = words;
    window.__mockMicLevel = level;
  }, { words, level });
  await page.waitForTimeout(milliseconds);
  await page.evaluate(() => { window.__mockMicLevel = 0; });
  await page.waitForTimeout(1_850);
}

try {
  await page.goto(process.env.CONVERSIQ_TEST_URL || "http://127.0.0.1:8765/workspace");
  await page.locator('input[type="file"]').setInputFiles({
    name: "sample.pdf", mimeType: "application/pdf", buffer: Buffer.from("fixture"),
  });
  await page.getByText("sample.pdf").first().waitFor();
  await page.getByRole("button", { name: "Play narration" }).click();
  await page.getByRole("button", { name: "Turn off Sarah wake-word listening" }).waitFor();
  await speak("Background speech.");
  assert.equal(transcriptionRequests.at(-1), "1");
  await page.getByRole("button", { name: "Pause narration" }).waitFor();
  await speak("Hey, Sarah.", 0.013, 180);
  await page.locator(".message.ai").getByText("Hey Keshav.").waitFor();
  assert.equal(transcriptionRequests.length, 2);
  await speak("What is a Portkey?");
  await page.locator(".message.user").getByText("What is a Portkey?").waitFor();
  await page.locator(".message.ai").getByText("A Portkey transports people to a destination.").waitFor();
  assert.equal(transcriptionRequests.length, 2); // Follow-ups bypass Whisper.
  await speak("What does it do?");
  await page.locator(".message.user").getByText("What does it do?").waitFor();
  assert.deepEqual(await page.locator(".message.user p").allTextContents(), [
    "What is a Portkey?", "What does it do?",
  ]);
  assert.ok(await page.evaluate(() => window.__audioSourceStops >= 2)); // Greeting and answer were interrupted.
  assert.deepEqual(await page.evaluate(() => window.__liveTurns.slice(0, 2)), [
    "What is a Portkey?", "What does it do?",
  ]);
  await speak("Continue narration");
  await page.getByRole("button", { name: "Pause narration" }).waitFor();
  await speak("Sarah");
  await page.waitForFunction(() => [...document.querySelectorAll(".message.ai")]
    .filter((message) => message.textContent?.includes("Hey Keshav.")).length === 2);
  await page.getByRole("button", { name: "Continue narration", exact: true }).click();
  await page.getByRole("button", { name: "Turn off Sarah wake-word listening" }).click();
  assert.ok(await page.evaluate(() => window.__trackStops >= 1));
  assert.deepEqual(errors, []);
  console.log("Whisper wake, persistent Live turns, grounded sources, resume, repeat wake, and mic cleanup passed.");
} finally {
  await browser.close();
}
