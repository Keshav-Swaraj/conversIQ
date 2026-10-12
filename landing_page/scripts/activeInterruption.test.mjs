import assert from "node:assert/strict";
import test from "node:test";
import { SpeechGate } from "../src/activeInterruption.ts";
import { wakeQuestion, wantsNarration } from "../src/sarahConversation.ts";

test("a brief sound is discarded without interrupting narration", () => {
  const gate = new SpeechGate();
  assert.equal(gate.update(0.01, 0, true), null);
  assert.equal(gate.update(0.08, 100, true), "start");
  assert.equal(gate.update(0.01, 300, true), "discard");
  assert.equal(gate.update(0.01, 350, true), null);
});

test("sustained speech interrupts, then silence finishes one question", () => {
  const gate = new SpeechGate();
  assert.equal(gate.update(0.09, 100, true), "start");
  assert.equal(gate.update(0.10, 280, true), "interrupt");
  assert.equal(gate.update(0.08, 800, false), null);
  assert.equal(gate.update(0.01, 2100, false), "finish");
  assert.equal(gate.update(0.01, 2200, false), null);
});

test("monitoring while narration is paused cannot start a recording", () => {
  const gate = new SpeechGate();
  assert.equal(gate.update(0.12, 100, false), null);
  assert.equal(gate.update(0.12, 200, true), "start");
  assert.equal(gate.update(0.12, 240, false), "discard");
});

test("a long question ends at the recording cap", () => {
  const gate = new SpeechGate();
  assert.equal(gate.update(0.1, 0, true), "start");
  assert.equal(gate.update(0.1, 200, true), "interrupt");
  assert.equal(gate.update(0.1, 15_250, false), "finish");
});

test("quiet laptop microphone speech still triggers an interruption", () => {
  const gate = new SpeechGate();
  assert.equal(gate.update(0.018, 100, true), "start");
  assert.equal(gate.update(0.018, 300, true), "interrupt");
  assert.equal(gate.update(0.001, 1800, false), "finish");
});

test("a short quiet wake word is recorded for transcription", () => {
  const gate = new SpeechGate();
  assert.equal(gate.update(0.013, 100, true), "start");
  assert.equal(gate.update(0.013, 190, true), "interrupt");
  assert.equal(gate.update(0.001, 1_500, true), "finish");
});

test("Sarah only wakes at the start of a transcript", () => {
  assert.equal(wakeQuestion("Sarah"), "");
  assert.equal(wakeQuestion("Hey Sarah, what happens here?"), "what happens here?");
  assert.equal(wakeQuestion("Hey, Sarah."), "");
  assert.equal(wakeQuestion("Sara."), "");
  assert.equal(wakeQuestion("Sera?"), "");
  assert.equal(wakeQuestion("Saira?"), "");
  assert.equal(wakeQuestion("Hey, Sayra, explain that"), "explain that");
  assert.equal(wakeQuestion("I heard Sarah in the book"), null);
  assert.equal(wakeQuestion("What happens at the Riddle House?"), null);
});

test("continue narration is a command, not a document question", () => {
  assert.equal(wantsNarration("Yes please."), true);
  assert.equal(wantsNarration("Continue reading"), true);
  assert.equal(wantsNarration("Continue narration"), true);
  assert.equal(wantsNarration("What happened next?"), false);
});
