import { SpeechGate } from "./activeInterruption";
import { LiveAudioPlayer } from "./liveAudio";

/** Events from the local, server-owned Gemini Live session. */
export type VoiceEvent =
  | { type: "ready" | "done" | "playback_done" | "interrupted" | "resume" | "not_grounded" | "input_start" }
  | { type: "input"; text: string; finished: boolean }
  | { type: "delta"; text: string }
  | { type: "sources" | "cited_pages"; relevant_pages: number[] }
  | { type: "audio"; pcm_base64: string }
  | { type: "error"; message: string };

function pcm16(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const length = Math.max(1, Math.floor(samples.length * 16_000 / sampleRate));
  const output = new ArrayBuffer(length * 2);
  const view = new DataView(output);
  for (let index = 0; index < length; index += 1) {
    const source = index * sampleRate / 16_000;
    const left = Math.min(samples.length - 1, Math.floor(source));
    const fraction = source - left;
    const value = samples[left] * (1 - fraction)
      + samples[Math.min(samples.length - 1, left + 1)] * fraction;
    view.setInt16(index * 2, Math.round(Math.max(-1, Math.min(1, value)) * 32767), true);
  }
  return output;
}

export class LiveVoiceSession {
  private socket: WebSocket | null = null;
  private mic: MediaStream | null = null;
  private context: AudioContext | null = null;
  private processor: ScriptProcessorNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private silentOutput: GainNode | null = null;
  private player: LiveAudioPlayer | null = null;
  private readonly gate = new SpeechGate();
  private stopped = false;

  constructor(private readonly onEvent: (event: VoiceEvent) => void) {}

  async start(docId: string): Promise<void> {
    const response = await fetch(`/api/live-voice-config?doc_id=${encodeURIComponent(docId)}`);
    const setup = await response.json() as { url?: string; error?: string };
    if (!response.ok || !setup.url) throw new Error(setup.error || "Live voice is unavailable.");
    await new Promise<void>((resolve, reject) => {
      const socket = new WebSocket(setup.url!);
      this.socket = socket;
      const timeout = window.setTimeout(() => reject(new Error("Sarah could not connect.")), 15_000);
      let ready = false;
      socket.onmessage = (message) => {
        const event = JSON.parse(message.data as string) as VoiceEvent;
        if (event.type === "ready") {
          ready = true;
          window.clearTimeout(timeout);
          resolve();
        } else if (event.type === "error" && !ready) {
          window.clearTimeout(timeout);
          reject(new Error(event.message));
        } else if (ready) {
          if (event.type === "audio") {
            if (!this.player) this.player = new LiveAudioPlayer();
            this.player.append(event.pcm_base64);
          } else if (event.type === "interrupted") {
            this.stopPlayback();
          } else if (event.type === "done") {
            const player = this.player;
            if (player) void player.finish().then(() => {
              if (!this.stopped && this.player === player) this.onEvent({ type: "playback_done" });
            });
          }
          this.onEvent(event);
        }
      };
      socket.onerror = () => {
        if (!ready) { window.clearTimeout(timeout); reject(new Error("Sarah's voice connection failed.")); }
      };
      socket.onclose = () => {
        if (!ready) { window.clearTimeout(timeout); reject(new Error("Sarah's voice connection closed.")); }
        else if (!this.stopped) this.onEvent({ type: "error", message: "Sarah's voice session disconnected. Resume narration or try again." });
      };
    });
    if (this.stopped) return;
    await this.startMicrophone();
  }

  sendText(text: string): void {
    if (this.socket?.readyState !== WebSocket.OPEN) throw new Error("Sarah is not connected.");
    this.socket.send(JSON.stringify({ type: "text", text }));
  }

  stopPlayback(): void {
    this.player?.stop();
    this.player = null;
  }

  close(): void {
    if (this.stopped) return;
    this.stopped = true;
    this.stopPlayback();
    if (this.processor) this.processor.onaudioprocess = null;
    this.source?.disconnect();
    this.processor?.disconnect();
    this.silentOutput?.disconnect();
    this.mic?.getTracks().forEach((track) => track.stop());
    void this.context?.close();
    if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(JSON.stringify({ type: "close" }));
    this.socket?.close();
  }

  private async startMicrophone(): Promise<void> {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("Microphone access is unavailable.");
    const mic = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    if (this.stopped) { mic.getTracks().forEach((track) => track.stop()); return; }
    this.mic = mic;
    const context = new AudioContext();
    this.context = context;
    if (!context.createScriptProcessor) throw new Error("This browser cannot stream microphone PCM.");
    const source = context.createMediaStreamSource(mic);
    const processor = context.createScriptProcessor(2048, 1, 1);
    const silentOutput = context.createGain();
    silentOutput.gain.value = 0;
    this.source = source;
    this.processor = processor;
    this.silentOutput = silentOutput;
    processor.onaudioprocess = (event) => {
      if (this.stopped || this.socket?.readyState !== WebSocket.OPEN) return;
      const samples = event.inputBuffer.getChannelData(0);
      let energy = 0;
      for (const sample of samples) energy += sample * sample;
      const action = this.gate.update(Math.sqrt(energy / samples.length), performance.now(), true);
      if (action === "start") this.socket.send(JSON.stringify({ type: "speech_start" }));
      if (action === "interrupt" && this.player) {
        this.stopPlayback();
        this.onEvent({ type: "interrupted" });
      }
      this.socket.send(pcm16(samples, context.sampleRate));
      if (action === "finish") this.socket.send(JSON.stringify({ type: "audio_end" }));
    };
    source.connect(processor);
    processor.connect(silentOutput);
    silentOutput.connect(context.destination);
    await context.resume();
  }
}
