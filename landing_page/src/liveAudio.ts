/** Play Gemini Live's ordered, raw 24 kHz mono PCM chunks without a TTS round trip. */
export class LiveAudioPlayer {
  private context: AudioContext;
  private sources = new Set<AudioBufferSourceNode>();
  private endTime = 0;
  private lastSource: AudioBufferSourceNode | null = null;
  private stopped = false;
  private finishResolve: (() => void) | null = null;

  constructor() {
    this.context = new AudioContext();
    void this.context.resume();
  }

  append(pcmBase64: string): void {
    if (this.stopped) return;
    const binary = atob(pcmBase64);
    if (binary.length % 2) throw new Error("Gemini Live sent malformed audio.");
    const buffer = this.context.createBuffer(1, binary.length / 2, 24_000);
    const samples = buffer.getChannelData(0);
    for (let index = 0; index < samples.length; index += 1) {
      const low = binary.charCodeAt(index * 2);
      const high = binary.charCodeAt(index * 2 + 1);
      const sample = (high << 8) | low;
      samples[index] = (sample >= 0x8000 ? sample - 0x10000 : sample) / 0x8000;
    }
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.context.destination);
    this.sources.add(source);
    source.onended = () => this.sources.delete(source);
    const start = Math.max(this.context.currentTime + 0.03, this.endTime);
    source.start(start);
    this.endTime = start + buffer.duration;
    this.lastSource = source;
  }

  async finish(): Promise<void> {
    if (!this.lastSource || this.stopped) return;
    const last = this.lastSource;
    await new Promise<void>((resolve) => {
      this.finishResolve = resolve;
      const previous = last.onended;
      last.onended = (event) => { previous?.call(last, event); this.finishResolve = null; resolve(); };
      if (this.context.currentTime >= this.endTime) { this.finishResolve = null; resolve(); }
    });
  }

  stop(): void {
    if (this.stopped) return;
    this.stopped = true;
    this.finishResolve?.();
    this.finishResolve = null;
    for (const source of this.sources) {
      try { source.stop(); } catch { /* Source may already have ended. */ }
    }
    this.sources.clear();
    void this.context.close();
  }
}
