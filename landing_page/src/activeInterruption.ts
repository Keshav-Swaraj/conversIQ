/** Opt-in browser speech onset detection and question capture. */

export type GateAction = "start" | "interrupt" | "discard" | "finish" | null;

export class SpeechGate {
  private noiseFloor = 0.004;
  private candidateAt: number | null = null;
  private confirmedAt: number | null = null;
  private lastVoiceAt = 0;
  private cooldownUntil = 0;

  /** Decide what to do with one microphone level; time is in milliseconds. */
  update(level: number, now: number, eligible: boolean): GateAction {
    if (!eligible && this.confirmedAt === null) {
      if (this.candidateAt !== null) {
        this.candidateAt = null;
        return "discard";
      }
      return null;
    }

    // Laptop microphones with browser noise suppression can report quiet speech
    // below 0.03 RMS; retain an adaptive floor to reject steady room noise.
    const threshold = Math.max(0.012, Math.min(0.024, this.noiseFloor * 2.3));
    const loud = level >= threshold;
    if (this.candidateAt === null) {
      if (!loud) this.noiseFloor = this.noiseFloor * 0.98 + level * 0.02;
      if (eligible && now >= this.cooldownUntil && loud) {
        this.candidateAt = now;
        this.lastVoiceAt = now;
        return "start";
      }
      return null;
    }

    if (loud) this.lastVoiceAt = now;
    if (this.confirmedAt === null) {
      // A single wake word can have less than 160 ms of voiced audio.
      if (loud && now - this.candidateAt >= 80) {
        this.confirmedAt = now;
        return "interrupt";
      }
      if (!loud && now - this.lastVoiceAt >= 180) {
        this.candidateAt = null;
        this.cooldownUntil = now + 250;
        return "discard";
      }
      return null;
    }

    if (
      now - this.confirmedAt >= 15_000 ||
      (now - this.confirmedAt >= 450 && now - this.lastVoiceAt >= 1_200)
    ) {
      this.candidateAt = null;
      this.confirmedAt = null;
      this.cooldownUntil = now + 500;
      return "finish";
    }
    return null;
  }
}

type MonitorOptions = {
  canListen: () => boolean;
  onSpeechStart: () => void;
  onQuestion: (audio: Blob, extension: string) => void;
  onError: (error: Error) => void;
};

type Candidate = {
  recorder: MediaRecorder;
  chunks: Blob[];
  accepted: boolean;
  discarded: boolean;
};

export class ActiveInterruptionMonitor {
  private readonly options: MonitorOptions;
  private readonly gate = new SpeechGate();
  private stream: MediaStream | null = null;
  private context: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private sampleBuffer: Float32Array<ArrayBuffer> | null = null;
  private timer: number | null = null;
  private candidate: Candidate | null = null;
  private mime = "";
  private stopped = false;

  constructor(options: MonitorOptions) {
    this.options = options;
  }

  /** Ask for microphone permission once, then monitor only while narration plays. */
  async start(): Promise<void> {
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder || !window.AudioContext) {
      throw new Error("This browser cannot monitor microphone audio here.");
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    if (this.stopped) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }
    this.stream = stream;
    this.mime = MediaRecorder.isTypeSupported("audio/webm")
      ? "audio/webm"
      : MediaRecorder.isTypeSupported("audio/ogg") ? "audio/ogg" : "";
    if (!this.mime) {
      this.stop();
      throw new Error("This browser has no supported microphone recording format.");
    }
    try {
      const context = new AudioContext();
      this.context = context;
      const source = context.createMediaStreamSource(stream);
      const analyser = context.createAnalyser();
      analyser.fftSize = 2048;
      const silentOutput = context.createGain();
      silentOutput.gain.value = 0;
      source.connect(analyser);
      analyser.connect(silentOutput);
      silentOutput.connect(context.destination); // Keep analysis active without mic feedback.
      this.analyser = analyser;
      this.sampleBuffer = new Float32Array(analyser.fftSize);
      await context.resume();
      if (this.stopped) return;
      stream.getAudioTracks().forEach((track) => track.addEventListener("ended", () => {
        if (!this.stopped) this.fail(new Error("Microphone access ended. Enable hands-free again."));
      }));
      this.timer = window.setInterval(() => this.sample(), 40);
    } catch (error) {
      this.stop();
      throw error;
    }
  }

  /** Release the microphone and discard any unfinished candidate recording. */
  stop(): void {
    this.stopped = true;
    if (this.timer !== null) window.clearInterval(this.timer);
    this.timer = null;
    if (this.candidate) this.discardCandidate();
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    if (this.context) void this.context.close();
    this.context = null;
    this.analyser = null;
  }

  private sample(): void {
    if (!this.analyser || !this.sampleBuffer) return;
    this.analyser.getFloatTimeDomainData(this.sampleBuffer);
    let energy = 0;
    for (const value of this.sampleBuffer) energy += value * value;
    const level = Math.sqrt(energy / this.sampleBuffer.length);
    const action = this.gate.update(level, performance.now(), this.options.canListen());
    try {
      if (action === "start") this.startCandidate();
      if (action === "discard") this.discardCandidate();
      if (action === "interrupt" && this.candidate) {
        this.candidate.accepted = true;
        this.options.onSpeechStart();
      }
      if (action === "finish") this.finishCandidate();
    } catch (error) {
      this.fail(error instanceof Error ? error : new Error("Microphone recording failed."));
    }
  }

  private startCandidate(): void {
    if (!this.stream) return;
    const recorder = new MediaRecorder(this.stream, { mimeType: this.mime });
    const candidate: Candidate = { recorder, chunks: [], accepted: false, discarded: false };
    recorder.ondataavailable = (event) => {
      if (event.data.size) candidate.chunks.push(event.data);
    };
    recorder.onstop = () => {
      if (this.candidate === candidate) this.candidate = null;
      if (candidate.accepted && !candidate.discarded && !this.stopped) {
        if (!candidate.chunks.length) {
          this.fail(new Error("No microphone audio was captured. Try again."));
        } else {
          this.options.onQuestion(
            new Blob(candidate.chunks, { type: this.mime }), this.mime.split("/")[1],
          );
        }
      }
    };
    recorder.onerror = () => this.fail(new Error("Microphone recording failed."));
    this.candidate = candidate;
    recorder.start(250); // Begin at first suspected speech, before sustained confirmation.
  }

  private discardCandidate(): void {
    const candidate = this.candidate;
    this.candidate = null;
    if (!candidate) return;
    candidate.discarded = true;
    if (candidate.recorder.state !== "inactive") candidate.recorder.stop();
  }

  private finishCandidate(): void {
    const candidate = this.candidate;
    this.candidate = null;
    if (candidate?.recorder.state === "recording") candidate.recorder.stop();
  }

  private fail(error: Error): void {
    this.stop();
    this.options.onError(error);
  }
}
