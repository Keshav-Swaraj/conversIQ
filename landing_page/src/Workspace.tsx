import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AudioLines, BookOpen, Check, ChevronLeft, ChevronRight, FileText,
  Headphones, Mic, Pause, Play, Plus, Send, ShieldCheck, SkipBack,
  SkipForward, Upload, X,
} from "lucide-react";
import PdfReadingPage from "./PdfReadingPage";
import { ActiveInterruptionMonitor } from "./activeInterruption";
import { LiveAudioPlayer } from "./liveAudio";
import { LiveVoiceSession, type VoiceEvent } from "./liveVoice";
import { wakeQuestion, wantsNarration } from "./sarahConversation";
import {
  narrateFrom, streamLiveAnswer, streamSarahSpeech, transcribeRecording,
  uploadPdf, type ConversationTurn, type NarrationSegment, type WorkspaceDocument,
} from "./workspaceApi";

type Cursor = { page: number; offset: number };
type ActiveSegment = Cursor & { text: string };
type Message = { role: "user" | "ai"; text: string; pages: number[]; error?: boolean; prompt?: boolean; id?: number };
type ReadySegment = Exclude<NarrationSegment, { complete: true }>;
type PreparedSegment = { segment: ReadySegment; audio: HTMLAudioElement };
type AnswerRun = {
  controller: AbortController;
  stopped: boolean;
  player: LiveAudioPlayer | null;
};
type PromptRun = { controller: AbortController; player: LiveAudioPlayer | null };

function segmentKey(docId: string, position: Cursor): string {
  return `${docId}:${position.page}:${position.offset}`;
}

function clock(seconds: number): string {
  const whole = Math.floor(Number.isFinite(seconds) ? seconds : 0);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

export default function Workspace() {
  const [documentInfo, setDocumentInfo] = useState<WorkspaceDocument | null>(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [selectedOffset, setSelectedOffset] = useState<number | null>(null);
  const [activeSegment, setActiveSegment] = useState<ActiveSegment | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [phase, setPhase] = useState("Ready when you are");
  const [busy, setBusy] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [recording, setRecording] = useState(false);
  const [handsFree, setHandsFree] = useState(false);
  const [conversationOpen, setConversationOpen] = useState(false);
  const [autoRecording, setAutoRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [answerActive, setAnswerActive] = useState(false);
  const [micError, setMicError] = useState("");
  const [speed, setSpeed] = useState(1);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [sourcePages, setSourcePages] = useState<number[] | null>(null);
  const [tab, setTab] = useState("read");
  const fileInput = useRef<HTMLInputElement>(null);
  const chat = useRef<HTMLDivElement>(null);
  const readerScroll = useRef<HTMLDivElement>(null);
  const seekReaderRef = useRef<(page: number, offset: number) => void>(() => {});
  const seekReader = useCallback((page: number, offset: number) => {
    seekReaderRef.current(page, offset);
  }, []);
  const narration = useRef<HTMLAudioElement>(null!);
  if (!narration.current) narration.current = new Audio();
  const answerRun = useRef<AnswerRun | null>(null);
  const voiceSession = useRef<LiveVoiceSession | null>(null);
  const voiceUserId = useRef<number | null>(null);
  const voiceAnswerId = useRef<number | null>(null);
  const voiceInputText = useRef("");
  const voiceAnswerText = useRef("");
  const voicePages = useRef<number[]>([]);
  const nextVoiceId = useRef(0);
  const micOptOut = useRef(false);
  const recorder = useRef<MediaRecorder | null>(null);
  const interruptionMonitor = useRef<ActiveInterruptionMonitor | null>(null);
  const pendingInterruption = useRef<{ requestId: number; docId: string } | null>(null);
  const conversationMode = useRef<"inactive" | "speaking" | "listening" | "processing">("inactive");
  const conversationHistory = useRef<ConversationTurn[]>([]);
  const promptToken = useRef(0);
  const promptTimeout = useRef<number | null>(null);
  const promptRun = useRef<PromptRun | null>(null);
  const busyRef = useRef(busy);
  const recordingRef = useRef(recording);
  const transcribingRef = useRef(transcribing);
  const autoRecordingRef = useRef(autoRecording);
  busyRef.current = busy;
  recordingRef.current = recording;
  transcribingRef.current = transcribing;
  autoRecordingRef.current = autoRecording;
  const cursor = useRef<Cursor>({ page: 1, offset: 0 });
  const desiredPlayback = useRef(false);
  const generation = useRef(0);
  const speedValue = useRef(1);
  const resumeAfterRecording = useRef(false);
  const preparedNext = useRef<{
    key: string;
    promise: Promise<PreparedSegment | null>;
  } | null>(null);

  useEffect(() => {
    return () => {
      generation.current += 1;
      narration.current.pause();
      void preparedNext.current?.promise.then((prepared) => {
        prepared?.audio.removeAttribute("src");
      });
      answerRun.current?.controller.abort();
      answerRun.current?.player?.stop();
      voiceSession.current?.close();
      recorder.current?.stop();
      interruptionMonitor.current?.stop();
      if (promptTimeout.current !== null) window.clearTimeout(promptTimeout.current);
      promptRun.current?.controller.abort();
      promptRun.current?.player?.stop();
    };
  }, []);

  useEffect(() => {
    if (chat.current) chat.current.scrollTop = chat.current.scrollHeight;
  }, [messages, phase]);

  const highlightedOffset = useMemo(() => {
    if (!activeSegment || !duration || !Number.isFinite(duration)) return selectedOffset;
    const words = [...activeSegment.text.matchAll(/\S+/g)];
    if (!words.length) return activeSegment.offset;
    const progress = Math.max(0, Math.min(1, currentTime / duration));
    const word = words[Math.min(words.length - 1, Math.floor(progress * words.length))];
    return activeSegment.offset + (word.index ?? 0);
  }, [activeSegment, currentTime, duration, selectedOffset]);
  const currentSection = documentInfo?.sections
    .filter((section) => section.page <= pageNumber)
    .at(-1);
  const currentSectionIndex = currentSection
    ? documentInfo!.sections.findIndex((section) => section.id === currentSection.id)
    : -1;

  useEffect(() => {
    if (!documentInfo) return;
    readerScroll.current?.querySelector(`[data-pdf-page="${pageNumber}"]`)
      ?.scrollIntoView({ block: "start", behavior: "smooth" });
  }, [documentInfo?.doc_id, pageNumber]);

  function stopAudio(): void {
    cancelAnswer(false);
    endConversation(false);
    generation.current += 1;
    desiredPlayback.current = false;
    narration.current.pause();
    narration.current.removeAttribute("src");
    const pending = preparedNext.current;
    preparedNext.current = null;
    if (pending) void pending.promise.then((prepared) => {
      prepared?.audio.removeAttribute("src");
    });
    setPlaying(false);
    setCurrentTime(0);
    setDuration(0);
    setActiveSegment(null);
  }

  async function handleUpload(file: File): Promise<void> {
    stopActiveInterruption(false);
    stopAudio();
    recorder.current?.stop();
    recorder.current = null;
    setRecording(false);
    setTranscribing(false);
    transcribingRef.current = false;
    setBusy(true);
    setPhase("Opening and indexing your PDF…");
    setMessages([]);
    conversationHistory.current = [];
    try {
      const uploaded = await uploadPdf(file);
      setDocumentInfo(uploaded);
      setPageNumber(1);
      setSelectedOffset(null);
      cursor.current = { page: 1, offset: 0 };
      setTab("read");
      setPhase("Document ready. Click a sentence or press Play.");
    } catch (error) {
      setPhase(error instanceof Error ? error.message : "Could not open this PDF.");
    } finally {
      setBusy(false);
    }
  }

  function goToPage(nextPage: number): void {
    if (recording || autoRecording || transcribing) return;
    if (!documentInfo || nextPage < 1 || nextPage > documentInfo.num_pages) return;
    stopAudio();
    cursor.current = { page: nextPage, offset: 0 };
    setPageNumber(nextPage);
    setSelectedOffset(null);
    setTab("read");
    setPhase(`Page ${nextPage} ready to read.`);
  }

  async function prepareSegment(
    docId: string, position: Cursor, requestId: number,
  ): Promise<PreparedSegment | null> {
    const segment = await narrateFrom(docId, position.page, position.offset);
    if (requestId !== generation.current || segment.complete) return null;
    const audio = new Audio(segment.audio_url);
    audio.preload = "auto";
    audio.load();
    return { segment, audio };
  }

  async function playSegment(docId: string, position: Cursor, requestId: number): Promise<void> {
    setPhase("Preparing narration…");
    try {
      const pending = preparedNext.current?.key === segmentKey(docId, position)
        ? preparedNext.current.promise : null;
      preparedNext.current = null;
      const prepared = (pending ? await pending : null)
        ?? await prepareSegment(docId, position, requestId);
      if (requestId !== generation.current) return;
      if (!prepared) {
        desiredPlayback.current = false;
        setPlaying(false);
        setPhase("Narration complete.");
        return;
      }
      const { segment, audio } = prepared;
      cursor.current = { page: segment.page, offset: segment.offset };
      setPageNumber(segment.page);
      setSelectedOffset(segment.offset);
      setActiveSegment({ page: segment.page, offset: segment.offset, text: segment.text });
      setCurrentTime(0);
      if (narration.current !== audio) {
        narration.current.pause();
        narration.current.removeAttribute("src");
      }
      narration.current = audio;
      audio.playbackRate = speedValue.current;
      if (Number.isFinite(audio.duration)) setDuration(audio.duration);
      audio.onloadedmetadata = () => setDuration(audio.duration);
      audio.ontimeupdate = () => setCurrentTime(audio.currentTime);
      audio.onended = () => {
        if (!desiredPlayback.current || requestId !== generation.current) return;
        if (segment.next_complete) {
          desiredPlayback.current = false;
          setPlaying(false);
          setPhase("Narration complete.");
        } else {
          void playSegment(docId, {
            page: segment.next_page,
            offset: segment.next_offset,
          }, requestId);
        }
      };
      if (desiredPlayback.current) {
        await audio.play();
        setPlaying(true);
        setPhase(`Narrating page ${segment.page}`);
        if (!segment.next_complete) {
          const next = { page: segment.next_page, offset: segment.next_offset };
          preparedNext.current = {
            key: segmentKey(docId, next),
            promise: prepareSegment(docId, next, requestId).catch(() => null),
          };
        }
      }
    } catch (error) {
      if (requestId !== generation.current) return;
      desiredPlayback.current = false;
      setPlaying(false);
      setPhase(error instanceof Error ? error.message : "Narration could not start.");
    }
  }

  function startAt(position: Cursor): void {
    if (!documentInfo || busy || recording || autoRecording || transcribing) return;
    stopAudio();
    cursor.current = position;
    desiredPlayback.current = true;
    if (!micOptOut.current) void enableActiveInterruption();
    const requestId = generation.current;
    void playSegment(documentInfo.doc_id, position, requestId);
  }

  seekReaderRef.current = (page, offset) => startAt({ page, offset });

  function togglePlayback(): void {
    if (!documentInfo || busy || recording || autoRecording || transcribing) return;
    if (conversationMode.current !== "inactive") {
      endConversation(true);
      return;
    }
    const audio = narration.current;
    if (desiredPlayback.current) {
      desiredPlayback.current = false;
      audio.pause();
      setPlaying(false);
      setPhase("Narration paused. Your place is saved.");
    } else if (audio.src && !audio.ended) {
      desiredPlayback.current = true;
      if (!micOptOut.current) void enableActiveInterruption();
      void audio.play().then(() => {
        setPlaying(true);
        setPhase(`Narrating page ${cursor.current.page}`);
      }).catch((error: Error) => setPhase(error.message));
    } else {
      startAt(cursor.current);
    }
  }

  function resumeNarration(): void {
    if (!documentInfo) return;
    desiredPlayback.current = true;
    if (!micOptOut.current) void enableActiveInterruption();
    if (!narration.current.src || narration.current.ended) {
      void playSegment(documentInfo.doc_id, cursor.current, generation.current);
      return;
    }
    void narration.current.play().then(() => {
      setPlaying(true);
      setPhase(`Narrating page ${cursor.current.page}`);
    }).catch((error: Error) => setPhase(error.message));
  }

  function stopPrompt(): void {
    promptToken.current += 1;
    if (promptTimeout.current !== null) window.clearTimeout(promptTimeout.current);
    promptTimeout.current = null;
    promptRun.current?.controller.abort();
    promptRun.current?.player?.stop();
    promptRun.current = null;
  }

  function speakPrompt(text: string, after: () => void): void {
    stopPrompt();
    const token = promptToken.current;
    const run: PromptRun = { controller: new AbortController(), player: null };
    promptRun.current = run;
    let settled = false;
    const finish = (): void => {
      if (settled || promptToken.current !== token) return;
      settled = true;
      stopPrompt();
      after();
    };
    promptTimeout.current = window.setTimeout(() => {
      if (promptToken.current !== token) return;
      setPhase("Sarah's spoken prompt timed out. You can still ask a question.");
      finish();
    }, 35_000);
    void (async () => {
      let receivedAudio = false;
      let completed = false;
      for await (const event of streamSarahSpeech(text, run.controller.signal)) {
        if (promptToken.current !== token) return;
        if (event.type === "audio") {
          if (!run.player) run.player = new LiveAudioPlayer();
          run.player.append(event.pcm_base64);
          receivedAudio = true;
        } else if (event.type === "done") completed = true;
      }
      if (!completed || !receivedAudio) throw new Error("Sarah's voice returned no audio.");
      await run.player?.finish();
      finish();
    })().catch((error: unknown) => {
      if (promptToken.current !== token) return;
      const message = error instanceof Error ? error.message : "Sarah's voice is unavailable.";
      setMessages((items) => [...items, {
        role: "ai", text: `Voice unavailable: ${message}`, pages: [], error: true, prompt: true,
      }]);
      finish();
    });
  }

  function endConversation(resume: boolean): void {
    stopPrompt();
    voiceSession.current?.close();
    voiceSession.current = null;
    voiceUserId.current = null;
    voiceAnswerId.current = null;
    voiceInputText.current = "";
    voiceAnswerText.current = "";
    voicePages.current = [];
    setAnswerActive(false);
    conversationMode.current = "inactive";
    setConversationOpen(false);
    if (resume) resumeNarration();
  }

  function inviteFollowup(): void {
    if (conversationMode.current === "inactive") return;
    conversationMode.current = "speaking";
    const prompt = "Should I continue narration, or do you have another question?";
    setMessages((items) => [...items, { role: "ai", text: prompt, pages: [], prompt: true }]);
    setPhase("Sarah is asking if you'd like to continue.");
    speakPrompt(prompt, () => {
      if (conversationMode.current === "speaking") {
        conversationMode.current = "listening";
        setPhase("Ask another question, or say continue narration.");
      }
    });
  }

  function onVoiceEvent(event: VoiceEvent): void {
    if (event.type === "resume") {
      endConversation(true);
      return;
    }
    if (event.type === "error") {
      setMessages((items) => [...items, {
        role: "ai", text: event.message, pages: [], error: true, prompt: true,
      }]);
      setPhase(event.message);
      endConversation(false);
      return;
    }
    if (event.type === "not_grounded") {
      setMessages((items) => [...items, {
        role: "ai", text: "I need to check the document before answering that. Please ask again.",
        pages: [], error: true, prompt: true,
      }]);
      setAnswerActive(false);
      setPhase("Sarah is listening. Please ask again.");
      voiceAnswerId.current = null;
      voiceAnswerText.current = "";
      voicePages.current = [];
      voiceUserId.current = null;
      voiceInputText.current = "";
      return;
    }
    if (event.type === "input_start") {
      voiceUserId.current = null;
      voiceInputText.current = "";
      return;
    }
    if (event.type === "input") {
      if (voiceUserId.current === null) {
        voiceUserId.current = ++nextVoiceId.current;
        voiceInputText.current = "";
        const id = voiceUserId.current;
        setMessages((items) => [...items, {
          id, role: "user", text: "", pages: [],
        }]);
      }
      voiceInputText.current = event.text.startsWith(voiceInputText.current)
        ? event.text : voiceInputText.current + event.text;
      const id = voiceUserId.current;
      const text = voiceInputText.current;
      setMessages((items) => items.map((item) => item.id === id ? { ...item, text } : item));
      if (event.finished) {
        voiceUserId.current = null;
        voiceInputText.current = "";
      }
      return;
    }
    if (event.type === "sources") {
      voicePages.current = event.relevant_pages;
      if (voiceAnswerId.current === null) {
        voiceAnswerId.current = ++nextVoiceId.current;
        const id = voiceAnswerId.current;
        setMessages((items) => [...items, {
          id, role: "ai", text: "", pages: event.relevant_pages,
        }]);
      } else {
        const id = voiceAnswerId.current;
        setMessages((items) => items.map((item) => item.id === id
          ? { ...item, pages: event.relevant_pages, prompt: false } : item));
      }
      setPhase(event.relevant_pages.length ? "Sarah found relevant pages…" : "Sarah found no matching pages.");
      return;
    }
    if (event.type === "cited_pages") {
      voicePages.current = event.relevant_pages;
      const id = voiceAnswerId.current;
      if (id !== null) setMessages((items) => items.map((item) => item.id === id
        ? { ...item, pages: event.relevant_pages } : item));
      return;
    }
    if (event.type === "delta") {
      if (voiceAnswerId.current === null) {
        voiceAnswerId.current = ++nextVoiceId.current;
        voiceAnswerText.current = "";
        const id = voiceAnswerId.current;
        setMessages((items) => [...items, {
          id, role: "ai", text: "", pages: voicePages.current,
          prompt: voicePages.current.length === 0,
        }]);
      }
      voiceAnswerText.current += event.text;
      const id = voiceAnswerId.current;
      const text = voiceAnswerText.current;
      setMessages((items) => items.map((item) => item.id === id ? { ...item, text } : item));
      return;
    }
    if (event.type === "audio") {
      setAnswerActive(true);
      setPhase("Sarah is speaking. Interrupt whenever you like.");
    } else if (event.type === "interrupted") {
      setAnswerActive(false);
      voiceAnswerId.current = null;
      voiceAnswerText.current = "";
      voicePages.current = [];
      setPhase("Sarah is listening…");
    } else if (event.type === "playback_done") {
      setAnswerActive(false);
      setPhase("Sarah is listening…");
    } else if (event.type === "done") {
      voiceAnswerId.current = null;
      voiceAnswerText.current = "";
      voicePages.current = [];
      voiceUserId.current = null;
      voiceInputText.current = "";
    }
  }

  function openConversation(queuedQuestion = ""): void {
    if (!documentInfo) return;
    stopActiveInterruption(false);
    narration.current.pause();
    desiredPlayback.current = false;
    setPlaying(false);
    conversationMode.current = "listening";
    setConversationOpen(true);
    setTab("chat");
    setPhase("Connecting Sarah's live voice…");
    const session = new LiveVoiceSession(onVoiceEvent);
    voiceSession.current = session;
    void session.start(documentInfo.doc_id).then(() => {
      if (voiceSession.current !== session) return;
      setPhase("Sarah is listening. Ask about your document or say continue narration.");
      if (queuedQuestion) session.sendText(queuedQuestion);
    }).catch((error: unknown) => {
      if (voiceSession.current !== session) return;
      const message = error instanceof Error ? error.message : "Sarah could not connect.";
      onVoiceEvent({ type: "error", message });
    });
  }

  function submitQuestion(): void {
    const text = question.trim();
    if (!text) return;
    const session = voiceSession.current;
    if (!session) { void ask(text); return; }
    setQuestion("");
    voiceUserId.current = ++nextVoiceId.current;
    voiceInputText.current = text;
    const id = voiceUserId.current;
    setMessages((items) => [...items, {
      id, role: "user", text, pages: [],
    }]);
    try { session.sendText(text); }
    catch (error) {
      onVoiceEvent({
        type: "error", message: error instanceof Error ? error.message : "Sarah is not connected.",
      });
    }
  }

  function cancelAnswer(resume: boolean): void {
    const run = answerRun.current;
    if (!run) return;
    run.stopped = true;
    run.controller.abort();
    run.player?.stop();
    answerRun.current = null;
    setAnswerActive(false);
    setBusy(false);
    if (resume) {
      endConversation(false);
      setPhase("Answer stopped. Resuming narration…");
      resumeNarration();
    }
  }

  async function ask(text: string, shouldResume = desiredPlayback.current): Promise<void> {
    if (!documentInfo || !text.trim() || busy) return;
    const inConversation = conversationMode.current !== "inactive";
    if (inConversation) {
      stopPrompt();
      conversationMode.current = "processing";
    }
    const history = conversationHistory.current.slice(-4);
    let answerText = "";
    const run: AnswerRun = {
      controller: new AbortController(), stopped: false,
      player: null,
    };
    answerRun.current = run;
    narration.current.pause();
    desiredPlayback.current = false;
    setPlaying(false);
    setBusy(true);
    setAnswerActive(true);
    setQuestion("");
    setTab("chat");
    setMessages((items) => [...items,
      { role: "user", text: text.trim(), pages: [] },
      { role: "ai", text: "", pages: [] },
    ]);
    setPhase("Finding an answer in your document…");
    let receivedDone = false;
    const appendDelta = (delta: string): void => {
      answerText += delta;
      setMessages((items) => items.map((item, index) => index === items.length - 1
        ? { ...item, text: item.text + delta } : item));
    };
    try {
      for await (const event of streamLiveAnswer(
        documentInfo.doc_id, text.trim(), run.controller.signal, history,
      )) {
        if (run.stopped) return;
        if (event.type === "sources") {
          setMessages((items) => items.map((item, index) => index === items.length - 1
            ? { ...item, pages: event.relevant_pages } : item));
        } else if (event.type === "delta") {
          appendDelta(event.text);
        } else if (event.type === "audio") {
          if (!run.player) run.player = new LiveAudioPlayer();
          run.player.append(event.pcm_base64);
          setPhase("Speaking answer…");
        } else if (event.type === "done") {
          receivedDone = true;
        }
      }
      if (!receivedDone) throw new Error("The answer stream ended unexpectedly.");
      await run.player?.finish();
      if (run.stopped) return;
      setPhase("Answer complete.");
      if (answerText.trim()) conversationHistory.current = [
        ...conversationHistory.current,
        { question: text.trim(), answer: answerText.trim().slice(0, 2_000) },
      ].slice(-4);
      if (inConversation) {
        if (run.player || !answerText.trim()) inviteFollowup();
        else speakPrompt(answerText.trim(), inviteFollowup);
      }
      else if (shouldResume) resumeNarration();
    } catch (error) {
      if (!run.stopped) {
        const message = error instanceof Error ? error.message : "Could not answer the question.";
        setMessages((items) => items.map((item, index) => index === items.length - 1
          ? { ...item, text: item.text ? `${item.text}\n\n${message}` : message, error: true }
          : item));
        setPhase(message);
        if (inConversation) {
          conversationMode.current = "listening";
          setPhase("Sarah is listening. Ask again, or say continue narration.");
        } else if (shouldResume) resumeNarration();
      }
    } finally {
      run.player?.stop();
      if (answerRun.current === run) {
        answerRun.current = null;
        setAnswerActive(false);
        setBusy(false);
      }
    }
  }

  async function processRecordedQuestion(
    blob: Blob, extension: string, shouldResume: boolean,
    target: { requestId: number; docId: string },
  ): Promise<void> {
    if (target.requestId !== generation.current || target.docId !== documentInfo?.doc_id) return;
    setPhase("Transcribing your question…");
    setTranscribing(true);
    transcribingRef.current = true;
    try {
      const transcript = await transcribeRecording(blob, extension);
      if (target.requestId !== generation.current || target.docId !== documentInfo?.doc_id) return;
      if (!transcript.trim()) throw new Error("No speech was recognized. Try again.");
      setTranscribing(false);
      transcribingRef.current = false;
      await ask(transcript, shouldResume);
    } catch (error) {
      if (target.requestId === generation.current && target.docId === documentInfo?.doc_id) {
        setPhase(error instanceof Error ? error.message : "Could not transcribe the question.");
        if (shouldResume) resumeNarration();
      }
    } finally {
      if (target.requestId === generation.current && target.docId === documentInfo?.doc_id) {
        setTranscribing(false);
        transcribingRef.current = false;
      }
    }
  }

  async function processHandsFreeSpeech(
    blob: Blob, extension: string, target: { requestId: number; docId: string },
  ): Promise<void> {
    if (target.requestId !== generation.current || target.docId !== documentInfo?.doc_id) return;
    const wasListening = conversationMode.current === "listening";
    setTranscribing(true);
    transcribingRef.current = true;
    if (!wasListening) setPhase("Checking for Sarah while narration continues…");
    if (wasListening) conversationMode.current = "processing";
    try {
      const transcript = (await transcribeRecording(blob, extension, !wasListening)).trim();
      if (target.requestId !== generation.current || target.docId !== documentInfo?.doc_id) return;
      setMicError("");
      if (wasListening) {
        if (!transcript) {
          conversationMode.current = "listening";
          setPhase("I didn't catch that. Please ask Sarah again.");
          return;
        }
        if (wantsNarration(transcript)) {
          endConversation(true);
          return;
        }
        if (/^(?:no|another question|one more question)[.!?]*$/i.test(transcript)) {
          conversationMode.current = "listening";
          conversationMode.current = "speaking";
          speakPrompt("Sure. What's your question?", () => {
            if (conversationMode.current === "speaking") conversationMode.current = "listening";
          });
          return;
        }
        conversationMode.current = "listening";
        setTranscribing(false);
        transcribingRef.current = false;
        await ask(transcript, false);
      } else {
        const queuedQuestion = wakeQuestion(transcript);
        if (queuedQuestion !== null && !busyRef.current) openConversation(queuedQuestion);
        else if (!narration.current.paused) setPhase(transcript
          ? `Heard “${transcript.slice(0, 60)}”, not Sarah. Narration continues.`
          : "Heard speech, but no words were recognized. Try saying Hey Sarah clearly.");
      }
    } catch (error) {
      if (target.requestId === generation.current && target.docId === documentInfo?.doc_id) {
        const message = error instanceof Error ? error.message : "Speech recognition failed.";
        setMicError(message);
        setPhase(`Microphone question failed: ${message}`);
        if (wasListening) {
          conversationMode.current = "listening";
          setMessages((items) => [...items, {
            role: "ai", text: `I couldn't hear that question. ${message}`, pages: [], error: true, prompt: true,
          }]);
        }
      }
    } finally {
      if (target.requestId === generation.current && target.docId === documentInfo?.doc_id) {
        setTranscribing(false);
        transcribingRef.current = false;
      }
    }
  }

  function stopActiveInterruption(resumeOnCancel = true): void {
    interruptionMonitor.current?.stop();
    interruptionMonitor.current = null;
    pendingInterruption.current = null;
    autoRecordingRef.current = false;
    setAutoRecording(false);
    setHandsFree(false);
    if (conversationMode.current !== "inactive") endConversation(resumeOnCancel);
  }

  async function enableActiveInterruption(): Promise<void> {
    if (interruptionMonitor.current) return;
    if (!documentInfo || busy || recording || transcribing) return;
    const docId = documentInfo.doc_id;
    const monitor = new ActiveInterruptionMonitor({
      canListen: () => (
        conversationMode.current === "listening" ||
        (conversationMode.current === "inactive" && desiredPlayback.current && !narration.current.paused)
      ) && !busyRef.current && !recordingRef.current && !transcribingRef.current,
      onSpeechStart: () => {
        pendingInterruption.current = { requestId: generation.current, docId };
        autoRecordingRef.current = true;
        setAutoRecording(true);
        setPhase(conversationMode.current === "listening"
          ? "Sarah is listening…" : "Speech detected; checking for Sarah…");
      },
      onQuestion: (blob, extension) => {
        const target = pendingInterruption.current;
        pendingInterruption.current = null;
        autoRecordingRef.current = false;
        setAutoRecording(false);
        if (target) void processHandsFreeSpeech(blob, extension, target);
      },
      onError: (error) => {
        const target = pendingInterruption.current;
        stopActiveInterruption(false);
        setMicError(error.message);
        setPhase(error.message);
        if (target?.requestId === generation.current && narration.current.paused) resumeNarration();
      },
    });
    interruptionMonitor.current = monitor;
    setMicError("");
    try {
      await monitor.start();
      if (interruptionMonitor.current === monitor) {
        setHandsFree(true);
      }
    } catch (error) {
      if (interruptionMonitor.current === monitor) {
        stopActiveInterruption(false);
        micOptOut.current = true;
        const message = error instanceof Error ? error.message : "Microphone unavailable.";
        setMicError(message);
        setPhase(`Wake-word microphone unavailable: ${message}`);
      }
    }
  }

  function toggleActiveInterruption(): void {
    if (interruptionMonitor.current) {
      micOptOut.current = true;
      stopActiveInterruption();
      return;
    }
    micOptOut.current = false;
    void enableActiveInterruption();
  }

  async function toggleMicrophone(): Promise<void> {
    if (recording && recorder.current) {
      recorder.current.stop();
      recorder.current = null;
      setRecording(false);
      return;
    }
    if (!documentInfo || busy || autoRecording || transcribing) return;
    if (interruptionMonitor.current) stopActiveInterruption(false);
    try {
      if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
        throw new Error("This browser cannot record microphone audio here.");
      }
      resumeAfterRecording.current = desiredPlayback.current;
      narration.current.pause();
      desiredPlayback.current = false;
      setPlaying(false);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mime = MediaRecorder.isTypeSupported("audio/webm")
        ? "audio/webm"
        : MediaRecorder.isTypeSupported("audio/ogg") ? "audio/ogg" : "";
      if (!mime) {
        stream.getTracks().forEach((track) => track.stop());
        throw new Error("This browser has no supported recording format.");
      }
      const activeRecorder = new MediaRecorder(stream, { mimeType: mime });
      const recordingTarget = { requestId: generation.current, docId: documentInfo.doc_id };
      const chunks: Blob[] = [];
      activeRecorder.ondataavailable = (event) => {
        if (event.data.size) chunks.push(event.data);
      };
      activeRecorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        void processRecordedQuestion(
          new Blob(chunks, { type: mime }), mime.split("/")[1],
          resumeAfterRecording.current,
          recordingTarget,
        );
      };
      recorder.current = activeRecorder;
      activeRecorder.start();
      setRecording(true);
      setPhase("Listening… click the microphone again when finished.");
    } catch (error) {
      setPhase(error instanceof Error ? error.message : "Microphone unavailable.");
      if (resumeAfterRecording.current) resumeNarration();
    }
  }

  return (
    <div className="workspace-shell">
      <div className="app-window workspace-window">
        <div className="app-topbar">
          <a className="workspace-brand" href="/"><AudioLines size={15} /> ConversIQ
            <span className="top-divider">/</span> Workspace</a>
          <span className="demo-badge"><span /> Local workspace</span>
        </div>
        <div className="mobile-tabs">
          {[["chapters", "Contents"], ["read", "Document"], ["chat", "Conversation"]].map(([id, label]) => (
            <button key={id} className={tab === id ? "selected" : ""} onClick={() => setTab(id)}>{label}</button>
          ))}
        </div>
        <div className="app-body">
          <aside className={`document-sidebar ${tab === "chapters" ? "mobile-visible" : ""}`}>
            <div className="panel-heading">YOUR LIBRARY
              <button className="icon-button" aria-label="Upload a PDF" onClick={() => fileInput.current?.click()} disabled={busy}><Plus size={17} /></button>
            </div>
            <input ref={fileInput} type="file" accept=".pdf,application/pdf" hidden
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void handleUpload(file);
                event.target.value = "";
              }} />
            <div className="document-name">
              <span className="pdf-icon"><FileText size={20} /></span>
              <div>{documentInfo?.filename || "No document yet"}<small>{documentInfo ? `PDF · ${documentInfo.num_pages} pages` : "Choose a PDF to begin"}</small></div>
            </div>
            <div className="sidebar-section">CONTENTS <span>{documentInfo?.sections.length || 0} sections</span></div>
            <div className="chapter-list workspace-chapters">
              {documentInfo?.sections.map((section, index) => (
                <button key={`${section.id}-${index}`} className={currentSectionIndex === index ? "selected" : ""}
                  onClick={() => goToPage(section.page)} title={`Pages ${section.page}–${section.end_page}`}>
                  <span>{String(index + 1).padStart(2, "0")}</span>{section.title}
                  {currentSectionIndex === index && <AudioLines size={14} />}
                </button>
              ))}
              {documentInfo && !documentInfo.sections.length && <p className="workspace-hint">No chapter headings were found. Use the page controls to explore the PDF.</p>}
            </div>
            <div className="indexed"><ShieldCheck size={16} />
              <div>{documentInfo ? "Structure understood" : "Waiting for a document"}
                <small>{documentInfo ? "Indexed with PageIndex Flash" : "Your PDF stays on this computer"}</small></div>
              {documentInfo && <span className="status-dot" />}
            </div>
          </aside>

          <main className={`document-reader ${tab === "read" ? "mobile-visible" : ""}`}>
            <div className="reader-toolbar">
              <span><BookOpen size={15} /> Reading view</span>
              <span>{documentInfo ? `Page ${pageNumber} of ${documentInfo.num_pages}` : "Your next chapter starts here"}</span>
            </div>
            <div className={`book-content workspace-book${documentInfo ? " pdf-book" : ""}`} ref={readerScroll}>
              {documentInfo ? (
                <>
                  <div className="pdf-page-stack" aria-label={`Original pages from ${documentInfo.filename}`}>
                    {documentInfo.pages.map((page) => (
                      <PdfReadingPage key={`${documentInfo.doc_id}-${page.page_number}`}
                        docId={documentInfo.doc_id}
                        pageNumber={page.page_number}
                        text={page.text}
                        activeOffset={activeSegment?.page === page.page_number ? highlightedOffset : null}
                        playing={playing && activeSegment?.page === page.page_number}
                        onSeek={seekReader}
                      />
                    ))}
                  </div>
                  <div className="reader-page workspace-page-nav">
                    <button disabled={pageNumber <= 1} onClick={() => goToPage(pageNumber - 1)}><ChevronLeft size={15} /> Previous page</button>
                    <span>Page {pageNumber} / {documentInfo.num_pages}</span>
                    <button disabled={pageNumber >= documentInfo.num_pages} onClick={() => goToPage(pageNumber + 1)}>Next page <ChevronRight size={15} /></button>
                  </div>
                </>
              ) : (
                <div className="workspace-empty">
                  <span className="workspace-empty-icon"><Upload size={24} /></span>
                  <div className="chapter-kicker">YOUR READING SPACE</div>
                  <h3>Bring a document to life.</h3>
                  <p>Upload a PDF to explore its structure, listen to it, and ask grounded questions.</p>
                  <button className="button primary" onClick={() => fileInput.current?.click()} disabled={busy}>Choose a PDF <Plus size={16} /></button>
                </div>
              )}
            </div>
            <div className="narration-status" role="status"><span className={`status-dot ${playing ? "pulse" : ""}`} />{phase}
              <button className={`interrupt-toggle ${handsFree ? "active" : ""}`} type="button"
                aria-pressed={handsFree} aria-label={handsFree ? "Turn off Sarah wake-word listening" : "Turn on Sarah wake-word listening"}
                title={micError || "Say Sarah to pause narration and start a conversation. Headphones reduce false triggers."}
                disabled={!documentInfo || conversationOpen || (!handsFree && (busy || recording || transcribing))}
                onClick={() => void toggleActiveInterruption()}>
                <Mic size={12} /> {handsFree ? conversationOpen ? "Talking with Sarah" : "Say Sarah to talk" : micError ? "Microphone off - retry" : "Wake word starts with narration"}
              </button>
            </div>
          </main>

          <aside className={`conversation-panel ${tab === "chat" ? "mobile-visible" : ""}`}>
            <div className="panel-heading"><span><AudioLines size={16} /> CONVERSATION</span><span className="context-pill">Context on</span></div>
            <div className="chat-messages" ref={chat}>
              <div className="chat-start">{documentInfo ? "Say Sarah during narration, or type a question about this document." : "Your conversation begins after upload."}</div>
              {messages.map((message, index) => (
                <div className={`message ${message.role}`} key={index}>
                  <div className="message-author">{message.role === "ai" ? <><span className="mini-logo"><AudioLines size={12} /></span> Sarah</> : <><span className="avatar">Y</span> You</>}</div>
                  <p>{message.text || (message.role === "ai"
                    ? message.pages.length ? "Preparing spoken answer…" : "Finding the right pages…"
                    : "")}</p>
                  {message.role === "ai" && !message.prompt && answerActive && index === messages.length - 1 &&
                    <button className="stop-answer" type="button" onClick={() => {
                      if (voiceSession.current) endConversation(true);
                      else cancelAnswer(true);
                    }}>
                      <Pause size={12} /> Stop answer and resume narration
                    </button>}
                  {message.role === "ai" && !message.prompt && <div className="source-row">
                    {message.pages.length > 0 && <button className="source-chip" onClick={() => setSourcePages(message.pages)}><BookOpen size={12} /> {message.pages.length <= 4 ? `Page${message.pages.length > 1 ? "s" : ""} ${message.pages.join(", ")}` : `Section pages ${message.pages[0]}–${message.pages.at(-1)}`}</button>}
                    <span><Check size={12} /> {message.error ? "Answer unavailable" : message.pages.length ? "Grounded" : "No matching source"}</span>
                  </div>}
                </div>
              ))}
              {busy && !answerActive && documentInfo && <div className="processing"><AudioLines size={17} /> Working with your document…</div>}
              {conversationOpen && <button className="stop-answer" type="button" onClick={() => {
                cancelAnswer(false);
                endConversation(true);
              }}>Continue narration</button>}
            </div>
            <form className="chat-input" onSubmit={(event) => { event.preventDefault(); submitQuestion(); }}>
              <input aria-label="Ask about the document" placeholder="Ask about this document…" value={question}
                onChange={(event) => setQuestion(event.target.value)} disabled={!documentInfo || busy || recording} />
              <button aria-label="Send question" disabled={!documentInfo || !question.trim() || busy || recording}><Send size={17} /></button>
            </form>
            <div className="chat-footnote"><ShieldCheck size={12} /> Answers stay within your document.</div>
          </aside>
        </div>

        <div className="player">
          <div className="player-title"><span className="headphones"><Headphones size={20} /></span>
            <div>{currentSection?.title || (documentInfo ? `Page ${pageNumber}` : "Your document")}
              <small>{documentInfo ? `Page ${pageNumber} · Narration` : "Upload a PDF to begin"}</small></div>
          </div>
          <div className="playback">
            <div className="playback-buttons">
              <button className="icon-button" aria-label="Previous page" disabled={!documentInfo || pageNumber <= 1} onClick={() => goToPage(pageNumber - 1)}><SkipBack size={17} /></button>
              <button className="play-button" aria-label={playing ? "Pause narration" : "Play narration"} disabled={!documentInfo || busy || recording || autoRecording || transcribing} onClick={togglePlayback}>
                {playing ? <Pause size={18} fill="currentColor" /> : <Play size={18} fill="currentColor" />}
              </button>
              <button className="icon-button" aria-label="Next page" disabled={!documentInfo || pageNumber >= documentInfo.num_pages} onClick={() => goToPage(pageNumber + 1)}><SkipForward size={17} /></button>
              <button className="speed-button" aria-label="Change narration speed" onClick={() => {
                const next = speed === 1 ? 1.25 : speed === 1.25 ? 1.5 : 1;
                speedValue.current = next;
                narration.current.playbackRate = next;
                setSpeed(next);
              }}>{speed}×</button>
            </div>
            <div className="progress-row"><span>{clock(currentTime)}</span>
              <input aria-label="Current audio segment progress" type="range" min="0" max={duration || 1} step="0.1" value={Math.min(currentTime, duration || 1)}
                disabled={!documentInfo || !duration} onChange={(event) => {
                  narration.current.currentTime = Number(event.target.value);
                  setCurrentTime(Number(event.target.value));
                }} style={{ background: `linear-gradient(to right,#a593f5 ${duration ? currentTime / duration * 100 : 0}%,#30303a 0%)` }} />
              <span>{clock(duration)}</span>
            </div>
          </div>
          <button className={`ask-button ${recording ? "listening" : ""}`} disabled={!documentInfo || conversationOpen || autoRecording || transcribing || (busy && !recording)} onClick={() => void toggleMicrophone()}>
            <Mic size={18} />{recording ? "Stop recording" : "Ask ConversIQ"}<span>↵</span>
          </button>
        </div>
      </div>
      <div className="workspace-footer"><span><ShieldCheck size={13} /> Your document. Your context. Every answer grounded.</span><a href="/">Back to ConversIQ <ChevronRight size={13} /></a></div>
      {sourcePages && documentInfo && <div className="modal-backdrop" onClick={() => setSourcePages(null)}>
        <div className="source-modal" role="dialog" aria-modal="true" aria-label="Source pages" onClick={(event) => event.stopPropagation()}>
          <button autoFocus className="icon-button modal-close" aria-label="Close sources" onClick={() => setSourcePages(null)}><X /></button>
          <BookOpen className="violet" /><h3>Grounded in your document</h3>
          <p>Relevant pages: {sourcePages.join(", ")}</p>
          <small>{documentInfo.pages[sourcePages[0] - 1]?.text.slice(0, 450) || "No extractable text on this page."}</small>
          <button className="button primary" onClick={() => { goToPage(sourcePages[0]); setSourcePages(null); }}>Open page {sourcePages[0]}</button>
        </div>
      </div>}
    </div>
  );
}
