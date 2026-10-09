import { useEffect, useMemo, useRef, useState } from "react";
import {
  AudioLines, BookOpen, Check, ChevronLeft, ChevronRight, FileText,
  Headphones, Mic, Pause, Play, Plus, Send, ShieldCheck, SkipBack,
  SkipForward, Upload, X,
} from "lucide-react";
import { sentencesWithOffsets } from "./readerText";
import {
  askDocument, getAnswerAudio, loadDocument, narrateFrom, transcribeRecording,
  uploadPdf, type NarrationSegment, type WorkspaceDocument,
} from "./workspaceApi";

type Cursor = { page: number; offset: number };
type Message = { role: "user" | "ai"; text: string; pages: number[] };
type ReadySegment = Exclude<NarrationSegment, { complete: true }>;
type PreparedSegment = { segment: ReadySegment; audio: HTMLAudioElement };

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
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [phase, setPhase] = useState("Ready when you are");
  const [busy, setBusy] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [recording, setRecording] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [sourcePages, setSourcePages] = useState<number[] | null>(null);
  const [tab, setTab] = useState("read");
  const fileInput = useRef<HTMLInputElement>(null);
  const chat = useRef<HTMLDivElement>(null);
  const narration = useRef<HTMLAudioElement>(null!);
  if (!narration.current) narration.current = new Audio();
  const answerAudio = useRef<HTMLAudioElement | null>(null);
  const recorder = useRef<MediaRecorder | null>(null);
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
    let active = true;
    const previousId = localStorage.getItem("conversiq-document-id");
    if (previousId) {
      loadDocument(previousId)
        .then((loaded) => {
          if (active) setDocumentInfo(loaded);
        })
        .catch(() => {
          if (active) localStorage.removeItem("conversiq-document-id");
        });
    }
    return () => {
      active = false;
      generation.current += 1;
      narration.current.pause();
      void preparedNext.current?.promise.then((prepared) => {
        prepared?.audio.removeAttribute("src");
      });
      answerAudio.current?.pause();
      recorder.current?.stop();
    };
  }, []);

  useEffect(() => {
    if (chat.current) chat.current.scrollTop = chat.current.scrollHeight;
  }, [messages, phase]);

  const currentPage = documentInfo?.pages[pageNumber - 1];
  const sentences = useMemo(
    () => sentencesWithOffsets(currentPage?.text || ""),
    [currentPage?.text],
  );
  const highlightedSentenceStart = selectedOffset === null
    ? null
    : sentences.filter((sentence) => sentence.start <= selectedOffset).at(-1)?.start;
  const currentSection = documentInfo?.sections
    .filter((section) => section.page <= pageNumber)
    .at(-1);
  const currentSectionIndex = currentSection
    ? documentInfo!.sections.findIndex((section) => section.id === currentSection.id)
    : -1;

  function stopAudio(): void {
    generation.current += 1;
    desiredPlayback.current = false;
    narration.current.pause();
    narration.current.removeAttribute("src");
    const pending = preparedNext.current;
    preparedNext.current = null;
    if (pending) void pending.promise.then((prepared) => {
      prepared?.audio.removeAttribute("src");
    });
    answerAudio.current?.pause();
    setPlaying(false);
    setCurrentTime(0);
    setDuration(0);
  }

  async function handleUpload(file: File): Promise<void> {
    stopAudio();
    setBusy(true);
    setPhase("Opening and indexing your PDF…");
    setMessages([]);
    try {
      const uploaded = await uploadPdf(file);
      setDocumentInfo(uploaded);
      localStorage.setItem("conversiq-document-id", uploaded.doc_id);
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
    if (!documentInfo || busy || recording) return;
    stopAudio();
    cursor.current = position;
    desiredPlayback.current = true;
    const requestId = generation.current;
    void playSegment(documentInfo.doc_id, position, requestId);
  }

  function togglePlayback(): void {
    if (!documentInfo || busy || recording) return;
    const audio = narration.current;
    if (desiredPlayback.current) {
      desiredPlayback.current = false;
      audio.pause();
      setPlaying(false);
      setPhase("Narration paused. Your place is saved.");
    } else if (audio.src && !audio.ended) {
      desiredPlayback.current = true;
      void audio.play().then(() => {
        setPlaying(true);
        setPhase(`Narrating page ${cursor.current.page}`);
      }).catch((error: Error) => setPhase(error.message));
    } else {
      startAt(cursor.current);
    }
  }

  function resumeNarration(): void {
    if (!narration.current.src) return;
    desiredPlayback.current = true;
    void narration.current.play().then(() => {
      setPlaying(true);
      setPhase(`Narrating page ${cursor.current.page}`);
    }).catch((error: Error) => setPhase(error.message));
  }

  async function ask(text: string, shouldResume = desiredPlayback.current): Promise<void> {
    if (!documentInfo || !text.trim() || busy) return;
    narration.current.pause();
    desiredPlayback.current = false;
    setPlaying(false);
    setBusy(true);
    setQuestion("");
    setTab("chat");
    setMessages((items) => [...items, { role: "user", text: text.trim(), pages: [] }]);
    setPhase("Finding an answer in your document…");
    try {
      const result = await askDocument(documentInfo.doc_id, text.trim());
      setMessages((items) => [...items, {
        role: "ai", text: result.answer, pages: result.relevant_pages,
      }]);
      setPhase("Answer ready.");
      try {
        const audioUrl = await getAnswerAudio(result.answer);
        const spoken = new Audio(audioUrl);
        answerAudio.current = spoken;
        spoken.onended = () => {
          answerAudio.current = null;
          if (shouldResume) resumeNarration();
        };
        await spoken.play();
        setPhase("Speaking answer…");
      } catch {
        setPhase("Answer shown. Spoken audio is unavailable.");
        if (shouldResume) resumeNarration();
      }
    } catch (error) {
      setPhase(error instanceof Error ? error.message : "Could not answer the question.");
      if (shouldResume) resumeNarration();
    } finally {
      setBusy(false);
    }
  }

  async function toggleMicrophone(): Promise<void> {
    if (recording && recorder.current) {
      recorder.current.stop();
      recorder.current = null;
      setRecording(false);
      return;
    }
    if (!documentInfo || busy) return;
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
      const chunks: Blob[] = [];
      activeRecorder.ondataavailable = (event) => {
        if (event.data.size) chunks.push(event.data);
      };
      activeRecorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        setPhase("Transcribing your question…");
        void transcribeRecording(new Blob(chunks, { type: mime }), mime.split("/")[1])
          .then((transcript) => {
            if (!transcript.trim()) throw new Error("No speech was recognized. Try again.");
            return ask(transcript, resumeAfterRecording.current);
          })
          .catch((error: Error) => {
            setPhase(error.message);
            if (resumeAfterRecording.current) resumeNarration();
          });
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
            <div className="book-content workspace-book">
              {documentInfo ? (
                <>
                  <div className="chapter-kicker">{currentSection ? `SECTION ${String(currentSectionIndex + 1).padStart(2, "0")}` : `PAGE ${pageNumber}`}</div>
                  <h3>{currentSection?.title || documentInfo.filename.replace(/\.pdf$/i, "")}</h3>
                  <div className="book-rule">✦</div>
                  {sentences.length ? (
                    <div className="workspace-page-text" aria-label={`Extracted text from page ${pageNumber}`}>
                      {sentences.map((sentence) => (
                        <span key={sentence.start} className={`reader-sentence ${highlightedSentenceStart === sentence.start ? "selected" : ""}`}
                          role="button" tabIndex={0} title="Narrate from this sentence"
                          onClick={() => startAt({ page: pageNumber, offset: sentence.start })}
                          onKeyDown={(event) => {
                            if (event.key === "Enter" || event.key === " ") {
                              event.preventDefault();
                              startAt({ page: pageNumber, offset: sentence.start });
                            }
                          }}>{sentence.text}</span>
                      ))}
                    </div>
                  ) : <p className="workspace-hint">No extractable text on this page.</p>}
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
            <div className="narration-status" role="status"><span className={`status-dot ${playing ? "pulse" : ""}`} />{phase}<span>Click a sentence to start there</span></div>
          </main>

          <aside className={`conversation-panel ${tab === "chat" ? "mobile-visible" : ""}`}>
            <div className="panel-heading"><span><AudioLines size={16} /> CONVERSATION</span><span className="context-pill">Context on</span></div>
            <div className="chat-messages" ref={chat}>
              <div className="chat-start">{documentInfo ? "Ask about anything in this document." : "Your conversation begins after upload."}</div>
              {messages.map((message, index) => (
                <div className={`message ${message.role}`} key={index}>
                  <div className="message-author">{message.role === "ai" ? <><span className="mini-logo"><AudioLines size={12} /></span> ConversIQ</> : <><span className="avatar">Y</span> You</>}</div>
                  <p>{message.text}</p>
                  {message.role === "ai" && <div className="source-row">
                    {message.pages.length > 0 && <button className="source-chip" onClick={() => setSourcePages(message.pages)}><BookOpen size={12} /> Pages {message.pages[0]}–{message.pages.at(-1)}</button>}
                    <span><Check size={12} /> {message.pages.length ? "Grounded" : "No matching source"}</span>
                  </div>}
                </div>
              ))}
              {busy && documentInfo && <div className="processing"><AudioLines size={17} /> Working with your document…</div>}
            </div>
            <form className="chat-input" onSubmit={(event) => { event.preventDefault(); void ask(question); }}>
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
              <small>{documentInfo ? `Page ${pageNumber} · Piper narration` : "Upload a PDF to begin"}</small></div>
          </div>
          <div className="playback">
            <div className="playback-buttons">
              <button className="icon-button" aria-label="Previous page" disabled={!documentInfo || pageNumber <= 1} onClick={() => goToPage(pageNumber - 1)}><SkipBack size={17} /></button>
              <button className="play-button" aria-label={playing ? "Pause narration" : "Play narration"} disabled={!documentInfo || busy || recording} onClick={togglePlayback}>
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
          <button className={`ask-button ${recording ? "listening" : ""}`} disabled={!documentInfo || (busy && !recording)} onClick={() => void toggleMicrophone()}>
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
