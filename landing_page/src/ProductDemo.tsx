import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  AudioLines,
  BookOpen,
  Check,
  ChevronDown,
  FileText,
  Headphones,
  Mic,
  Pause,
  Play,
  Plus,
  Send,
  ShieldCheck,
  SkipBack,
  SkipForward,
  X,
} from "lucide-react";
import { Wave } from "./components";
const chapters = [
  "The Riddle House",
  "The Scar",
  "The Invitation",
  "Back to the Burrow",
  "Weasleys’ Wizard Wheezes",
  "The Portkey",
  "Bagman and Crouch",
  "The Quidditch World Cup",
];
const summaries = [
  [
    "Above the village of Little Hangleton stands the Riddle House, a once-grand manor that has fallen into disrepair. Its dark windows and overgrown grounds have become part of the village’s uneasy history.",
    "Years earlier, three members of the Riddle family were found dead inside. No clear explanation emerged, leaving the villagers to fill the silence with suspicion and stories.",
    "Frank Bryce, the elderly gardener, still tends the grounds. When he notices a light in the supposedly empty house, he goes to investigate — unaware of what awaits him.",
  ],
  [
    "Harry wakes with a pain in his scar and a troubling memory of a dream. Far from the wizarding world, he wonders whether the feeling is a warning.",
    "He decides to write to Sirius, someone who might understand the significance of what he has seen.",
  ],
  [
    "An invitation from the Weasley family offers Harry the chance to attend the Quidditch World Cup.",
    "The prospect of leaving Privet Drive makes the remaining days of summer feel brighter.",
  ],
  [
    "The Weasleys arrive to collect Harry, bringing magical confusion to an ordinary house.",
    "Soon Harry is back at the Burrow, surrounded by familiar friends and preparations for the World Cup.",
  ],
  [
    "Fred and George are developing magical jokes and inventions, even as their mother has other plans for their future.",
    "Their experiments suggest the beginnings of an ambitious business.",
  ],
  [
    "The group climbs a hill before dawn to find a Portkey: an ordinary-looking object enchanted to transport its users.",
    "By touching the old boot at the appointed moment, Harry and the others travel together to the World Cup campsite.",
  ],
  [
    "At the campsite, Harry meets two Ministry officials with very different personalities.",
    "Their work reveals the immense organization behind an international magical gathering.",
  ],
  [
    "The stadium comes alive as the teams take the field for the Quidditch World Cup.",
    "For Harry, the match opens a window into the scale and excitement of the wider wizarding world.",
  ],
];
type Message = { role: "user" | "ai"; text: string; chapter: number };
type Phase = "idle" | "Listening" | "Understanding" | "Answering" | "Resuming";
export default function ProductDemo() {
  const [chapter, setChapter] = useState(0),
    [playing, setPlaying] = useState(false),
    [progress, setProgress] = useState(24),
    [phase, setPhase] = useState<Phase>("idle"),
    [question, setQuestion] = useState(""),
    [speed, setSpeed] = useState(1),
    [tab, setTab] = useState("read"),
    [source, setSource] = useState<number | null>(null),
    [upload, setUpload] = useState(""),
    [messages, setMessages] = useState<Message[]>([
      {
        role: "user",
        text: "Why is the Riddle House considered dangerous?",
        chapter: 0,
      },
      {
        role: "ai",
        text: "The house is connected to the unexplained deaths of the Riddle family. Its isolation and unsettling history have made it a place the villagers avoid.",
        chapter: 0,
      },
    ]);
  const timers = useRef<ReturnType<typeof setTimeout>[]>([]),
    resume = useRef(false),
    file = useRef<HTMLInputElement>(null),
    chat = useRef<HTMLDivElement>(null);
  useEffect(() => () => timers.current.forEach(clearTimeout), []);
  useEffect(() => {
    if (source === null && !upload) return;
    const previous = document.activeElement as HTMLElement | null;
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setSource(null);
        setUpload("");
      }
      if (e.key === "Tab") {
        const items = document.querySelectorAll<HTMLButtonElement>(
          ".source-modal button",
        );
        const first = items[0],
          last = items[items.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener("keydown", key);
    return () => {
      document.removeEventListener("keydown", key);
      previous?.focus();
    };
  }, [source, upload]);
  useEffect(() => {
    if (!playing || phase !== "idle") return;
    const timer = setInterval(
      () =>
        setProgress((p) => {
          if (p >= 100) {
            setPlaying(false);
            return 100;
          }
          return Math.min(100, p + 0.22 * speed);
        }),
      250,
    );
    return () => clearInterval(timer);
  }, [playing, phase, speed]);
  useEffect(() => {
    if (chat.current) chat.current.scrollTop = chat.current.scrollHeight;
  }, [messages, phase]);
  function selectChapter(i: number) {
    setChapter(i);
    setProgress(0);
    setSource(null);
  }
  function ask(text: string, voice = false) {
    if (phase !== "idle" || !text.trim()) return;
    resume.current = playing;
    setPlaying(false);
    setQuestion("");
    setTab("chat");
    setPhase(voice ? "Listening" : "Understanding");
    const current = chapter;
    const answer = /portkey|\bboot\b/i.test(text)
      ? "A Portkey is an enchanted object that transports anyone touching it to a predetermined destination. In chapter 6, the group uses an old boot to reach the World Cup campsite."
      : /danger|riddle|house/i.test(text)
        ? "The Riddle House is associated with the unexplained deaths of the Riddle family. Frank Bryce remains its gardener, and investigates a light in the house despite its abandoned appearance."
        : `In “${chapters[current]},” ${summaries[current][0].charAt(0).toLowerCase() + summaries[current][0].slice(1)} This sample response illustrates how an answer stays connected to the selected chapter.`;
    const answerChapter = /portkey|\bboot\b/i.test(text)
      ? 5
      : /danger|riddle|house/i.test(text)
        ? 0
        : current;
    const later = (fn: () => void, ms: number) =>
      timers.current.push(setTimeout(fn, ms));
    later(
      () => {
        setMessages((m) => [...m, { role: "user", text, chapter: current }]);
        setPhase("Understanding");
      },
      voice ? 1100 : 100,
    );
    later(() => setPhase("Answering"), voice ? 2100 : 900);
    later(
      () => {
        setMessages((m) => [
          ...m,
          { role: "ai", text: answer, chapter: answerChapter },
        ]);
        setPhase(resume.current ? "Resuming" : "Answering");
      },
      voice ? 3300 : 2100,
    );
    later(
      () => {
        setPhase("idle");
        setPlaying(resume.current);
      },
      voice ? 4400 : 3200,
    );
  }
  return (
    <div className="app-window" id="demo">
      <div className="app-topbar">
        <div className="window-dots">
          <i />
          <i />
          <i />
        </div>
        <span>
          <AudioLines size={14} /> ConversIQ{" "}
          <span className="top-divider">/</span> Workspace
        </span>
        <span className="demo-badge">
          <span /> Interactive demo
        </span>
      </div>
      <div className="mobile-tabs">
        {[
          ["chapters", "Chapters"],
          ["read", "Document"],
          ["chat", "Conversation"],
        ].map(([id, name]) => (
          <button
            className={tab === id ? "selected" : ""}
            key={id}
            onClick={() => setTab(id)}
          >
            {name}
          </button>
        ))}
      </div>
      <div className="app-body">
        <aside
          className={`document-sidebar ${tab === "chapters" ? "mobile-visible" : ""}`}
        >
          <div className="panel-heading">
            YOUR LIBRARY{" "}
            <button
              className="icon-button"
              aria-label="Upload a sample document"
              onClick={() => file.current?.click()}
            >
              <Plus size={15} />
            </button>
          </div>
          <input
            type="file"
            accept=".pdf,.doc,.docx,.txt"
            hidden
            ref={file}
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f)
                setUpload(
                  `${f.name} selected. This demo uses the sample book; no file is uploaded or processed.`,
                );
              e.target.value = "";
            }}
          />
          <div className="document-name">
            <span className="pdf-icon">
              <FileText size={19} />
            </span>
            <div>
              Harry Potter and the
              <br />
              Goblet of Fire <small>PDF · 636 pages</small>
            </div>
            <ChevronDown size={13} />
          </div>
          <div className="sidebar-section">
            CONTENTS <span>08 chapters</span>
          </div>
          <div className="chapter-list">
            {chapters.map((name, i) => (
              <button
                disabled={phase !== "idle"}
                key={name}
                className={chapter === i ? "selected" : ""}
                onClick={() => {
                  selectChapter(i);
                  setTab("read");
                }}
              >
                <span>{String(i + 1).padStart(2, "0")}</span>
                {name}
                {chapter === i && <AudioLines size={13} />}
              </button>
            ))}
          </div>
          <div className="indexed">
            <ShieldCheck size={15} />
            <div>
              Structure understood<small>Indexed with PageIndex</small>
            </div>
            <span className="status-dot" />
          </div>
        </aside>
        <main
          className={`document-reader ${tab === "read" ? "mobile-visible" : ""}`}
        >
          <div className="reader-toolbar">
            <span>
              <BookOpen size={14} /> Reading view
            </span>
            <span>Chapter {chapter + 1} of 8</span>
          </div>
          <div className="book-content">
            <div className="chapter-kicker">
              CHAPTER {String(chapter + 1).padStart(2, "0")}
            </div>
            <h3>{chapters[chapter]}</h3>
            <div className="book-rule">✦</div>
            {summaries[chapter].map((text, i) => (
              <p
                key={text}
                className={
                  Math.min(
                    summaries[chapter].length - 1,
                    Math.floor((progress / 100) * summaries[chapter].length),
                  ) === i
                    ? "narrated"
                    : ""
                }
              >
                {text}
                {i === 0 && playing && <span className="reading-cursor" />}
              </p>
            ))}
            <div className="reader-page">
              Original demo summary · not the book text
              <span>
                Page {7 + chapter * 16} <span className="muted">/ 636</span>
              </span>
            </div>
          </div>
          <div className="narration-status">
            <span className={`status-dot ${playing ? "pulse" : ""}`} />
            {phase !== "idle"
              ? `${phase}…`
              : playing
                ? "Narrating your document"
                : "Ready when you are"}
            <span>Your place is always saved</span>
          </div>
        </main>
        <aside
          className={`conversation-panel ${tab === "chat" ? "mobile-visible" : ""}`}
        >
          <div className="panel-heading">
            <span>
              <AudioLines size={15} /> CONVERSATION
            </span>
            <span className="context-pill">Context on</span>
          </div>
          <div className="chat-messages" ref={chat}>
            <div className="chat-start">
              A little curiosity goes a long way.
            </div>
            <AnimatePresence initial={false}>
              {messages.map((m, i) => (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`message ${m.role}`}
                  key={i}
                >
                  <div className="message-author">
                    {m.role === "ai" ? (
                      <>
                        <span className="mini-logo">
                          <AudioLines size={12} />
                        </span>{" "}
                        ConversIQ
                      </>
                    ) : (
                      <>
                        <span className="avatar">Y</span>You
                      </>
                    )}
                  </div>
                  <p>{m.text}</p>
                  {m.role === "ai" && (
                    <div className="source-row">
                      <button
                        className="source-chip"
                        onClick={() => setSource(m.chapter)}
                      >
                        <BookOpen size={11} /> Pages {7 + m.chapter * 16}–
                        {22 + m.chapter * 16}
                      </button>
                      <span>
                        <Check size={11} /> Grounded
                      </span>
                    </div>
                  )}
                </motion.div>
              ))}
            </AnimatePresence>
            {phase !== "idle" && (
              <div role="status" className="processing">
                <Wave active />
                {phase}…
              </div>
            )}
          </div>
          <form
            className="chat-input"
            onSubmit={(e) => {
              e.preventDefault();
              ask(question);
            }}
          >
            <input
              aria-label="Ask about the document"
              placeholder="Ask about this document…"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              disabled={phase !== "idle"}
            />
            <button
              disabled={!question.trim() || phase !== "idle"}
              aria-label="Send question"
            >
              <Send size={15} />
            </button>
          </form>
          <div className="chat-footnote">
            <ShieldCheck size={11} /> Answers stay within your document.
          </div>
        </aside>
      </div>
      <div className="player">
        <div className="player-title">
          <span className="headphones">
            <Headphones size={19} />
          </span>
          <div>
            {chapters[chapter]}
            <small>Chapter {chapter + 1} · Simulated narration</small>
          </div>
        </div>
        <div className="playback">
          <div className="playback-buttons">
            <button
              className="icon-button"
              aria-label="Previous chapter"
              disabled={chapter === 0 || phase !== "idle"}
              onClick={() => selectChapter(chapter - 1)}
            >
              <SkipBack size={16} />
            </button>
            <button
              className="play-button"
              aria-label={playing ? "Pause narration" : "Play narration"}
              disabled={phase !== "idle"}
              onClick={() => {
                if (progress >= 100) setProgress(0);
                setPlaying(!playing);
              }}
            >
              {playing ? (
                <Pause size={17} fill="currentColor" />
              ) : (
                <Play size={17} fill="currentColor" />
              )}
            </button>
            <button
              className="icon-button"
              aria-label="Next chapter"
              disabled={chapter === 7 || phase !== "idle"}
              onClick={() => selectChapter(chapter + 1)}
            >
              <SkipForward size={16} />
            </button>
            <button
              className="speed-button"
              aria-label="Change narration speed"
              onClick={() =>
                setSpeed((s) => (s === 1 ? 1.25 : s === 1.25 ? 1.5 : 1))
              }
            >
              {speed}×
            </button>
          </div>
          <div className="progress-row">
            <span>
              {Math.floor((progress * 7.42) / 60)}:
              {String(Math.floor((progress * 7.42) % 60)).padStart(2, "0")}
            </span>
            <input
              aria-label="Narration progress"
              type="range"
              min="0"
              max="100"
              step=".1"
              value={progress}
              disabled={phase !== "idle"}
              onChange={(e) => setProgress(Number(e.target.value))}
              style={{
                background: `linear-gradient(to right,#a593f5 ${progress}%,#30303a ${progress}%)`,
              }}
            />
            <span>12:22</span>
          </div>
        </div>
        <button
          className={`ask-button ${phase !== "idle" ? "listening" : ""}`}
          disabled={phase !== "idle"}
          onClick={() => ask("What happens at the Riddle House?", true)}
        >
          <Mic size={17} />
          {phase === "idle" ? "Ask ConversIQ" : `${phase}…`}
          <span>↵</span>
        </button>
      </div>
      {(source !== null || upload) && (
        <div
          className="modal-backdrop"
          onClick={() => {
            setSource(null);
            setUpload("");
          }}
        >
          <div
            className="source-modal"
            role="dialog"
            aria-modal="true"
            aria-label={upload ? "Sample upload" : "Document source"}
            onClick={(e) => e.stopPropagation()}
          >
            <button
              autoFocus
              className="icon-button modal-close"
              aria-label="Close dialog"
              onClick={() => {
                setSource(null);
                setUpload("");
              }}
            >
              <X />
            </button>
            <BookOpen className="violet" />
            <h3>
              {upload ? "Document selected" : `Source · ${chapters[source!]}`}
            </h3>
            <p>{upload || summaries[source!].join(" ")}</p>
            <small>
              {upload
                ? "Frontend demo · files remain on your device"
                : `Sample evidence · Pages ${7 + source! * 16}–${22 + source! * 16}`}
            </small>
            <button
              className="button primary"
              onClick={() => {
                if (source !== null) {
                  selectChapter(source);
                  setTab("read");
                }
                setSource(null);
                setUpload("");
              }}
            >
              {upload ? "Continue exploring" : "Open chapter"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
