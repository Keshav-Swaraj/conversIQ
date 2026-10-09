import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  ArrowDown,
  ArrowRight,
  ArrowUpRight,
  AudioLines,
  BookOpen,
  Check,
  ChevronRight,
  FileText,
  GitBranch,
  Layers,
  Mic,
  Network,
  Play,
  ShieldCheck,
  Sparkles,
  Upload,
} from "lucide-react";
import { Counter, Label, Logo, Navbar, Reveal, Wave } from "./components";
import ProductDemo from "./ProductDemo";
function Hero() {
  return (
    <section className="hero shell" id="product">
      <div className="hero-grain" />
      <Reveal>
        <div className="announcement">
          <span className="status-dot" /> INTRODUCING CONVERSIQ{" "}
          <span className="announcement-divider" /> A new way to read{" "}
          <ArrowUpRight size={12} />
        </div>
        <h1>
          Your documents.
          <br />
          <span>Now, a conversation.</span>
        </h1>
        <p className="hero-description">
          Less scrolling. More understanding.
          <br />
          Listen, ask, and explore your documents — simply by talking.
        </p>
        <div className="hero-actions">
          <a className="button primary" href="/workspace">
            Try ConversIQ <ArrowUpRight size={17} />
          </a>
          <a className="button secondary" href="#how-it-works">
            <span className="circle-play">
              <Play size={10} fill="currentColor" />
            </span>{" "}
            See how it works
          </a>
        </div>
        <div className="hero-meta">
          <span>
            <Check size={12} /> No signup needed
          </span>
          <span>
            <Check size={12} /> Built around your curiosity
          </span>
        </div>
      </Reveal>
      <div className="hero-signal">
        <div className="signal-line" />
        <div className="signal-doc">
          <FileText size={17} />
          <span>Your document</span>
        </div>
        <span className="signal-path" />
        <div className="signal-tree">
          <GitBranch size={19} />
        </div>
        <span className="signal-path" />
        <Wave active />
        <span className="signal-caption">A whole new wavelength.</span>
        <div className="signal-line" />
      </div>
      <div className="demo-intro">
        <span>
          <span className="status-dot" /> YOUR NEXT CHAPTER STARTS HERE
        </span>
        <span>
          Go ahead. Interrupt the story. <ArrowDown size={13} />
        </span>
      </div>
      <ProductDemo />
      <div className="demo-under">
        <span>
          <ShieldCheck size={13} /> Your document. Your context. Every answer
          grounded.
        </span>
        <span>Frontend concept · simulated interactions</span>
      </div>
      <div className="built-with">
        <span>INTELLIGENCE, BEAUTIFULLY CONNECTED</span>
        <div>
          <span>
            <GitBranch /> PageIndex
          </span>
          <span>
            <Network /> LangChain
          </span>
          <span>
            <Layers /> LangGraph
          </span>
          <span>
            <AudioLines /> Voice AI
          </span>
          <span>
            <ShieldCheck /> Vectorless retrieval
          </span>
        </div>
      </div>
    </section>
  );
}
function HowItWorks() {
  return (
    <section className="section shell" id="how-it-works">
      <Reveal>
        <div className="section-heading">
          <div>
            <Label>EFFORTLESS BY DESIGN</Label>
            <h2>From a file to a flow.</h2>
          </div>
          <p>
            No new habits to learn.
            <br />
            Just bring a document and your curiosity.
          </p>
        </div>
        <div className="steps">
          {[
            {
              icon: Upload,
              title: "Drop it in.",
              text: "Start with a PDF, a paper, or a book. Give your curiosity something to work with.",
            },
            {
              icon: GitBranch,
              title: "Find the structure.",
              text: "PageIndex maps chapters, sections, and pages. The whole picture, connected.",
            },
            {
              icon: Mic,
              title: "Make it a conversation.",
              text: "Listen to the story. Ask a question. Get an answer with the pages to back it up.",
            },
            {
              icon: Play,
              title: "Pick up your thread.",
              text: "Return to the exact moment you left. Your place — and your context — stay with you.",
            },
          ].map((s, i) => (
            <div className="step" key={s.title}>
              <div className="step-top">
                <span className="step-icon">
                  <s.icon size={21} />
                </span>
                <span>0{i + 1}</span>
                {i < 3 && <ArrowRight size={15} />}
              </div>
              <h3>{s.title}</h3>
              <p>{s.text}</p>
            </div>
          ))}
        </div>
      </Reveal>
    </section>
  );
}
function Technology() {
  return (
    <section className="technology section" id="technology">
      <div className="shell">
        <Reveal className="tech-layout">
          <div>
            <Label>A DIFFERENT KIND OF INTELLIGENCE</Label>
            <h2>
              Understands the structure.
              <br />
              <span>Not just the similarity.</span>
            </h2>
            <p className="section-copy">
              RAG without a vector database. ConversIQ follows your document’s
              natural hierarchy to reason its way to the right pages.
            </p>
            <div className="tech-promises">
              <span>
                <Check /> No embeddings
              </span>
              <span>
                <Check /> No vector database
              </span>
              <span>
                <Check /> Structure-aware retrieval
              </span>
            </div>
            <a className="text-link" href="#intelligence">
              Explore the architecture <ArrowUpRight size={15} />
            </a>
          </div>
          <div className="comparison">
            <div className="traditional">
              <div className="diagram-label">
                TRADITIONAL RAG <span>Similarity-based</span>
              </div>
              <div className="pipeline">
                <FileText size={22} />
                <ArrowRight />
                <span>Chunks</span>
                <ArrowRight />
                <span>Embeddings</span>
              </div>
              <div className="traditional-bottom">
                <span>Vector database</span>
                <ArrowRight size={14} />
                <span>Similarity search</span>
              </div>
            </div>
            <div className="conversiq-path">
              <div className="diagram-label">
                <span className="violet">
                  <AudioLines size={14} /> THE CONVERSIQ WAY
                </span>
                <span>Reasoning-based</span>
              </div>
              <div className="structure-diagram">
                <div className="diagram-node root-node">
                  <FileText size={15} /> Your document
                </div>
                <div className="tree-connect" />
                <div className="diagram-branches">
                  <span>Chapter 01</span>
                  <span className="active-node">Chapter 02</span>
                  <span>Chapter 03</span>
                </div>
                <div className="tree-connect short" />
                <div className="diagram-node evidence-node">
                  <BookOpen size={14} /> Exact source pages <span>32–44</span>
                </div>
              </div>
              <div className="diagram-note">
                <Sparkles size={13} /> Follow meaning. Find evidence.
              </div>
            </div>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
function Conversation() {
  const [selected, setSelected] = useState(0);
  const questions = [
    "What is a Portkey?",
    "Why was it important?",
    "Who used it?",
  ];
  const answers = [
    "An everyday object enchanted to transport people to a specific place. Even an old boot can take you somewhere extraordinary.",
    "It gets the group to the Quidditch World Cup at a scheduled time. Here, “it” still refers to the Portkey — no need to explain again.",
    "Harry, Hermione, Ron, Ginny, Fred, George, Arthur, Amos, and Cedric travel together using the boot.",
  ];
  return (
    <section className="section shell conversation-feature">
      <Reveal className="feature-layout">
        <div className="context-demo">
          <div className="context-top">
            <span className="mini-logo">
              <AudioLines size={15} />
            </span>
            <span>One conversation. Connected thoughts.</span>
            <span className="context-pill">
              <span className="status-dot" /> Context maintained
            </span>
          </div>
          {questions.map((q, i) => (
            <div
              className={`context-turn ${selected === i ? "active" : ""}`}
              key={q}
            >
              <button onClick={() => setSelected(i)}>
                <span>0{i + 1}</span>
                {q}
                <ChevronRight size={15} />
              </button>
              <AnimatePresence initial={false}>
                {selected === i && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                  >
                    <p>{answers[i]}</p>
                    <span className="context-source">
                      <BookOpen size={11} /> Chapter 6 · The Portkey
                    </span>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          ))}
        </div>
        <div>
          <Label>IT FOLLOWS YOUR THINKING</Label>
          <h2>
            Follow-up questions.
            <br />
            <span>Without starting over.</span>
          </h2>
          <p className="section-copy">
            “Why?” “Who was that?” “Tell me more.”
            <br />A real conversation doesn’t reset after every sentence.
            Neither should your documents.
          </p>
          <span className="small-note">
            <GitBranch size={15} /> Context travels with the conversation.
          </span>
        </div>
      </Reveal>
    </section>
  );
}
function Intelligence() {
  const [expanded, setExpanded] = useState<number[]>([1]),
    [selected, setSelected] = useState(1);
  return (
    <section className="section shell" id="intelligence">
      <Reveal className="feature-layout intelligence">
        <div>
          <Label>SEE THE WHOLE PICTURE</Label>
          <h2>
            Every chapter.
            <br />
            In the right context.
          </h2>
          <p className="section-copy">
            A document is more than a pile of text. Explore the relationships
            between its ideas, with a navigable map that keeps every answer
            close to its source.
          </p>
          <div className="node-detail">
            <BookOpen size={18} />
            <div>
              Pages {selected * 16 + 16}–{selected * 16 + 28}
              <small>3 child sections · Relevant to current query</small>
            </div>
            <span className="context-pill">Source located</span>
          </div>
        </div>
        <div className="document-tree">
          <div className="tree-header">
            <span>
              <GitBranch size={15} /> PAGEINDEX EXPLORER
            </span>
            <span>Live structure</span>
          </div>
          <div className="tree-document">
            <FileText size={19} /> Harry Potter and the Goblet of Fire{" "}
            <span>636 p.</span>
          </div>
          {["The Riddle House", "The Scar", "The Invitation"].map((name, i) => (
            <div
              className={`tree-chapter ${selected === i ? "active" : ""}`}
              key={name}
            >
              <button
                onClick={() => {
                  setSelected(i);
                  setExpanded((e) =>
                    e.includes(i) ? e.filter((n) => n !== i) : [...e, i],
                  );
                }}
                aria-expanded={expanded.includes(i)}
              >
                <ChevronRight
                  className={expanded.includes(i) ? "rotated" : ""}
                  size={14}
                />
                <BookOpen size={15} />
                <span>
                  0{i + 1} &nbsp;{name}
                </span>
                <span>
                  {i * 16 + 16}–{i * 16 + 28}
                </span>
              </button>
              <AnimatePresence initial={false}>
                {expanded.includes(i) && (
                  <motion.div
                    className="tree-children"
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                  >
                    {[
                      "Setting & context",
                      "Key events",
                      "Characters & connections",
                    ].map((s, j) => (
                      <button onClick={() => setSelected(i)} key={s}>
                        <span className="tree-leaf" />
                        {s}
                        <span>p. {i * 16 + 16 + j * 4}</span>
                      </button>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          ))}
          <div className="tree-footer">
            <Check size={13} /> Document structure mapped{" "}
            <span>8 chapters · 24 sections</span>
          </div>
        </div>
      </Reveal>
    </section>
  );
}
function Voice() {
  const [phase, setPhase] = useState(-1);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  useEffect(
    () => () => {
      if (timer.current) clearInterval(timer.current);
    },
    [],
  );
  function run() {
    if (phase >= 0) return;
    setPhase(0);
    let step = 0;
    timer.current = setInterval(() => {
      step++;
      if (step > 3) {
        clearInterval(timer.current!);
        setPhase(-1);
      } else setPhase(step);
    }, 1500);
  }
  const captions = [
    "“Wait — who is Frank Bryce?”",
    "Finding the passage in chapter one…",
    "“Frank Bryce is the gardener who tends the Riddle House.”",
    "Returning to your exact place in the story…",
  ];
  return (
    <section className="voice-section shell" id="voice">
      <Reveal>
        <Label>MADE FOR THE WAY YOU THINK</Label>
        <h2>
          Don’t type.
          <br />
          <span>Just interrupt.</span>
        </h2>
        <p>Good questions don’t wait for the end of a chapter.</p>
        <div className={`voice-stage ${phase >= 0 ? "running" : ""}`}>
          <Wave active={phase >= 0} />
          <button
            onClick={run}
            disabled={phase >= 0}
            className="voice-orb"
            aria-label="Simulate interrupting narration"
          >
            <Mic size={34} />
          </button>
          <Wave active={phase >= 0} />
          <div className="orb-ring ring-one" />
          <div className="orb-ring ring-two" />
        </div>
        <div className="voice-caption" aria-live="polite">
          {phase < 0 ? (
            <>
              <span className="status-dot" /> Click the mic. See how an
              interruption feels.
            </>
          ) : (
            captions[phase]
          )}
        </div>
        <div className="voice-timeline">
          {["Listening", "Understanding", "Answering", "Resuming"].map(
            (s, i) => (
              <div key={s} className={phase === i ? "active" : ""}>
                <span>
                  {phase > i ? (
                    <Check size={12} />
                  ) : (
                    String(i + 1).padStart(2, "0")
                  )}
                </span>
                {s}
                {i < 3 && <ArrowRight size={14} />}
              </div>
            ),
          )}
        </div>
        <div className="voice-progress">
          <span>NARRATION</span>
          <div>
            <i style={{ width: phase === 3 ? "48%" : "42%" }} />
          </div>
          <span>
            {phase >= 0 && phase < 3
              ? "Place saved"
              : phase === 3
                ? "Resuming…"
                : "Ready to explore"}
          </span>
        </div>
      </Reveal>
    </section>
  );
}
function Metrics() {
  return (
    <section className="metrics-section shell">
      <Reveal>
        <div className="trust-heading">
          <ShieldCheck size={19} />
          <span>Built for answers you can trace.</span>
          <span>Illustrative targets · not measured benchmarks</span>
        </div>
        <div className="metrics">
          {[
            ["~1", "s", "Retrieval reasoning"],
            ["~2", "s", "Answer generation"],
            ["100", "%", "Source-grounded by design"],
            ["0", "", "Vector databases"],
          ].map(([number, unit, label]) => (
            <div key={label}>
              <div>
                <Counter value={number} />
                <span className="metric-unit">{unit}</span>
              </div>
              <p>{label}</p>
            </div>
          ))}
        </div>
        <div className="trust-evidence">
          <span>Every answer has a place to point to.</span>
          <span>
            <FileText size={13} /> Your document <ArrowRight size={13} />
            <BookOpen size={13} /> Exact pages <ArrowRight size={13} />
            <ShieldCheck size={13} /> Grounded answer
          </span>
        </div>
      </Reveal>
    </section>
  );
}
export default function App() {
  return (
    <>
      <Navbar />
      <Hero />
      <HowItWorks />
      <Technology />
      <Conversation />
      <Intelligence />
      <Voice />
      <Metrics />
      <section className="final-cta shell">
        <Reveal>
          <div className="closing-mark">
            <AudioLines size={31} />
          </div>
          <Label>YOUR DOCUMENTS HAVE MORE TO SAY</Label>
          <h2>
            Turn the page.
            <br />
            <span>Start a conversation.</span>
          </h2>
          <p>Voice-first. Vectorless. A little more human.</p>
          <div className="hero-actions">
          <a href="/workspace" className="button primary">
            Experience ConversIQ <ArrowUpRight size={17} />
            </a>
            <a href="#technology" className="text-link">
              Explore the technology <ArrowRight size={15} />
            </a>
          </div>
        </Reveal>
      </section>
      <footer className="shell">
        <div className="footer-top">
          <div>
            <Logo />
            <p>Talk to your documents.</p>
          </div>
          <div className="footer-links">
            <a href="#product">Product</a>
            <a href="#intelligence">Architecture</a>
            <a href="#technology">Technology</a>
            <a
              href="https://github.com/topics/vectorless-rag"
              target="_blank"
              rel="noreferrer"
            >
              GitHub <ArrowUpRight size={12} />
            </a>
          </div>
        </div>
        <div className="footer-bottom">
          <span>
            © {new Date().getFullYear()} ConversIQ. A new way to understand.
          </span>
          <span>
            <span className="status-dot" /> Designed around curiosity.
          </span>
        </div>
      </footer>
    </>
  );
}
