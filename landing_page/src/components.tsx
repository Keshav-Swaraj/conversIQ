import { animate, motion, useInView, useReducedMotion } from "framer-motion";
import { AudioLines, ArrowUpRight, Menu, X } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
export function Counter({ value }: { value: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const visible = useInView(ref, { once: true });
  const reduced = useReducedMotion();
  const target = Number(value.replace("~", ""));
  const [count, setCount] = useState(0);
  useEffect(() => {
    if (!visible || reduced) return;
    const control = animate(0, target, {
      duration: 1.2,
      ease: "easeOut",
      onUpdate: (v) => setCount(Math.round(v)),
    });
    return () => control.stop();
  }, [visible, reduced, target]);
  return (
    <span ref={ref}>
      {value.startsWith("~") ? "~" : ""}
      {reduced ? target : count}
    </span>
  );
}
export function Logo() {
  return (
    <a className="logo" href="#" aria-label="ConversIQ home">
      <span className="logo-icon">
        <AudioLines size={22} />
      </span>
      convers<span>iq</span>
      <span className="logo-dot">®</span>
    </a>
  );
}
export function Wave({
  active = false,
  className = "",
}: {
  active?: boolean;
  className?: string;
}) {
  return (
    <div
      className={`wave ${active ? "active" : ""} ${className}`}
      aria-hidden="true"
    >
      {Array.from({ length: 33 }, (_, i) => (
        <i
          key={i}
          style={{
            height: `${8 + Math.abs(Math.sin(i * 1.9)) * 21 + Math.sin((i / 32) * Math.PI) * 55}%`,
            animationDelay: `${i * 0.037}s`,
          }}
        />
      ))}
    </div>
  );
}
export function Reveal({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  const reduced = useReducedMotion();
  return (
    <motion.div
      className={className}
      initial={reduced ? false : { opacity: 0, y: 22 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.12 }}
      transition={{ duration: 0.6 }}
    >
      {children}
    </motion.div>
  );
}
export function Label({ children }: { children: ReactNode }) {
  return (
    <div className="eyebrow">
      <span />
      {children}
    </div>
  );
}
export function Navbar() {
  const [open, setOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const scroll = () => setScrolled(window.scrollY > 24);
    window.addEventListener("scroll", scroll);
    return () => window.removeEventListener("scroll", scroll);
  }, []);
  return (
    <header className={scrolled ? "scrolled" : ""}>
      <nav className="nav shell">
        <Logo />
        <div className={`nav-links ${open ? "open" : ""}`}>
          {[
            ["Product", "product"],
            ["How it works", "how-it-works"],
            ["Technology", "technology"],
            ["Demo", "demo"],
          ].map(([text, id]) => (
            <a key={id} href={`#${id}`} onClick={() => setOpen(false)}>
              {text}
            </a>
          ))}
        </div>
        <a className="button nav-cta" href="/workspace">
          Open workspace <ArrowUpRight size={15} />
        </a>
        <button
          className="mobile-menu icon-button"
          onClick={() => setOpen(!open)}
          aria-label={open ? "Close navigation" : "Open navigation"}
          aria-expanded={open}
        >
          {open ? <X /> : <Menu />}
        </button>
      </nav>
    </header>
  );
}
