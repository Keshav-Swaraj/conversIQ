"""Local desktop interface for the existing ConversIQ voice workflow."""

from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import sounddevice as sd

from src.session import SessionResult, run_continuous_session


PROJECT_ROOT = Path(__file__).resolve().parent
VAD_PATH = PROJECT_ROOT / "cache" / "vad" / "silero_vad.onnx"
OUTPUT_DIR = PROJECT_ROOT / "cache" / "tts"


class ConversIQWindow:
    """Select a local PDF and run narration on the computer's audio devices."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ConversIQ")
        self.root.geometry("720x520")
        self.pdf_path: Path | None = None
        self._stop = Event()
        self._worker: Thread | None = None
        self._events: Queue[tuple[str, str | SessionResult]] = Queue()
        self._closing = False

        frame = ttk.Frame(root, padding=20)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="ConversIQ", font=("Segoe UI", 20, "bold")).pack(anchor="w")
        ttk.Label(
            frame,
            text="Choose a PDF, start narration, and speak during playback to ask a question.",
            wraplength=650,
        ).pack(anchor="w", pady=(4, 14))

        document_row = ttk.Frame(frame)
        document_row.pack(fill="x")
        self._document_label = ttk.Label(document_row, text="No PDF selected", width=68)
        self._document_label.pack(side="left", fill="x", expand=True)
        self._choose_button = ttk.Button(document_row, text="Choose PDF", command=self.choose_pdf)
        self._choose_button.pack(side="right")

        microphone_row = ttk.Frame(frame)
        microphone_row.pack(fill="x", pady=(12, 0))
        ttk.Label(microphone_row, text="Microphone:").pack(side="left")
        self._input_devices = [
            (index, str(device["name"]))
            for index, device in enumerate(sd.query_devices())
            if device["max_input_channels"] > 0
        ]
        self._microphone = ttk.Combobox(
            microphone_row,
            values=[f"{index}: {name}" for index, name in self._input_devices],
            state="readonly",
        )
        self._microphone.pack(side="left", fill="x", expand=True, padx=(8, 0))
        default_input = sd.default.device[0]
        default_position = next(
            (position for position, (index, _) in enumerate(self._input_devices)
             if index == default_input),
            0,
        )
        if self._input_devices:
            self._microphone.current(default_position)

        controls = ttk.Frame(frame)
        controls.pack(fill="x", pady=14)
        self._start_button = ttk.Button(controls, text="Start narration", command=self.start)
        self._start_button.pack(side="left")
        self._stop_button = ttk.Button(controls, text="Stop", command=self.stop, state="disabled")
        self._stop_button.pack(side="left", padx=10)
        self._status = tk.StringVar(value="Ready. Choose your actual microphone above; speakers are local.")
        ttk.Label(frame, textvariable=self._status, wraplength=650).pack(anchor="w", pady=(0, 10))

        self._log = scrolledtext.ScrolledText(frame, height=15, wrap="word", state="disabled")
        self._log.pack(fill="both", expand=True)
        ttk.Label(
            frame,
            text="Headphones reduce speaker echo, but are not required. Stop finishes the current processing step; Start begins the PDF again.",
            wraplength=650,
        ).pack(anchor="w", pady=(10, 0))

        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self._poll_events)

    def choose_pdf(self) -> None:
        """Select a PDF without copying or modifying the source document."""
        selected = filedialog.askopenfilename(
            parent=self.root, title="Choose a PDF", filetypes=[("PDF documents", "*.pdf")]
        )
        if selected:
            self.pdf_path = Path(selected)
            self._document_label.configure(text=str(self.pdf_path))
            self._status.set("Ready to narrate.")

    def start(self) -> None:
        """Run the existing continuous session outside Tk's event loop."""
        if self._worker is not None and self._worker.is_alive():
            return
        if self.pdf_path is None or not self.pdf_path.is_file():
            messagebox.showerror("Choose a PDF", "Select an existing PDF first.", parent=self.root)
            return
        for required in (VAD_PATH,):
            if not required.is_file():
                messagebox.showerror("Missing local model", f"Required file not found:\n{required}", parent=self.root)
                return
        microphone_position = self._microphone.current()
        if microphone_position < 0:
            messagebox.showerror("Choose a microphone", "Select an input device first.", parent=self.root)
            return
        input_device, microphone_name = self._input_devices[microphone_position]

        self._stop.clear()
        self._start_button.configure(state="disabled")
        self._choose_button.configure(state="disabled")
        self._microphone.configure(state="disabled")
        self._stop_button.configure(state="normal")
        self._status.set("Starting narration...")
        self._append(f"Document: {self.pdf_path.name}")
        self._append(f"Microphone: {microphone_name} (device {input_device})")
        self._worker = Thread(target=self._run_session, args=(self.pdf_path, input_device))
        self._worker.start()

    def stop(self) -> None:
        """Request a safe stop without advancing an unfinished audio segment."""
        if self._worker is not None and self._worker.is_alive():
            self._stop.set()
            self._stop_button.configure(state="disabled")
            self._status.set("Stopping after the current processing step...")

    def close(self) -> None:
        """Wait for a running session to stop before closing the window."""
        self._closing = True
        self.stop()
        if self._worker is None or not self._worker.is_alive():
            self.root.destroy()

    def _run_session(self, pdf_path: Path, input_device: int) -> None:
        try:
            result = run_continuous_session(
                pdf_path,
                VAD_PATH,
                OUTPUT_DIR,
                stop_requested=self._stop.is_set,
                on_update=lambda message: self._events.put(("update", message)),
                input_device=input_device,
            )
            self._events.put(("done", result))
        except Exception as error:
            self._events.put(("error", f"{type(error).__name__}: {error}"))

    def _append(self, message: str) -> None:
        self._log.configure(state="normal")
        self._log.insert("end", message + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _poll_events(self) -> None:
        while True:
            try:
                kind, payload = self._events.get_nowait()
            except Empty:
                break
            if kind == "update":
                assert isinstance(payload, str)
                self._status.set(payload)
                if not payload.startswith("Narrating page "):
                    self._append(payload)
            elif kind == "done":
                assert isinstance(payload, SessionResult)
                state = payload.narration_state
                self._append(
                    f"Segments: {payload.segments_attempted}; interruptions: "
                    f"{payload.interruptions}; cursor: page {state.page_number}, "
                    f"character {state.text_offset}."
                )
                self._status.set("Narration complete." if state.is_complete else "Session stopped.")
                self._finish_run()
            else:
                assert isinstance(payload, str)
                self._append(payload)
                self._status.set("Session failed. See the log for details.")
                self._finish_run()

        if self._closing and (self._worker is None or not self._worker.is_alive()):
            self.root.destroy()
        else:
            self.root.after(100, self._poll_events)

    def _finish_run(self) -> None:
        self._start_button.configure(state="normal")
        self._choose_button.configure(state="normal")
        self._microphone.configure(state="readonly")
        self._stop_button.configure(state="disabled")


def main() -> None:
    """Launch the local PDF and voice interface."""
    root = tk.Tk()
    ConversIQWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
