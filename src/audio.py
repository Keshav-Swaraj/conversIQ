"""Local recording, interruptible playback, and microphone monitoring."""

from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from math import ceil, isfinite
from pathlib import Path
from queue import Empty, Full, Queue
import wave

import sounddevice as sd

from src.vad import SileroVad


def record_microphone(
    output_path: str | Path, duration_seconds: float, sample_rate: int = 16_000
) -> Path:
    """Record a fixed-length mono, 16-bit WAV from the default microphone.

    The call blocks until recording is complete. It does not perform speech
    detection or interrupt audio playback.
    """
    wav_path = Path(output_path)
    if wav_path.suffix.lower() != ".wav":
        raise ValueError(f"Microphone recording must be saved as .wav: {wav_path}")
    if not isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be a finite positive number.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")

    frames = int(duration_seconds * sample_rate)
    if frames == 0:
        raise ValueError("Recording duration is too short for the sample rate.")
    try:
        recording = sd.rec(
            frames, samplerate=sample_rate, channels=1, dtype="int16", blocking=True
        )
    except sd.PortAudioError as error:
        raise RuntimeError("Unable to record from the default microphone.") from error

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(recording.tobytes())
    return wav_path


def play_wav_interruptible(
    file_path: str | Path,
    should_interrupt: Callable[[], bool],
    block_frames: int = 1_024,
) -> bool:
    """Play a PCM WAV in small blocks; return false if playback is stopped.

    The caller supplies the interrupt signal. A stopped WAV is not considered
    fully heard; narration must not advance its text cursor for that segment.
    """
    wav_path = Path(file_path)
    if not wav_path.is_file():
        raise FileNotFoundError(f"WAV file does not exist: {wav_path}")
    if wav_path.suffix.lower() != ".wav":
        raise ValueError(f"Expected a .wav file: {wav_path}")
    if block_frames <= 0:
        raise ValueError("block_frames must be positive.")

    try:
        with wave.open(str(wav_path), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE" or wav_file.getsampwidth() != 2:
                raise ValueError("Interruptible playback requires a 16-bit PCM WAV.")
            with sd.RawOutputStream(
                samplerate=wav_file.getframerate(),
                channels=wav_file.getnchannels(),
                dtype="int16",
                blocksize=block_frames,
            ) as stream:
                while True:
                    try:
                        interrupted = should_interrupt()
                    except BaseException:
                        stream.abort()
                        raise
                    if interrupted:
                        stream.abort()
                        return False
                    audio_block = wav_file.readframes(block_frames)
                    if not audio_block:
                        return True
                    stream.write(audio_block)
    except wave.Error as error:
        raise ValueError(f"Unable to read WAV file: {wav_path}") from error
    except sd.PortAudioError as error:
        raise RuntimeError("Unable to play WAV through the default output device.") from error


class SpeechMonitor:
    """Detect sustained speech and retain microphone audio for the question."""

    def __init__(
        self,
        model_path: str | Path,
        threshold: float = 0.5,
        min_speech_frames: int = 6,
    ) -> None:
        if not 0 < threshold < 1:
            raise ValueError("VAD threshold must be between 0 and 1.")
        if min_speech_frames <= 0:
            raise ValueError("min_speech_frames must be positive.")
        self._vad = SileroVad(model_path)
        self._threshold = threshold
        self._min_speech_frames = min_speech_frames
        self._consecutive_speech = 0
        self._triggered = False
        self._frames: Queue[bytes] = Queue(maxsize=128)
        self._recent: deque[bytes] = deque(maxlen=32)
        self._audio_error: str | None = None

    def _on_audio(
        self, indata: memoryview, frames: int, time_info: object, status: object
    ) -> None:
        """Copy audio from PortAudio's callback; inference runs elsewhere."""
        if status or frames != 512:
            self._audio_error = f"Microphone stream error: {status or 'unexpected frame size'}"
            return
        try:
            self._frames.put_nowait(bytes(indata))
        except Full:
            self._audio_error = "Microphone input overflowed before VAD could process it."

    def speech_detected(self) -> bool:
        """Process queued 32 ms frames and report sustained speech onset."""
        if self._audio_error:
            raise RuntimeError(self._audio_error)
        if self._triggered:
            return True
        while True:
            try:
                frame = self._frames.get_nowait()
            except Empty:
                return False
            self._recent.append(frame)
            if self._vad.speech_probability(frame) >= self._threshold:
                self._consecutive_speech += 1
            else:
                self._consecutive_speech = 0
            if self._consecutive_speech >= self._min_speech_frames:
                self._triggered = True
                return True

    def capture_question(self, output_path: str | Path, duration_seconds: float) -> Path:
        """Save recent speech plus a fixed period after interruption to WAV."""
        if not self._triggered:
            raise RuntimeError("Cannot capture a question before speech is detected.")
        if not isfinite(duration_seconds) or duration_seconds <= 0:
            raise ValueError("duration_seconds must be a finite positive number.")
        wav_path = Path(output_path)
        if wav_path.suffix.lower() != ".wav":
            raise ValueError(f"Question recording must be a .wav file: {wav_path}")
        chunks = list(self._recent)
        for _ in range(ceil(duration_seconds * 16_000 / 512)):
            if self._audio_error:
                raise RuntimeError(self._audio_error)
            try:
                chunks.append(self._frames.get(timeout=2))
            except Empty as error:
                raise RuntimeError("Microphone stopped providing question audio.") from error
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16_000)
            wav_file.writeframes(b"".join(chunks))
        return wav_path


@contextmanager
def monitor_microphone(
    model_path: str | Path, input_device: int | None = None
) -> Iterator[SpeechMonitor]:
    """Monitor a selected microphone, or the default input if none is supplied."""
    monitor = SpeechMonitor(model_path)
    try:
        with sd.RawInputStream(
            samplerate=16_000,
            channels=1,
            dtype="int16",
            blocksize=512,
            device=input_device,
            callback=monitor._on_audio,
        ):
            yield monitor
    except sd.PortAudioError as error:
        raise RuntimeError("Unable to monitor the default microphone.") from error
