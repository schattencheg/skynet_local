"""pyttsx3 text-to-speech engine — runs in a dedicated daemon thread so it never
blocks the main frame loop.

On Windows the SAPI5 backend is used automatically; Russian is supported when
a Russian SAPI5 voice is installed (e.g. Microsoft Irina).
"""

from __future__ import annotations

import queue
import threading


class Pyttsx3TTS:
    """Non-blocking TTS: callers enqueue text, a background thread speaks it."""

    def __init__(self, rate: int = 175, volume: float = 1.0, voice_id: str | None = None) -> None:
        self._rate = rate
        self._volume = volume
        self._voice_id = voice_id
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="tts-worker")
        self._engine = None
        self._ready = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialise the pyttsx3 engine and start the worker thread."""
        self._thread.start()

    def stop(self) -> None:
        """Signal the worker to shut down and wait for it to finish."""
        self._queue.put(None)  # sentinel
        self._thread.join(timeout=3.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str) -> None:
        """Enqueue *text* for asynchronous speech output."""
        if text and self._ready:
            self._queue.put(text)

    # ------------------------------------------------------------------
    # Internal worker
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        """Runs in a daemon thread; owns the pyttsx3 engine for its lifetime."""
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self._rate)
            self._engine.setProperty("volume", self._volume)
            if self._voice_id:
                self._engine.setProperty("voice", self._voice_id)
            self._ready = True
        except Exception as exc:
            print(f"[Pyttsx3TTS] init failed: {exc}")
            return

        while True:
            text = self._queue.get()  # blocks until something arrives
            if text is None:          # shutdown sentinel
                break
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as exc:
                print(f"[Pyttsx3TTS] speak error: {exc}")
