"""Vosk-based offline speech-to-text recogniser.

Expects a downloaded Vosk model directory, e.g.:
  models/audio/vosk-model-small-ru-0.22
Download from: https://alphacephei.com/vosk/models
"""

from __future__ import annotations

import json
import queue
from pathlib import Path


class VoskSpeechRecognizer:
    """Wraps Vosk KaldiRecognizer and converts raw PCM chunks to recognised text."""

    def __init__(self, model_path: str | Path, sample_rate: int = 16_000) -> None:
        # Import lazily so the rest of the app starts even without vosk installed.
        try:
            from vosk import KaldiRecognizer, Model, SetLogLevel
            SetLogLevel(-1)  # silence Vosk startup noise
            self._model = Model(str(model_path))
            self._rec = KaldiRecognizer(self._model, sample_rate)
            self._rec.SetWords(True)
            self._ready = True
        except Exception as exc:
            print(f"[VoskSpeechRecognizer] could not load model: {exc}")
            self._ready = False
            self._rec = None

        self._pending: queue.SimpleQueue[str] = queue.SimpleQueue()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def accept_chunk(self, pcm_bytes: bytes) -> None:
        """Feed one PCM-16 chunk into the recogniser; enqueue any final result."""
        if not self._ready or not pcm_bytes:
            return
        if self._rec.AcceptWaveform(pcm_bytes):
            result = json.loads(self._rec.Result())
            text = result.get("text", "").strip()
            if text:
                self._pending.put(text)

    def poll_text(self) -> str:
        """Return the latest finalised utterance, or empty string if none ready."""
        try:
            return self._pending.get_nowait()
        except queue.Empty:
            return ""

    def partial_text(self) -> str:
        """Return the in-progress partial result (not yet a full sentence)."""
        if not self._ready:
            return ""
        result = json.loads(self._rec.PartialResult())
        return result.get("partial", "").strip()

    @property
    def is_ready(self) -> bool:
        """True when the Vosk model loaded successfully."""
        return self._ready
