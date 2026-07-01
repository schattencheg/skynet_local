"""AudioService — background thread that ties together microphone, STT, speaker ID
and TTS into a single startable/stoppable service.

The service updates a shared SpeakerObservation atomically so the main frame
loop can read it each cycle without blocking.
"""

from __future__ import annotations

import threading
from pathlib import Path

from skynet_local.domain.entities.observation import SpeakerObservation
from skynet_local.infrastructure.audio.microphone import MicrophoneStream
from skynet_local.infrastructure.audio.speaker.onnx_speaker_recognizer import OnnxSpeakerRecognizer
from skynet_local.infrastructure.audio.stt.vosk_stt import VoskSpeechRecognizer
from skynet_local.infrastructure.audio.tts.pyttsx3_tts import Pyttsx3TTS
from skynet_local.infrastructure.audio.tts.null_tts import NullTTS

# Minimum RMS energy to consider a chunk as containing speech (avoids processing silence).
_ENERGY_THRESHOLD = 300


class AudioService:
    """Coordinates mic → STT → speaker ID and exposes the latest observation + TTS."""

    def __init__(
        self,
        vosk_model_path: str | Path | None = None,
        voiceprints_dir: str | Path | None = None,
        tts_rate: int = 175,
        tts_voice_id: str | None = None,
        enable_tts: bool = True,
    ) -> None:
        self._mic = MicrophoneStream()
        self._stt = VoskSpeechRecognizer(vosk_model_path) if vosk_model_path else None
        self._speaker = OnnxSpeakerRecognizer(voiceprints_dir=voiceprints_dir)
        self._tts: Pyttsx3TTS | NullTTS = (
            Pyttsx3TTS(rate=tts_rate, voice_id=tts_voice_id) if enable_tts else NullTTS()
        )

        # Latest recognised observation; written by audio thread, read by frame loop.
        self._latest: SpeakerObservation | None = None
        self._lock = threading.Lock()

        self._thread = threading.Thread(
            target=self._run, daemon=True, name="audio-service"
        )
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start microphone capture, TTS engine and the processing thread."""
        self._tts.start()
        self._mic.start()
        self._thread.start()
        print("[AudioService] started")

    def stop(self) -> None:
        """Gracefully stop all audio subsystems."""
        self._stop_event.set()
        self._mic.stop()
        self._tts.stop()
        self._thread.join(timeout=3.0)
        print("[AudioService] stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def latest_observation(self) -> SpeakerObservation | None:
        """Thread-safe read of the most recent speaker/STT observation."""
        with self._lock:
            return self._latest

    def speak(self, text: str) -> None:
        """Enqueue *text* for TTS playback (non-blocking)."""
        self._tts.speak(text)

    # ------------------------------------------------------------------
    # Internal processing loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Main audio loop: read mic chunks, feed STT, update observation."""
        while not self._stop_event.is_set():
            chunk = self._mic.read_chunk(timeout=0.1)
            if chunk is None:
                continue

            # Skip silent chunks to avoid wasting CPU on STT.
            if not _is_speech(chunk):
                continue

            # Feed the chunk into Vosk STT if available.
            text = ""
            if self._stt:
                self._stt.accept_chunk(chunk)
                text = self._stt.poll_text()

            # Speaker identification (stub for now).
            obs = self._speaker.recognize(chunk, self._mic.sample_rate)

            if obs is not None:
                obs.text = text
                with self._lock:
                    self._latest = obs


def _is_speech(pcm_bytes: bytes, threshold: int = _ENERGY_THRESHOLD) -> bool:
    """Cheap energy-based voice activity check — avoids importing webrtcvad."""
    import struct
    samples = struct.unpack_from(f"{len(pcm_bytes) // 2}h", pcm_bytes)
    if not samples:
        return False
    rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
    return rms > threshold
