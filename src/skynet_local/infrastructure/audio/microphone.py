"""Microphone capture: streams PCM-16 mono chunks from the default input device."""

from __future__ import annotations

import queue
import threading

import numpy as np
import sounddevice as sd

# Default audio parameters expected by Vosk and most STT backends.
SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "int16"
# One chunk ≈ 100 ms of audio — long enough for VAD, short enough to feel responsive.
CHUNK_FRAMES = 1_600


class MicrophoneStream:
    """Opens the default microphone and delivers raw PCM bytes via a thread-safe queue."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        chunk_frames: int = CHUNK_FRAMES,
        device: int | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._chunk_frames = chunk_frames
        self._device = device
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=50)
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the sounddevice input stream and begin buffering chunks."""
        with self._lock:
            if self._stream is not None:
                return
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=self._chunk_frames,
                device=self._device,
                callback=self._callback,
            )
            self._stream.start()

    def stop(self) -> None:
        """Stop and close the input stream."""
        with self._lock:
            if self._stream is None:
                return
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read_chunk(self, timeout: float = 0.2) -> bytes | None:
        """Return the next buffered PCM chunk, or None if no data arrived in time."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def sample_rate(self) -> int:
        """Sample rate used by this stream."""
        return self._sample_rate

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,  # noqa: ARG002
        time: object,  # noqa: ARG002
        status: sd.CallbackFlags,
    ) -> None:
        """sounddevice callback — converts ndarray to bytes and enqueues it."""
        if status:
            # Log overflow / underflow without crashing the stream.
            print(f"[MicrophoneStream] status: {status}")
        if not self._queue.full():
            self._queue.put_nowait(indata.tobytes())
