"""Speaker recogniser stub — returns unknown until a real embedding backend is wired.

The interface mirrors the SpeakerRecognizer protocol so it can be swapped for a
SpeechBrain or ONNX implementation without touching higher-level code.
"""

from __future__ import annotations

from skynet_local.domain.entities.observation import SpeakerObservation


class OnnxSpeakerRecognizer:
    """Placeholder speaker recogniser that always returns an unknown-speaker result."""

    def __init__(self, voiceprints_dir: str | None = None, threshold: float = 0.70) -> None:
        # voiceprints_dir reserved for future .npy embedding storage.
        self._voiceprints_dir = voiceprints_dir
        self._threshold = threshold

    def recognize(self, pcm_bytes: bytes, sample_rate: int) -> SpeakerObservation | None:  # noqa: ARG002
        """Accept a PCM chunk and return an observation; currently always unknown."""
        # TODO: replace with SpeechBrain ECAPA-TDNN embedding + cosine match.
        return SpeakerObservation(speaker_id=None, confidence=0.0, text="")
