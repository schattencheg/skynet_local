"""EmotionAnalyzer: vote-buffer + cooldown wrapper around any EmotionDetectorBase."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from skynet_local.infrastructure.vision.attributes.emotion_detector_base import EmotionDetectorBase
from skynet_local.domain.entities import FaceObservation

_EMOTION_EMOTICONS: dict[str, str] = {
    "happiness": ":-)",
    "sadness":   ":-(",
    "anger":     ">:(",
    "surprise":  ":-O",
    "fear":      "D-:",
    "disgust":   ":-/",
    "contempt":  ":-|",
    "neutral":   "",
}


def emotion_to_emoticon(emotion: str | None) -> str:
    if not emotion:
        return ""
    return _EMOTION_EMOTICONS.get(emotion.lower(), "")


@dataclass
class _EmotionState:
    vote_buffer: deque[str] = field(default_factory=deque[str])
    committed: str | None = None
    cooldown_left: int = 0


class _NullEmotionDetector(EmotionDetectorBase):
    """No-op detector used when no emotion backend is configured."""

    @property
    def labels(self) -> list[str]:
        return []

    def infer(self, crop: np.ndarray, track_id: str) -> dict[str, float]:
        return {}

    def set_track_landmarks(self, track_id: str, landmarks: object | None) -> None:
        pass


class EmotionAnalyzer:
    """Vote-buffer + cooldown wrapper. Backend is any EmotionDetectorBase."""

    MIN_FACE_PX: int = 48

    _detector: EmotionDetectorBase
    _n_det: int
    _n_cooldown: int
    _states: dict[str, _EmotionState]
    _last_probs: dict[str, dict[str, float]]

    def __init__(
        self,
        detector: EmotionDetectorBase,
        n_det: int = 3,
        n_cooldown: int = 10,
    ) -> None:
        self._detector = detector
        self._n_det = n_det
        self._n_cooldown = n_cooldown
        self._states: dict[str, _EmotionState] = {}
        self._last_probs: dict[str, dict[str, float]] = {}

    @classmethod
    def null(cls) -> "EmotionAnalyzer":
        """Return a no-op analyzer used as a safe default."""
        return cls(detector=_NullEmotionDetector())

    def analyze(
        self, frame: np.ndarray, faces: list[FaceObservation]
    ) -> list[FaceObservation]:
        active_ids = {f.track_id for f in faces}
        for tid in list(self._states):
            if tid not in active_ids:
                del self._states[tid]
                self._last_probs.pop(tid, None)

        for face in faces:
            x1, y1, x2, y2 = face.bbox
            w, h = x2 - x1, y2 - y1

            state = self._states.setdefault(
                face.track_id,
                _EmotionState(vote_buffer=deque(maxlen=self._n_det)),
            )

            if w < self.MIN_FACE_PX or h < self.MIN_FACE_PX:
                face.emotion = state.committed
                continue

            crop = frame[
                max(0, y1) : min(frame.shape[0], y2),
                max(0, x1) : min(frame.shape[1], x2),
            ]
            if crop.size == 0:
                face.emotion = state.committed
                continue

            probs = self._detector.infer(crop, face.track_id)
            if probs:
                self._last_probs[face.track_id] = probs
                detected = max(probs, key=lambda k: probs[k])
                state.vote_buffer.append(detected)

            if len(state.vote_buffer) == self._n_det:
                counts = Counter(state.vote_buffer)
                top_emotion, top_count = counts.most_common(1)[0]
                majority = self._n_det // 2 + 1
                if top_count >= majority:
                    if top_emotion != state.committed:
                        if state.cooldown_left == 0:
                            state.committed = top_emotion
                            state.cooldown_left = self._n_cooldown
                            state.vote_buffer.clear()
                    else:
                        state.cooldown_left = self._n_cooldown
                        state.vote_buffer.clear()

            if state.cooldown_left > 0:
                state.cooldown_left -= 1

            face.emotion = state.committed

        return faces

    def set_track_landmarks(self, track_id: str, landmarks: object | None) -> None:
        """Forward landmarks to a landmark-aware backend if it supports them."""
        if hasattr(self._detector, "set_landmarks"):
            self._detector.set_landmarks(track_id, landmarks)  # type: ignore[union-attr]
