"""EmotionAnalyzer: vote-buffer + cooldown wrapper around any EmotionDetectorBase."""

from __future__ import annotations
from collections import deque, Counter
from dataclasses import dataclass, field
from typing import Optional

from skynet_local.infrastructure.vision.attributes.emotion_detector_base import EmotionDetectorBase

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
    vote_buffer: deque = field(default_factory=lambda: deque())
    committed: Optional[str] = None
    cooldown_left: int = 0


class EmotionAnalyzer:
    """Vote-buffer + cooldown wrapper. Backend is any EmotionDetectorBase."""

    MIN_FACE_PX = 48

    def __init__(
        self,
        detector: EmotionDetectorBase,
        n_det: int = 3,
        n_cooldown: int = 10,
    ) -> None:
        self._detector   = detector
        self._n_det      = n_det
        self._n_cooldown = n_cooldown
        self._states: dict[str, _EmotionState] = {}
        self._last_probs: dict[str, dict[str, float]] = {}

    def analyze(self, frame, faces):
        active_ids = {f.track_id for f in faces}
        for tid in list(self._states):
            if tid not in active_ids:
                del self._states[tid]
                self._last_probs.pop(tid, None)

        for face in faces:
            x1, y1, x2, y2 = face.bbox
            w, h = x2 - x1, y2 - y1

            state = self._states.setdefault(face.track_id, _EmotionState(
                vote_buffer=deque(maxlen=self._n_det)
            ))

            if w < self.MIN_FACE_PX or h < self.MIN_FACE_PX:
                face.emotion = state.committed
                continue

            crop = frame[
                max(0, y1):min(frame.shape[0], y2),
                max(0, x1):min(frame.shape[1], x2),
            ]
            if crop.size == 0:
                face.emotion = state.committed
                continue

            probs = self._detector.infer(crop, face.track_id)
            if probs:
                self._last_probs[face.track_id] = probs
                detected = max(probs, key=probs.get)
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
