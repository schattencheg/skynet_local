"""Emotion analyzer using the FER+ ONNX model via cv2.dnn."""

from __future__ import annotations
from collections import deque, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_FERPLUS_LABELS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger",   "disgust",   "fear",     "contempt",
]

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

# Per-class probability threshold multipliers relative to base min_prob.
# Classes that FER+ under-detects get a lower effective threshold so they
# can actually win against the happiness/neutral dominance.
_CLASS_THRESHOLD_MULT: dict[str, float] = {
    "happiness": 1.0,
    "neutral":   0.70,
    "surprise":  0.85,
    "sadness":   0.65,   # heavily under-detected — lower bar
    "anger":     0.65,   # heavily under-detected — lower bar
    "disgust":   0.60,
    "fear":      0.60,
    "contempt":  0.60,
}

_TEMPERATURE: float = 3.0   # higher = flatter distribution = less happy bias


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
    """FER+ ONNX emotion with vote buffering and cooldown persistence."""

    _INPUT_SIZE = (64, 64)

    def __init__(
        self,
        model_path: str | Path,
        min_face_px: int = 48,
        n_det: int = 3,
        n_cooldown: int = 10,
        min_prob: float = 0.23,
    ) -> None:
        self._model_path = Path(model_path)
        self._min_face_px = min_face_px
        self._n_det = n_det
        self._n_cooldown = n_cooldown
        self._min_prob = min_prob
        self._net: cv2.dnn.Net | None = None
        self._load_failed = False
        self._states: dict[str, _EmotionState] = {}
        self._last_probs: dict[str, dict[str, float]] = {}   # track_id → {label: prob}

    def _load(self) -> bool:
        if self._net is not None:
            return True
        if self._load_failed:
            return False
        if not self._model_path.exists():
            print(f"[EmotionAnalyzer] Model not found: {self._model_path}")
            self._load_failed = True
            return False
        try:
            self._net = cv2.dnn.readNetFromONNX(str(self._model_path))
        except Exception as exc:
            print(f"[EmotionAnalyzer] Failed to load model: {exc}")
            self._load_failed = True
            return False
        return True

    @staticmethod
    def _softmax_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
        scaled = logits / temperature
        scaled -= scaled.max()
        exp = np.exp(scaled)
        return exp / exp.sum()

    def _infer_emotion(self, crop, track_id: str) -> Optional[str]:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self._INPUT_SIZE)
        blob = cv2.dnn.blobFromImage(resized, 1.0, self._INPUT_SIZE, 0, swapRB=False)
        self._net.setInput(blob)
        logits = self._net.forward()[0]
        probs = self._softmax_temperature(logits, _TEMPERATURE)

        # Store full distribution for SceneTracker
        self._last_probs[track_id] = {
            _FERPLUS_LABELS[i]: float(probs[i]) for i in range(len(_FERPLUS_LABELS))
        }

        order = np.argsort(probs)[::-1]
        for idx in order:
            label = _FERPLUS_LABELS[idx]
            threshold = self._min_prob * _CLASS_THRESHOLD_MULT.get(label, 1.0)
            if probs[idx] >= threshold:
                return label
        return "neutral"

    def analyze(self, frame, faces):
        if not self._load():
            return faces

        active_ids = {f.track_id for f in faces}
        # Purge stale states
        for tid in list(self._states):
            if tid not in active_ids:
                del self._states[tid]

        for face in faces:
            x1, y1, x2, y2 = face.bbox
            w, h = x2 - x1, y2 - y1

            state = self._states.setdefault(face.track_id, _EmotionState(
                vote_buffer=deque(maxlen=self._n_det)
            ))

            if w < self._min_face_px or h < self._min_face_px:
                face.emotion = state.committed
                continue

            crop = frame[
                max(0, y1):min(frame.shape[0], y2),
                max(0, x1):min(frame.shape[1], x2),
            ]
            if crop.size == 0:
                face.emotion = state.committed
                continue

            try:
                detected = self._infer_emotion(crop, face.track_id)
            except Exception:
                detected = None

            if detected:
                state.vote_buffer.append(detected)

            # Commit when buffer full and majority exists
            if len(state.vote_buffer) == self._n_det:
                counts = Counter(state.vote_buffer)
                top_emotion, top_count = counts.most_common(1)[0]
                majority = self._n_det // 2 + 1
                if top_count >= majority:
                    if top_emotion != state.committed:
                        if state.cooldown_left == 0:          # ← only switch when cooldown done
                            state.committed = top_emotion
                            state.cooldown_left = self._n_cooldown
                            state.vote_buffer.clear()
                        # else: still in cooldown — keep old emotion, discard buffer
                        # (don't clear buffer so votes keep accumulating)
                    else:
                        # Same emotion confirmed again — refresh cooldown
                        state.cooldown_left = self._n_cooldown
                        state.vote_buffer.clear()

            # Tick cooldown — emotion persists until cooldown expires naturally
            if state.cooldown_left > 0:
                state.cooldown_left -= 1
            # NOTE: do NOT clear state.committed here — keep showing last emotion

            face.emotion = state.committed
        return faces
