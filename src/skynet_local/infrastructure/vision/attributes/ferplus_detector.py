"""FER+ ONNX emotion detector backend (cv2.dnn, no TF/Keras)."""

from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np

from skynet_local.infrastructure.vision.attributes.emotion_detector_base import EmotionDetectorBase

_LABELS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger",   "disgust",   "fear",     "contempt",
]

_THRESHOLD_MULT: dict[str, float] = {
    "happiness": 1.00,
    "neutral":   0.70,
    "surprise":  0.85,
    "sadness":   0.55,
    "anger":     0.55,
    "disgust":   0.55,
    "fear":      0.55,
    "contempt":  0.55,
}

_TEMPERATURE = 3.0
_INPUT_SIZE  = (64, 64)


class FerplusEmotionDetector(EmotionDetectorBase):
    """FER+ 8-class emotion via cv2.dnn.readNetFromONNX."""

    def __init__(self, model_path: str | Path, min_prob: float = 0.18) -> None:
        self._model_path = Path(model_path)
        self._min_prob   = min_prob
        self._net: cv2.dnn.Net | None = None
        self._failed     = False

    @property
    def labels(self) -> list[str]:
        return list(_LABELS)

    def _load(self) -> bool:
        if self._net is not None:
            return True
        if self._failed:
            return False
        if not self._model_path.exists():
            print(f"[FerplusDetector] model not found: {self._model_path}")
            self._failed = True
            return False
        try:
            self._net = cv2.dnn.readNetFromONNX(str(self._model_path))
        except Exception as e:
            print(f"[FerplusDetector] load error: {e}")
            self._failed = True
            return False
        return True

    @staticmethod
    def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
        s = logits / temperature
        s -= s.max()
        e = np.exp(s)
        return e / e.sum()

    def infer(self, crop: np.ndarray, track_id: str) -> dict[str, float]:
        if not self._load():
            return {}
        try:
            gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, _INPUT_SIZE)
            blob    = cv2.dnn.blobFromImage(resized, 1.0, _INPUT_SIZE, 0, swapRB=False)
            self._net.setInput(blob)
            logits  = self._net.forward()[0]
            probs   = self._softmax(logits, _TEMPERATURE)
            raw = {_LABELS[i]: float(probs[i]) for i in range(len(_LABELS))}
            # Suppress classes that don't clear their per-class threshold
            filtered = {
                label: prob if prob >= self._min_prob * _THRESHOLD_MULT.get(label, 1.0) else 0.0
                for label, prob in raw.items()
            }
            total = sum(filtered.values()) or 1.0
            return {k: v / total for k, v in filtered.items()}
        except Exception:
            return {}
