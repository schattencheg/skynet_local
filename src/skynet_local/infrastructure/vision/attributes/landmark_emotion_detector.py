"""Landmark-based emotion estimator using YuNet mouth/eye geometry.

No ONNX model required. Combine with FerplusEmotionDetector via
EnsembleEmotionDetector for better accuracy.
"""

from __future__ import annotations
import numpy as np
from skynet_local.infrastructure.vision.attributes.emotion_detector_base import EmotionDetectorBase

_LABELS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger",   "disgust",   "fear",     "contempt",
]


class LandmarkEmotionDetector(EmotionDetectorBase):
    """Geometry-based emotion estimator -- no model file needed.

    Call set_landmarks(track_id, raw_yunet_row) each frame before analyze().
    """

    def __init__(self) -> None:
        self._rows: dict[str, list] = {}

    @property
    def labels(self) -> list[str]:
        return list(_LABELS)

    def set_landmarks(self, track_id: str, raw_row) -> None:
        if raw_row is not None:
            self._rows[track_id] = raw_row

    def infer(self, crop: np.ndarray, track_id: str) -> dict[str, float]:
        row = self._rows.get(track_id)
        probs = {l: 0.0 for l in _LABELS}

        if row is None or len(row) < 14:
            probs["neutral"] = 1.0
            return probs

        try:
            # YuNet landmark cols: re=4,5  le=6,7  nose=8,9  mr=10,11  ml=12,13
            re  = (float(row[4]),  float(row[5]))
            le  = (float(row[6]),  float(row[7]))
            mr  = (float(row[10]), float(row[11]))
            ml  = (float(row[12]), float(row[13]))

            mouth_w = abs(ml[0] - mr[0]) + 1e-6
            mouth_h = abs(ml[1] - mr[1])
            mar     = mouth_h / mouth_w

            eye_w   = abs(le[0] - re[0]) + 1e-6
            eye_h   = abs(le[1] - re[1])
            ear     = eye_h / eye_w

            mouth_width_norm = mouth_w / (eye_w + 1e-6)
            happy_score    = float(np.clip((mouth_width_norm - 0.6) * 2.5, 0, 1))
            sad_score      = float(np.clip((0.12 - mar) * 8.0, 0, 1))
            sad_score      = max(sad_score, float(np.clip((0.55 - mouth_width_norm) * 3.0, 0, 1)))
            surprise_score = float(np.clip(mar * 6.0, 0, 1) * 0.5 +
                                   np.clip(ear * 4.0, 0, 1) * 0.5)
            anger_score    = float(np.clip(ear * 3.0, 0, 0.6))

            raw = {
                "happiness": happy_score,
                "sadness":   sad_score,
                "surprise":  surprise_score,
                "anger":     anger_score * 0.5,
                "neutral":   0.0,
                "disgust":   0.0,
                "fear":      0.0,
                "contempt":  0.0,
            }
            total = sum(raw.values()) + 1e-6
            neutral_fill = max(0.0, 1.0 - total)
            raw["neutral"] = neutral_fill
            total += neutral_fill
            return {k: v / total for k, v in raw.items()}

        except Exception:
            probs["neutral"] = 1.0
            return probs
