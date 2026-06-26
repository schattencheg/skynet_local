"""Ensemble detector: weighted average over multiple EmotionDetectorBase backends."""

from __future__ import annotations

import numpy as np

from skynet_local.infrastructure.vision.attributes.emotion_detector_base import EmotionDetectorBase


class EnsembleEmotionDetector(EmotionDetectorBase):
    """Combine N detectors with per-detector weights."""

    _detectors: list[tuple[EmotionDetectorBase, float]]
    _labels: list[str]

    def __init__(
        self, detectors: list[tuple[EmotionDetectorBase, float]]
    ) -> None:
        self._detectors = detectors
        seen: set[str] = set()
        all_labels: list[str] = []
        for det, _ in detectors:
            for lbl in det.labels:
                if lbl not in seen:
                    all_labels.append(lbl)
                    seen.add(lbl)
        self._labels = all_labels

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    def set_landmarks(self, track_id: str, raw_row: np.ndarray | None) -> None:
        """Forward landmark rows to any backend that accepts them."""
        for det, _ in self._detectors:
            if hasattr(det, "set_landmarks"):
                det.set_landmarks(track_id, raw_row)  # type: ignore[union-attr]

    def infer(self, crop: np.ndarray, track_id: str) -> dict[str, float]:
        combined: dict[str, float] = {lbl: 0.0 for lbl in self._labels}
        total_w: float = 0.0

        for det, w in self._detectors:
            result = det.infer(crop, track_id)
            if not result:
                continue
            for label, prob in result.items():
                if label in combined:
                    combined[label] += prob * w
            total_w += w

        if total_w > 0:
            combined = {k: v / total_w for k, v in combined.items()}

        s = sum(combined.values()) + 1e-6
        return {k: v / s for k, v in combined.items()}
