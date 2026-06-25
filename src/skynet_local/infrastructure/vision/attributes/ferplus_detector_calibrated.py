from .ferplus_detector import FerplusEmotionDetector


class FerplusEmotionDetectorCalibrated(FerplusEmotionDetector):
    """FER+ with per-class probability floor suppression and optional
    prior-correction weights to compensate dataset bias.

    per_class_scale: multiply each class raw prob before normalising.
    Higher scale = boost that class relative to others.
    """

    _DEFAULT_SCALE: dict[str, float] = {
        "happiness": 0.80,   # FER+ over-fires on happiness
        "neutral":   0.65,   # neutral dominates — pull back hard
        "surprise":  0.90,
        "sadness":   1.60,   # heavily under-detected — boost
        "anger":     1.60,   # heavily under-detected — boost
        "disgust":   1.40,
        "fear":      1.40,
        "contempt":  1.30,
    }

    def __init__(
        self,
        model_path,
        min_prob: float = 0.15,
        per_class_scale: dict[str, float] | None = None,
    ) -> None:
        super().__init__(model_path, min_prob)
        self._scale = per_class_scale or self._DEFAULT_SCALE

    def infer(self, crop, track_id: str) -> dict[str, float]:
        raw = super().infer(crop, track_id)   # gets threshold-filtered probs from parent
        if not raw:
            return {}
        # Apply prior-correction scale
        scaled = {k: v * self._scale.get(k, 1.0) for k, v in raw.items()}
        total = sum(scaled.values()) or 1.0
        return {k: v / total for k, v in scaled.items()}
