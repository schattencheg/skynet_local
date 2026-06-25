"""Abstract base class for single-frame emotion detectors."""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class EmotionDetectorBase(ABC):
    """Contract: receive a BGR face crop, return {emotion_label: probability}."""

    @abstractmethod
    def infer(self, crop: "np.ndarray", track_id: str) -> "dict[str, float]":
        """Run inference on a single face crop.

        Returns a dict mapping every emotion label to a probability in [0, 1].
        Probabilities should sum to ~1 but are not required to.
        Must never raise -- return an empty dict on failure.
        """

    @property
    @abstractmethod
    def labels(self) -> "list[str]":
        """Ordered list of all emotion labels this detector produces."""
