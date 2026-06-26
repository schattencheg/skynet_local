"""Protocol port for face-recognition services used by detectors and orchestrator."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from skynet_local.domain.entities import FaceRecognitionResult


class FaceRecognizerPort(Protocol):
    """Minimal contract expected by detectors that do inline recognition."""

    def recognize_detection(
        self,
        frame: np.ndarray,
        face_row: np.ndarray,
    ) -> FaceRecognitionResult: ...

    def enroll_detection(
        self,
        person_id: str,
        display_name: str,
        frame: np.ndarray,
        face_row: np.ndarray,
        quality: float,
    ) -> None: ...
