"""Protocols defining the pluggable contracts for detectors, recognizers and GUI adapters."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import numpy as np

from skynet_local.domain.entities import FaceObservation, SceneState, SpeakerObservation


class FrameSource(Protocol):
    """Produce frames from a camera, video file or still image source."""

    def frames(self) -> Iterable[np.ndarray]: ...


class FaceDetector(Protocol):
    """Locate faces in a frame and return preliminary observations."""

    def detect(self, frame: np.ndarray) -> list[FaceObservation]: ...


class FaceRecognizer(Protocol):
    """Enrich face observations with identity and confidence information."""

    def recognize(self, frame: np.ndarray, faces: list[FaceObservation]) -> list[FaceObservation]: ...


class SpeakerRecognizer(Protocol):
    """Infer the currently active speaker identity from audio input."""

    def recognize(self) -> SpeakerObservation | None: ...


class GuiBackend(Protocol):
    """Render a scene state through a concrete UI technology."""

    def render(self, scene: SceneState) -> None: ...
