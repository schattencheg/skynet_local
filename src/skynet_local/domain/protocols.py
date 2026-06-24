"""Protocols defining the pluggable contracts for detectors, recognizers and GUI adapters."""

from typing import Iterable, Protocol

from skynet_local.domain.entities import FaceObservation, SceneState, SpeakerObservation


class FrameSource(Protocol):
    """Produce frames from a camera, video file or still image source."""

    def frames(self) -> Iterable[object]: ...


class FaceDetector(Protocol):
    """Locate faces in a frame and return preliminary observations."""

    def detect(self, frame: object) -> list[FaceObservation]: ...


class FaceRecognizer(Protocol):
    """Enrich face observations with identity and confidence information."""

    def recognize(self, frame: object, faces: list[FaceObservation]) -> list[FaceObservation]: ...


class SpeakerRecognizer(Protocol):
    """Infer the currently active speaker identity from audio input."""

    def recognize(self) -> SpeakerObservation | None: ...


class GuiBackend(Protocol):
    """Render a scene state through a concrete UI technology."""

    def render(self, scene: SceneState) -> None: ...
