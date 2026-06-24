"""SFace-style recognizer stub preserving the architecture without requiring dlib."""

from skynet_local.domain.entities import FaceObservation


class SFaceRecognizer:
    """Return incoming observations unchanged until a real embedding backend is added."""

    def recognize(self, frame, faces: list[FaceObservation]) -> list[FaceObservation]:
        """Pass face observations through while keeping the recognizer interface stable."""
        return faces
