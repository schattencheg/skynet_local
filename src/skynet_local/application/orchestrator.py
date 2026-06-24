"""Main application orchestrator coordinating face processing, fusion and GUI mode state."""

from skynet_local.domain.entities import IdentityFusionResult, SceneState, SpeakerObservation
from skynet_local.domain.enums import GuiMode
from skynet_local.infrastructure.vision.detectors.face_detector_base import FaceDetectorBase


class SceneOrchestrator:
    """Coordinate perception results into a unified scene model."""

    def __init__(self, settings, detector, recognizer, repository) -> None:
        self.settings = settings
        self.detector: FaceDetectorBase = detector
        self.recognizer = recognizer
        self.repository = repository

    def handle_frame(self, frame) -> SceneState:
        """Process one frame and return the scene state for rendering."""
        faces = self.detector.detect(frame)
        speaker = SpeakerObservation(speaker_id=None, label="no-speaker", confidence=0.0, is_owner=False)
        fusion = IdentityFusionResult(face_id=faces[0].label if faces else None, speaker_id=speaker.speaker_id, fused_id=None, confidence=0.0, conflict=False)
        return SceneState(frame=frame, faces=faces, speaker=speaker, fusion=fusion, messages=["Skynet runtime active"], gui_mode=GuiMode(self.settings.gui.mode))
