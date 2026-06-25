"""Main application orchestrator coordinating face processing, fusion and GUI mode state."""

from skynet_local.domain.entities import IdentityFusionResult, SceneState, SpeakerObservation
from skynet_local.domain.enums import GuiMode
from skynet_local.infrastructure.vision.detectors.face_detector_base import FaceDetectorBase
from skynet_local.application.services.unknown_face_enrollment_service import UnknownFaceEnrollmentService
from skynet_local.application.services.face_recognition_service import FaceRecognitionService


class SceneOrchestrator:
    """Coordinate perception results into a unified scene model."""

    def __init__(
        self,
        settings,
        detector,
        repository,
        recognition_service,
        unknown_face_enrollment_service,
    ):
        self.settings = settings
        self.detector = detector
        self.repository = repository
        self.recognition_service = recognition_service
        self.unknown_face_enrollment_service: UnknownFaceEnrollmentService = unknown_face_enrollment_service

    def handle_frame(self, frame, last_key: int | None = None):
        faces = self.detector.detect(frame)

        scene = SceneState(
            frame=frame,
            faces=faces,
            should_exit=False,
        )

        prompt_track_id = self.unknown_face_enrollment_service.update(
            frame=frame,
            faces=faces,
            key=last_key,
        )

        scene.pending_unknown_track_id = prompt_track_id
        scene.pending_unknown_prompt = self.unknown_face_enrollment_service.get_prompt_text()
        scene.last_key = last_key

        return scene    