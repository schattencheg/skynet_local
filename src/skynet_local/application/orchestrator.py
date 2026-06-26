"""Main application orchestrator coordinating face processing, fusion and GUI mode state."""

from __future__ import annotations

import numpy as np

from skynet_local.config.settings import Settings
from skynet_local.domain.entities import SceneState
from skynet_local.infrastructure.vision.detectors.face_detector_base import FaceDetectorBase
from skynet_local.application.services.unknown_face_enrollment_service import UnknownFaceEnrollmentService
from skynet_local.application.services.face_recognition_service import FaceRecognitionService
from skynet_local.infrastructure.vision.attributes.emotion_analyzer import EmotionAnalyzer
from skynet_local.infrastructure.vision.attributes.chewing_detector import ChewingDetector
from skynet_local.domain.scene_tracker import SceneTracker
from skynet_local.infrastructure.storage.identity_repository import IdentityRepository


class SceneOrchestrator:
    """Coordinate perception results into a unified scene model."""

    settings: Settings
    detector: FaceDetectorBase
    repository: IdentityRepository
    recognition_service: FaceRecognitionService
    unknown_face_enrollment_service: UnknownFaceEnrollmentService
    _emotion_analyzer: EmotionAnalyzer
    _chewing_detector: ChewingDetector

    def __init__(
        self,
        settings: Settings,
        detector: FaceDetectorBase,
        repository: IdentityRepository,
        recognition_service: FaceRecognitionService,
        unknown_face_enrollment_service: UnknownFaceEnrollmentService,
        emotion_analyzer: EmotionAnalyzer | None = None,
        chewing_detector: ChewingDetector | None = None,
    ) -> None:
        self.settings = settings
        self.detector = detector
        self.repository = repository
        self.recognition_service = recognition_service
        self.unknown_face_enrollment_service = unknown_face_enrollment_service
        self._emotion_analyzer = emotion_analyzer or EmotionAnalyzer(model_path="")
        self._chewing_detector = chewing_detector or ChewingDetector()

    def handle_frame(self, frame: np.ndarray, last_key: int | None = None) -> SceneState:
        faces = self.detector.detect(frame)

        # Build raw landmark rows first
        raw_rows: dict[str, object] = {}
        if hasattr(self.detector, "get_raw_face_row"):
            for f in faces:
                row = self.detector.get_raw_face_row(f.track_id)
                if row is not None:
                    raw_rows[f.track_id] = row

        # Feed landmarks to any landmark-aware emotion backend
        if hasattr(self._emotion_analyzer._detector, "set_landmarks"):
            for f in faces:
                self._emotion_analyzer._detector.set_landmarks(f.track_id, raw_rows.get(f.track_id))

        faces = self._emotion_analyzer.analyze(frame, faces)
        faces = self._chewing_detector.analyze(frame, faces, raw_face_rows=raw_rows)
        # Collect "Bon appétit!" event — pick first person eating (usually only one)
        bon_appetit_name: str | None = None
        for f in faces:
            if getattr(f, "eating_event", False):
                bon_appetit_name = f.label
                break

        scene = SceneState(
            frame=frame,
            faces=faces,
            should_exit=False,
            bon_appetit_name=bon_appetit_name,
        )

        prompt_track_id = self.unknown_face_enrollment_service.update(
            frame=frame,
            faces=faces,
            key=last_key,
        )

        scene.pending_unknown_track_id = prompt_track_id
        scene.pending_unknown_prompt = self.unknown_face_enrollment_service.get_prompt_text()
        scene.last_key = last_key
        # Update scene state
        SceneTracker.instance().update(
            scene,
            emotion_probs_by_track=getattr(self._emotion_analyzer, "_last_probs", {}),
        )

        return scene
