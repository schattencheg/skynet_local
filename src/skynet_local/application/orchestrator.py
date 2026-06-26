"""Main application orchestrator coordinating face processing, fusion and GUI mode state."""

from __future__ import annotations

import numpy as np

from skynet_local.config.settings import Settings
from skynet_local.domain.entities.scene import SceneState
from skynet_local.domain.scene_tracker import SceneTracker
from skynet_local.infrastructure.vision.detectors.face_detector_base import FaceDetectorBase
from skynet_local.infrastructure.vision.attributes.emotion_analyzer import EmotionAnalyzer
from skynet_local.infrastructure.vision.attributes.chewing_detector import ChewingDetector
from skynet_local.infrastructure.storage.identity_repository import IdentityRepository
from skynet_local.domain.protocols import FaceRecognizer
from skynet_local.application.services.unknown_face_enrollment_service import (
    UnknownFaceEnrollmentService,
)


class SceneOrchestrator:
    """Coordinate perception results into a unified scene model."""

    settings: Settings
    detector: FaceDetectorBase
    repository: IdentityRepository
    recognition_service: FaceRecognizer
    unknown_face_enrollment_service: UnknownFaceEnrollmentService
    _emotion_analyzer: EmotionAnalyzer
    _chewing_detector: ChewingDetector

    def __init__(
        self,
        settings: Settings,
        detector: FaceDetectorBase,
        repository: IdentityRepository,
        recognition_service: FaceRecognizer,
        unknown_face_enrollment_service: UnknownFaceEnrollmentService,
        emotion_analyzer: EmotionAnalyzer | None = None,
        chewing_detector: ChewingDetector | None = None,
    ) -> None:
        self.settings = settings
        self.detector = detector
        self.repository = repository
        self.recognition_service = recognition_service
        self.unknown_face_enrollment_service = unknown_face_enrollment_service
        self._emotion_analyzer = emotion_analyzer if emotion_analyzer is not None else EmotionAnalyzer.null()
        self._chewing_detector = chewing_detector if chewing_detector is not None else ChewingDetector()

    def handle_frame(
        self, frame: np.ndarray, last_key: int | None = None
    ) -> SceneState:
        faces = self.detector.detect(frame)

        raw_rows: dict[str, np.ndarray] = {}
        for f in faces:
            row = self.detector.get_raw_face_row(f.track_id)
            if row is not None:
                raw_rows[f.track_id] = row

        for f in faces:
            self._emotion_analyzer.set_track_landmarks(f.track_id, raw_rows.get(f.track_id))

        faces = self._emotion_analyzer.analyze(frame, faces)
        faces = self._chewing_detector.analyze(frame, faces, raw_face_rows=raw_rows)

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
        scene.pending_unknown_prompt = (
            self.unknown_face_enrollment_service.get_prompt_text()
        )
        scene.last_key = last_key
        SceneTracker.instance().update(
            scene,
            emotion_probs_by_track=getattr(
                self._emotion_analyzer, "_last_probs", {}
            ),
        )

        return scene
