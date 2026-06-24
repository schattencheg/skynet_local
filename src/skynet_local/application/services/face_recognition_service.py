from __future__ import annotations

import numpy as np

from skynet_local.domain.entities import FaceRecognitionResult


class FaceRecognitionService:
    def __init__(
        self,
        registry,
        recognizer_sf,
        match_threshold: float = 0.363,
        ambiguity_margin: float = 0.04,
    ) -> None:
        self.registry = registry
        self.recognizer_sf = recognizer_sf
        self.match_threshold = match_threshold
        self.ambiguity_margin = ambiguity_margin

    def align_face(self, frame, face_row):
        return self.recognizer_sf.alignCrop(frame, face_row)

    def extract_embedding(self, aligned_face) -> np.ndarray:
        embedding = self.recognizer_sf.feature(aligned_face)
        return self._normalize_embedding(embedding)

    def recognize_detection(self, frame, face_row) -> FaceRecognitionResult:
        aligned_face = self.align_face(frame, face_row)
        embedding = self.extract_embedding(aligned_face)

        candidates = self.registry.find_top_candidates(embedding, limit=2)
        if not candidates:
            return FaceRecognitionResult(
                person_id=None,
                display_name=None,
                score=0.0,
                is_match=False,
            )

        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None

        if best.score < self.match_threshold:
            return FaceRecognitionResult(
                person_id=None,
                display_name=None,
                score=best.score,
                is_match=False,
            )

        if second and (best.score - second.score) < self.ambiguity_margin:
            return FaceRecognitionResult(
                person_id=None,
                display_name=None,
                score=best.score,
                is_match=False,
            )

        return FaceRecognitionResult(
            person_id=best.person_id,
            display_name=best.display_name,
            score=best.score,
            is_match=True,
        )

    def enroll_detection(
        self,
        person_id: str,
        display_name: str,
        frame,
        face_row,
        quality: float = 1.0,
    ) -> None:
        identity = self.registry.get_identity(person_id)
        if identity is None:
            self.registry.add_identity(person_id, display_name)

        aligned_face = self.align_face(frame, face_row)
        embedding = self.extract_embedding(aligned_face)
        self.registry.add_sample(
            person_id=person_id,
            embedding=embedding,
            quality=quality,
        )
        self.registry.save()

    @staticmethod
    def _normalize_embedding(vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(arr)
        return arr if norm == 0 else arr / norm
