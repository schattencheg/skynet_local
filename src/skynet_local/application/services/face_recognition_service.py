from __future__ import annotations

from typing import Any

import numpy as np

from skynet_local.domain.entities import FaceRecognitionResult
from skynet_local.application.services.cooldown_service import CooldownService


class FaceRecognitionService:
    registry: Any
    recognizer_sf: Any
    match_threshold: float
    ambiguity_margin: float
    auto_update_threshold: float
    auto_update_cooldown_seconds: int
    _update_cooldown: CooldownService

    def __init__(
        self,
        registry: Any,
        recognizer_sf: Any,
        match_threshold: float = 0.363,
        ambiguity_margin: float = 0.04,
        auto_update_threshold: float = 0.50,
        auto_update_cooldown_seconds: int = 30,
    ) -> None:
        self.registry = registry
        self.recognizer_sf = recognizer_sf
        self.match_threshold = match_threshold
        self.ambiguity_margin = ambiguity_margin
        self.auto_update_threshold = auto_update_threshold
        self.auto_update_cooldown_seconds = auto_update_cooldown_seconds
        self._update_cooldown = CooldownService()

    def align_face(self, frame: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        return self.recognizer_sf.alignCrop(frame, face_row)

    def extract_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        embedding = self.recognizer_sf.feature(aligned_face)
        return self._normalize_embedding(embedding)

    def recognize_detection(self, frame: np.ndarray, face_row: np.ndarray) -> FaceRecognitionResult:
        aligned_face = self.align_face(frame, face_row)
        embedding = self.extract_embedding(aligned_face)

        candidates = self.registry.find_top_candidates(embedding, limit=2)
        if not candidates:
            return FaceRecognitionResult(
                person_id=None, display_name=None, score=0.0, is_match=False,
            )

        best   = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None

        if best.score < self.match_threshold:
            return FaceRecognitionResult(
                person_id=None, display_name=None, score=best.score, is_match=False,
            )

        if second and (best.score - second.score) < self.ambiguity_margin:
            return FaceRecognitionResult(
                person_id=None, display_name=None, score=best.score, is_match=False,
            )

        result = FaceRecognitionResult(
            person_id=best.person_id,
            display_name=best.display_name,
            score=best.score,
            is_match=True,
        )

        # ── Adaptive template update, gated by per-identity cooldown ──────────
        if self._update_cooldown.allowed(
            key=best.person_id,
            cooldown_seconds=self.auto_update_cooldown_seconds,
        ):
            second_score = second.score if second else None
            try:
                quality = float(face_row[2]) * float(face_row[3])   # w * h
            except (TypeError, IndexError):
                quality = 0.0

            self.registry.try_adaptive_update(
                person_id=best.person_id,
                embedding=embedding,
                quality=quality,
                score=best.score,
                second_score=second_score,
                auto_update_threshold=self.auto_update_threshold,
            )

        return result

    def enroll_detection(
        self,
        person_id: str,
        display_name: str,
        frame: np.ndarray,
        face_row: np.ndarray,
        quality: float = 1.0,
    ) -> None:
        if self.registry.get_identity(person_id) is None:
            self.registry.add_identity(person_id, display_name)

        aligned_face = self.align_face(frame, face_row)
        embedding = self.extract_embedding(aligned_face)
        self.registry.add_sample(
            person_id=person_id,
            embedding=embedding,
            quality=quality,
        )
        self.registry.save()   # always force-save after manual enroll

    @staticmethod
    def _normalize_embedding(vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(arr)
        return arr if norm == 0 else arr / norm
