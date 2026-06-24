from __future__ import annotations

from typing import Any

from skynet_local.domain.entities import FaceObservation
from skynet_local.domain.enums import FaceCategory
from skynet_local.infrastructure.vision.detectors.face_detector_base import FaceDetectorBase


class OpenCvOnnxFaceDetector(FaceDetectorBase):
    """YuNet detector + optional SFace recognition adapter."""

    def __init__(
        self,
        detector_yn,
        recognition_service=None,
        score_threshold: float = 0.6,
        smooth_alpha: float = 0.25,
        expand_ratio: float = 0.25,
    ) -> None:
        self.detector_yn = detector_yn
        self.recognition_service = recognition_service
        self.score_threshold = score_threshold
        self.smooth_alpha = smooth_alpha
        self.expand_ratio = expand_ratio

        self._smoothed_boxes: dict[str, tuple[float, float, float, float]] = {}
        self._last_raw_faces_by_track_id: dict[str, Any] = {}

    def detect(self, frame) -> list[FaceObservation]:
        h, w = frame.shape[:2]
        self.detector_yn.setInputSize((w, h))
        _, faces = self.detector_yn.detect(frame)

        if faces is None:
            self._smoothed_boxes.clear()
            self._last_raw_faces_by_track_id.clear()
            return []

        observations: list[FaceObservation] = []
        active_ids: set[str] = set()
        raw_faces_by_track: dict[str, Any] = {}

        for i, face_row in enumerate(faces):
            x, y, bw, bh = face_row[:4]
            det_score = float(face_row[-1]) if len(face_row) > 14 else 1.0

            if det_score < self.score_threshold:
                continue

            track_id = f"face-{i}"
            active_ids.add(track_id)
            raw_faces_by_track[track_id] = face_row

            raw_box = (float(x), float(y), float(bw), float(bh))
            smoothed_box = self.smooth_box(track_id, raw_box)
            visual_box = self.expand_box(*smoothed_box, frame_shape=frame.shape, ratio=self.expand_ratio)

            vx, vy, vw, vh = visual_box
            x1, y1, x2, y2 = vx, vy, vx + vw, vy + vh

            label = "unknown"
            category = FaceCategory.UNKNOWN
            confidence = det_score
            prompt = "Register face / switch GUI mode / enable diagnostics"

            if self.recognition_service is not None:
                try:
                    result = self.recognition_service.recognize_detection(frame, face_row)
                    if result.is_match:
                        label = result.display_name or result.person_id or "known"
                        category = FaceCategory.KNOWN
                        confidence = result.score
                        prompt = None
                except Exception:
                    pass

            observations.append(
                FaceObservation(
                    track_id=track_id,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    label=label,
                    category=category,
                    confidence=float(confidence),
                    prompt=prompt,
                )
            )

        stale_ids = set(self._smoothed_boxes.keys()) - active_ids
        for stale_id in stale_ids:
            self._smoothed_boxes.pop(stale_id, None)

        self._last_raw_faces_by_track_id = raw_faces_by_track
        return observations

    def get_raw_face_row(self, track_id: str):
        return self._last_raw_faces_by_track_id.get(track_id)

    def smooth_box(
        self,
        track_id: str,
        detected_box: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        current = self._smoothed_boxes.get(track_id)
        if current is None:
            self._smoothed_boxes[track_id] = detected_box
            return detected_box

        smoothed = tuple(
            self.smooth_alpha * d + (1.0 - self.smooth_alpha) * c
            for d, c in zip(detected_box, current)
        )
        self._smoothed_boxes[track_id] = smoothed
        return smoothed

    def expand_box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        frame_shape,
        ratio: float = 0.25,
    ) -> tuple[int, int, int, int]:
        frame_h, frame_w = frame_shape[:2]

        cx = x + w / 2.0
        cy = y + h / 2.0
        nw = w * (1.0 + ratio)
        nh = h * (1.0 + ratio)

        nx = int(round(cx - nw / 2.0))
        ny = int(round(cy - nh / 2.0))
        nw = int(round(nw))
        nh = int(round(nh))

        nx = max(0, nx)
        ny = max(0, ny)
        nw = min(nw, frame_w - nx)
        nh = min(nh, frame_h - ny)

        return nx, ny, nw, nh
