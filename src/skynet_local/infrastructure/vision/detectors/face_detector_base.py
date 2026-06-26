from __future__ import annotations

import numpy as np

from skynet_local.domain.entities import FaceObservation


class FaceDetectorBase:
    smooth_alpha: float
    expand_ratio: float
    _smoothed_boxes: dict[str, tuple[float, float, float, float]]
    _last_raw_faces_by_track_id: dict[str, np.ndarray]

    def __init__(
        self,
        smooth_alpha: float = 0.25,
        expand_ratio: float = 0.05,
    ) -> None:
        self.smooth_alpha = smooth_alpha
        self.expand_ratio = expand_ratio
        self._smoothed_boxes: dict[str, tuple[float, float, float, float]] = {}
        self._last_raw_faces_by_track_id: dict[str, np.ndarray] = {}

    def detect(self, frame: np.ndarray) -> list[FaceObservation]:  # pragma: no cover
        return []

    def get_raw_face_row(self, track_id: str) -> np.ndarray | None:
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
        smoothed: tuple[float, ...] = tuple(
            self.smooth_alpha * d + (1.0 - self.smooth_alpha) * c
            for d, c in zip(detected_box, current)
        )
        result = (smoothed[0], smoothed[1], smoothed[2], smoothed[3])
        self._smoothed_boxes[track_id] = result
        return result

    def expand_box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        frame_shape: tuple[int, ...],
        ratio: float = 0.25,
    ) -> tuple[int, int, int, int]:
        frame_h, frame_w = frame_shape[0], frame_shape[1]
        cx = x + w / 2.0
        cy = y + h / 2.0
        nw = w * (1.0 + ratio)
        nh = h * (1.0 + ratio)
        nx = int(round(cx - nw / 2.0))
        ny = int(round(cy - nh / 2.0))
        nw_i = int(round(nw))
        nh_i = int(round(nh))
        nx = max(0, nx)
        ny = max(0, ny)
        nw_i = min(nw_i, frame_w - nx)
        nh_i = min(nh_i, frame_h - ny)
        return nx, ny, nw_i, nh_i
