from skynet_local.domain.entities import FaceObservation
from skynet_local.domain.enums import FaceCategory


class FaceDetectorBase:
    def __init__(self, params) -> None:
        self.smooth_alpha = 0.25
        self.expand_ratio = 0.05
        self._smoothed_boxes: dict[str, tuple[float, float, float, float]] = {}
        self._last_raw_faces_by_track_id: dict[str, object] = {}

    def detect(self, frame) -> list[FaceObservation]:
        pass

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