"""OpenCV-based camera frame source."""

from collections.abc import Iterator

import cv2
import numpy as np


class CameraSource:
    """Yield frames from a webcam or virtual camera device."""

    def __init__(self, index: int = 0) -> None:
        self.index = index

    def frames(self) -> Iterator[np.ndarray]:
        """Open the configured camera and yield frames until capture stops."""
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.index}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield cv2.flip(frame, 1) # Flip to make it mirrored instead of classic camera
        finally:
            cap.release()
