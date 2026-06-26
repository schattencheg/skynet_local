"""Windows-safe OpenCV/ONNX face detector stub used as the default detector backend."""

from skynet_local.domain.entities import FaceObservation
from skynet_local.domain.enums import FaceCategory


class OpenCvOnnxFaceDetectorStub(FaceDetectorBase):
    """Return placeholder face detections until a real ONNX model is connected."""

    def detect(self, frame) -> list[FaceObservation]:
        """Produce one demo observation centered in the frame for scaffold validation."""
        h, w = frame.shape[:2]
        return [
            FaceObservation(
                track_id="demo-face-1",
                bbox=(w // 4, h // 4, w // 2, h // 2),
                label="unknown",
                category=FaceCategory.UNKNOWN,
                confidence=0.0,
                prompt="Register face / switch GUI mode / enable diagnostics",
            )
        ]
