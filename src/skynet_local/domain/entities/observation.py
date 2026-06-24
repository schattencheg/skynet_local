from dataclasses import dataclass
from skynet_local.domain.enums import FaceCategory


@dataclass(slots=True)
class FaceObservation:
    """Face detection and recognition result for one target in the current frame."""

    track_id: str
    bbox: tuple[int, int, int, int]
    label: str
    category: FaceCategory
    confidence: float = 0.0
    emotion: str | None = None
    age: int | None = None
    gender: str | None = None
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    prompt: str | None = None


@dataclass(slots=True)
class SpeakerObservation:
    """Voice biometric result for the current active speaker."""

    speaker_id: str | None
    label: str
    confidence: float = 0.0
    is_owner: bool = False
