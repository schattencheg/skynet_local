"""Domain entities describing identities, observations, voice matches and rendered scene state."""

from dataclasses import dataclass, field

from skynet_local.domain.enums import FaceCategory, GuiMode


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


@dataclass(slots=True)
class IdentityFusionResult:
    """Combined decision produced from face identity and speaker identity channels."""

    face_id: str | None
    speaker_id: str | None
    fused_id: str | None
    confidence: float
    conflict: bool = False


@dataclass(slots=True)
class SceneState:
    """Complete scene model consumed by GUI backends and presentation modes."""

    frame: object | None = None
    faces: list[FaceObservation] = field(default_factory=list)
    speaker: SpeakerObservation | None = None
    fusion: IdentityFusionResult | None = None
    messages: list[str] = field(default_factory=list)
    gui_mode: GuiMode = GuiMode.CLASSIC
    should_exit: bool = False
