from dataclasses import dataclass, field
from skynet_local.domain.enums import GuiMode
from skynet_local.domain.entities import FaceObservation, SpeakerObservation
from skynet_local.domain.entities import IdentityFusionResult


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
