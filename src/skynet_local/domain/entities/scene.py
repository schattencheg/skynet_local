from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from skynet_local.domain.enums import GuiMode
from skynet_local.domain.entities.observation import (
    FaceObservation,
    SpeakerObservation,
    IdentityFusionResult,
)


@dataclass(slots=True)
class SceneState:
    """Complete scene model consumed by GUI backends and presentation modes."""

    frame: np.ndarray | None = None
    faces: list[FaceObservation] = field(default_factory=list)
    speaker: SpeakerObservation | None = None
    fusion: IdentityFusionResult | None = None
    messages: list[str] = field(default_factory=list)
    gui_mode: GuiMode = GuiMode.CLASSIC
    should_exit: bool = False
    last_key: int | None = None
    pending_unknown_track_id: str | None = None
    pending_unknown_prompt: str | None = None
    bon_appetit_name: str | None = None
