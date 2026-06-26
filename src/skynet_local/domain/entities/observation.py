"""Domain entities for a single observation / detected entity in a frame."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from skynet_local.domain.enums import FaceCategory


@dataclass
class FaceObservation:
    track_id: str
    bbox: tuple[int, int, int, int]
    confidence: float
    label: str = "unknown"
    embedding: np.ndarray | None = None
    emotion: str | None = None
    is_chewing: bool = False
    eating_event: bool = False
    category: str | None = None
    prompt: str | None = None


@dataclass
class SpeakerObservation:
    """Identified speaker segment from voice pipeline."""

    speaker_id: str | None
    confidence: float
    text: str = ""


@dataclass
class IdentityFusionResult:
    """Fused identity after combining face + voice observations."""

    identity_id: str
    display_name: str
    face_confidence: float = 0.0
    voice_confidence: float = 0.0


@dataclass
class SceneState:
    """Snapshot of all observations for a single frame."""

    frame: np.ndarray | None
    faces: list[FaceObservation] = field(default_factory=list)
    speaker: SpeakerObservation | None = None
    should_exit: bool = False
    pending_unknown_track_id: str | None = None
    pending_unknown_prompt: str | None = None
    last_key: int | None = None
    bon_appetit_name: str | None = None
