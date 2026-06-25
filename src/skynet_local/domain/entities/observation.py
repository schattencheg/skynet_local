"""Domain entities for a single observation / detected entity in a frame."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from skynet_local.domain.enums import FaceCategory


@dataclass
class FaceObservation:
    track_id: str
    bbox: tuple[int, int, int, int]
    confidence: float
    label: str = "unknown"
    embedding: Optional[object] = None
    emotion: Optional[str] = None
    is_chewing: bool = False
    eating_event: bool = False        # True for one frame when 20-s chewing sustained
    category: Optional[str] = None
    prompt: Optional[str] = None


@dataclass
class SpeakerObservation:
    """Identified speaker segment from voice pipeline."""

    speaker_id: str
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

    frame: object
    faces: list[FaceObservation] = field(default_factory=list)
    speaker: Optional[SpeakerObservation] = None
    should_exit: bool = False
    pending_unknown_track_id: Optional[str] = None
    pending_unknown_prompt: Optional[str] = None
    last_key: Optional[int] = None
    bon_appetit_name: Optional[str] = None   # set by orchestrator when event fires
