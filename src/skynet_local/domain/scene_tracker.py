"""Singleton state tracker for the whole scene.

Write path:  SceneTracker.instance().update(scene, emotion_probs_by_track)
Read path:   SceneTracker.instance().persons   → dict[track_id, PersonState]
             SceneTracker.instance().events    → list[SceneEvent] (consumed once)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from skynet_local.domain.entities.scene import SceneState


# ── Event types ──────────────────────────────────────────────────────────────

class EventKind(Enum):
    PERSON_APPEARED   = auto()
    PERSON_LEFT       = auto()
    EMOTION_CHANGED   = auto()
    IDENTITY_RESOLVED = auto()   # label changed from "unknown" to a real name
    BON_APPETIT       = auto()


@dataclass
class SceneEvent:
    kind: EventKind
    track_id: str
    label: str
    detail: str = ""          # emotion name, old→new label, etc.
    timestamp: float = field(default_factory=time.monotonic)


# ── Per-person running state ─────────────────────────────────────────────────

_FERPLUS_LABELS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger",   "disgust",   "fear",     "contempt",
]


@dataclass
class PersonState:
    track_id: str
    label: str = "unknown"

    # Identity
    identity_prob: float = 0.0          # face recognition confidence 0-1

    # Emotion — full distribution (softmax probs, 0-1 each)
    emotion_probs: dict[str, float] = field(
        default_factory=lambda: {e: 0.0 for e in _FERPLUS_LABELS}
    )
    dominant_emotion: Optional[str] = None
    emotion_prob: float = 0.0           # probability of dominant_emotion

    # Chewing / eating
    is_chewing: bool = False
    chewing_since: Optional[float] = None   # monotonic time chewing started
    chewing_duration_sec: float = 0.0       # seconds chewing continuously so far

    # Presence
    first_seen_at: float = field(default_factory=time.monotonic)
    last_seen_at: float  = field(default_factory=time.monotonic)

    @property
    def on_scene_sec(self) -> float:
        return self.last_seen_at - self.first_seen_at


# ── Singleton tracker ────────────────────────────────────────────────────────

class SceneTracker:
    """Thread-safe-enough (GIL) singleton updated every frame by the orchestrator."""

    _instance: Optional["SceneTracker"] = None

    @classmethod
    def instance(cls) -> "SceneTracker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self.persons: dict[str, PersonState] = {}
        self._events: list[SceneEvent] = []

    # ------------------------------------------------------------------
    def update(
        self,
        scene: SceneState,
        emotion_probs_by_track: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """Called by orchestrator every frame with the latest SceneState.

        emotion_probs_by_track: optional mapping track_id → {label: prob}
        produced by EmotionAnalyzer._last_probs (see EmotionAnalyzer change below).
        """
        now = time.monotonic()
        emotion_probs_by_track = emotion_probs_by_track or {}
        active_ids = {f.track_id for f in scene.faces}

        # Detect persons who left
        for tid in list(self.persons):
            if tid not in active_ids:
                p = self.persons.pop(tid)
                self._events.append(SceneEvent(
                    kind=EventKind.PERSON_LEFT,
                    track_id=tid,
                    label=p.label,
                ))

        for face in scene.faces:
            tid = face.track_id
            is_new = tid not in self.persons

            if is_new:
                self.persons[tid] = PersonState(track_id=tid)
                self._events.append(SceneEvent(
                    kind=EventKind.PERSON_APPEARED,
                    track_id=tid,
                    label=face.label,
                ))

            p = self.persons[tid]
            p.last_seen_at = now

            # Identity probability — use face.confidence as proxy;
            # override with 1.0 when label is resolved from "unknown"
            old_label = p.label
            p.label = face.label
            p.identity_prob = float(face.confidence)

            if old_label == "unknown" and face.label != "unknown":
                self._events.append(SceneEvent(
                    kind=EventKind.IDENTITY_RESOLVED,
                    track_id=tid,
                    label=face.label,
                    detail=f"was unknown → {face.label}",
                ))

            # Emotion distribution
            probs = emotion_probs_by_track.get(tid)
            if probs:
                p.emotion_probs.update(probs)

            old_emotion = p.dominant_emotion
            p.dominant_emotion = face.emotion
            p.emotion_prob = float(
                p.emotion_probs.get(face.emotion, 0.0) if face.emotion else 0.0
            )

            if face.emotion and face.emotion != old_emotion and not is_new:
                self._events.append(SceneEvent(
                    kind=EventKind.EMOTION_CHANGED,
                    track_id=tid,
                    label=face.label,
                    detail=f"{old_emotion} → {face.emotion}",
                ))

            # Chewing / eating
            p.is_chewing = face.is_chewing
            if face.is_chewing:
                if p.chewing_since is None:
                    p.chewing_since = now
                p.chewing_duration_sec = now - p.chewing_since
            else:
                p.chewing_since = None
                p.chewing_duration_sec = 0.0

            if getattr(face, "eating_event", False):
                self._events.append(SceneEvent(
                    kind=EventKind.BON_APPETIT,
                    track_id=tid,
                    label=face.label,
                ))

    # ------------------------------------------------------------------
    def consume_events(self) -> list[SceneEvent]:
        """Return and clear the pending event queue (call once per frame)."""
        evts, self._events = self._events, []
        return evts

    def peek_events(self) -> list[SceneEvent]:
        """Return pending events without clearing (for read-only inspection)."""
        return list(self._events)