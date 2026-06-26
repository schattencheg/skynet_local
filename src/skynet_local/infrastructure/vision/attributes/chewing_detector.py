from __future__ import annotations

import time
from collections import deque

import numpy as np

from skynet_local.domain.entities import FaceObservation

_MAR_OPEN_THRESH: float  = 0.18
_MAR_CLOSE_THRESH: float = 0.10
_CYCLE_WINDOW: int       = 24
_MIN_CYCLES: int         = 2

EATING_DURATION_SEC: float = 20.0
EATING_COOLDOWN_SEC: float = 60.0


class _TrackState:
    __slots__ = ("mar_history", "chewing_since", "last_event_at", "last_active")

    mar_history: deque[bool]
    chewing_since: float | None
    last_event_at: float
    last_active: float

    def __init__(self) -> None:
        self.mar_history: deque[bool] = deque(maxlen=_CYCLE_WINDOW)
        self.chewing_since: float | None = None
        self.last_event_at: float = 0.0
        self.last_active: float = time.monotonic()


class ChewingDetector:
    """Detect sustained chewing (>= EATING_DURATION_SEC) and fire a one-shot event."""

    _eating_duration: float
    _eating_cooldown: float
    _states: dict[str, _TrackState]

    def __init__(
        self,
        eating_duration_sec: float = EATING_DURATION_SEC,
        eating_cooldown_sec: float = EATING_COOLDOWN_SEC,
    ) -> None:
        self._eating_duration = eating_duration_sec
        self._eating_cooldown = eating_cooldown_sec
        self._states: dict[str, _TrackState] = {}

    def analyze(
        self,
        frame: np.ndarray,
        faces: list[FaceObservation],
        raw_face_rows: dict[str, np.ndarray] | None = None,
    ) -> list[FaceObservation]:
        raw_face_rows = raw_face_rows or {}
        now = time.monotonic()

        for face in faces:
            mar = self._compute_mar(face, raw_face_rows.get(face.track_id))
            if mar is None:
                face.is_chewing = False
                face.eating_event = False
                continue

            state = self._states.setdefault(face.track_id, _TrackState())
            state.last_active = now

            is_open = mar > _MAR_OPEN_THRESH
            state.mar_history.append(is_open)

            cycles, prev = 0, None
            for s in state.mar_history:
                if prev is True and s is False:
                    cycles += 1
                prev = s

            actively_chewing = cycles >= _MIN_CYCLES
            face.is_chewing = actively_chewing

            eating_event = False

            if actively_chewing:
                if state.chewing_since is None:
                    state.chewing_since = now
                elapsed = now - state.chewing_since
                cooldown_expired = (now - state.last_event_at) >= self._eating_cooldown

                if elapsed >= self._eating_duration and cooldown_expired:
                    eating_event = True
                    state.last_event_at = now
                    state.chewing_since = None
            else:
                state.chewing_since = None

            face.eating_event = eating_event

        stale = [
            tid
            for tid, st in self._states.items()
            if now - st.last_active > 5.0
        ]
        for tid in stale:
            del self._states[tid]

        return faces

    @staticmethod
    def _compute_mar(face: FaceObservation, raw_row: np.ndarray | None) -> float | None:
        if raw_row is not None and len(raw_row) >= 14:
            try:
                mx1, my1 = float(raw_row[10]), float(raw_row[11])
                mx2, my2 = float(raw_row[12]), float(raw_row[13])
                mouth_w = abs(mx2 - mx1) + 1e-6
                mouth_h = abs(my2 - my1)
                return mouth_h / mouth_w
            except (IndexError, ValueError):
                pass
        return None
