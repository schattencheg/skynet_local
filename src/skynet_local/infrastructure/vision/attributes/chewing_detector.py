"""Chewing detector: triggers a one-shot "Bon appétit!" event after the user
has been chewing continuously for at least EATING_DURATION_SEC seconds.

Design
------
* Per-frame mouth-aspect-ratio (MAR) oscillation detection is preserved as
  the inner signal (same YuNet landmark approach as before).
* `is_chewing` on FaceObservation is still set each frame (for debug display
  if needed), but the meaningful new output is `eating_event: bool` — it is
  True on the single frame the threshold is crossed, then goes False until
  a new eating session starts (cooldown prevents repeated firing).
"""

from __future__ import annotations

import time
from collections import deque


_MAR_OPEN_THRESH: float  = 0.18   # MAR above this → mouth open
_MAR_CLOSE_THRESH: float = 0.10   # MAR below this → mouth closed
_CYCLE_WINDOW: int       = 24     # frames of history for cycle counting
_MIN_CYCLES: int         = 2      # open→close edges to confirm active chewing

# How many seconds of continuous chewing before the event fires
EATING_DURATION_SEC: float = 20.0

# After the event fires, ignore this person for this many seconds
EATING_COOLDOWN_SEC: float = 60.0


class _TrackState:
    __slots__ = ("mar_history", "chewing_since", "last_event_at", "last_active")

    def __init__(self):
        self.mar_history: deque[bool] = deque(maxlen=_CYCLE_WINDOW)
        self.chewing_since: float | None = None   # monotonic time chewing started
        self.last_event_at: float = 0.0           # monotonic time of last event
        self.last_active: float = time.monotonic()


class ChewingDetector:
    """Detect sustained chewing (≥ EATING_DURATION_SEC) and fire a one-shot event."""

    def __init__(
        self,
        eating_duration_sec: float = EATING_DURATION_SEC,
        eating_cooldown_sec: float = EATING_COOLDOWN_SEC,
    ) -> None:
        self._eating_duration = eating_duration_sec
        self._eating_cooldown = eating_cooldown_sec
        self._states: dict[str, _TrackState] = {}

    # ------------------------------------------------------------------
    def analyze(self, frame, faces, raw_face_rows: dict | None = None):
        """Enrich each FaceObservation in-place.

        Sets on each face:
            is_chewing:    bool  — True when mouth-oscillation active this frame.
            eating_event:  bool  — True on the one frame the 20-s threshold is
                                   crossed (caller should trigger "Bon appétit!").
        """
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

            # Count open→close falling edges in recent history
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
                    state.chewing_since = now           # start timing
                elapsed = now - state.chewing_since
                cooldown_expired = (now - state.last_event_at) >= self._eating_cooldown

                if elapsed >= self._eating_duration and cooldown_expired:
                    eating_event = True
                    state.last_event_at = now
                    state.chewing_since = None          # reset for next session
            else:
                # Reset timer when chewing stops
                state.chewing_since = None

            face.eating_event = eating_event

        # Purge stale tracks (gone > 5 s)
        stale = [tid for tid, st in self._states.items()
                 if now - st.last_active > 5.0]
        for tid in stale:
            del self._states[tid]

        return faces

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_mar(face, raw_row) -> float | None:
        if raw_row is not None and len(raw_row) >= 14:
            try:
                mx1, my1 = float(raw_row[10]), float(raw_row[11])
                mx2, my2 = float(raw_row[12]), float(raw_row[13])
                mouth_w = abs(mx2 - mx1) + 1e-6
                mouth_h = abs(my2 - my1)
                return mouth_h / mouth_w
            except (IndexError, ValueError):
                pass
        # Bbox fallback
        x1, y1, x2, y2 = face.bbox
        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)
        return None
        return (bh * 0.15) / bw # Fake MAR
