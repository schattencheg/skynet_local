"""Smooth focus animation: interpolates bbox from screen edges to detected face."""

from __future__ import annotations

import time
from enum import Enum


def _lerp_box(
    a: list[float], b: list[float], t: float
) -> list[float]:
    return [a[i] + (b[i] - a[i]) * t for i in range(4)]


def _ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


def _smooth_ema(
    current: list[float] | None,
    target: list[float],
    alpha: float = 0.25,
) -> list[float]:
    if current is None:
        return list(target)
    return [alpha * t + (1 - alpha) * c for c, t in zip(current, target)]


class FocusState(Enum):
    SEARCHING = "searching"
    FOCUSING  = "focusing"
    LOCKED    = "locked"


class FocusAnimator:
    """Animate a bounding box from screen edges toward the detected face."""

    focus_duration: float
    smooth_alpha: float
    state: FocusState
    _focus_start: float | None
    _animated_box: list[float] | None
    _smoothed_raw: list[float] | None

    def __init__(
        self, focus_duration: float = 0.6, smooth_alpha: float = 0.25
    ) -> None:
        self.focus_duration = focus_duration
        self.smooth_alpha = smooth_alpha
        self.state = FocusState.SEARCHING
        self._focus_start: float | None = None
        self._animated_box: list[float] | None = None
        self._smoothed_raw: list[float] | None = None

    def update(
        self,
        raw_box: tuple[int, int, int, int] | None,
        frame_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int] | None:
        h, w = frame_shape[0], frame_shape[1]
        full_box: list[float] = [0.0, 0.0, float(w), float(h)]

        if raw_box is None:
            self.state = FocusState.SEARCHING
            if self._animated_box is not None:
                self._animated_box = _lerp_box(self._animated_box, full_box, 0.05)
            return None

        self._smoothed_raw = _smooth_ema(
            self._smoothed_raw, [float(v) for v in raw_box], self.smooth_alpha
        )
        target = self._smoothed_raw

        if self.state == FocusState.SEARCHING:
            self.state = FocusState.FOCUSING
            self._focus_start = time.monotonic()
            self._animated_box = list(full_box)

        if self.state == FocusState.FOCUSING:
            assert self._focus_start is not None
            elapsed = time.monotonic() - self._focus_start
            t = min(elapsed / self.focus_duration, 1.0)
            ease_t = _ease_out_cubic(t)
            self._animated_box = _lerp_box(self._animated_box or full_box, target, ease_t)
            if t >= 1.0:
                self.state = FocusState.LOCKED

        elif self.state == FocusState.LOCKED:
            self._animated_box = _lerp_box(self._animated_box or full_box, target, self.smooth_alpha)

        assert self._animated_box is not None
        x, y, bw, bh = [int(v) for v in self._animated_box]
        return x, y, bw, bh
