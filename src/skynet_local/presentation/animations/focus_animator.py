"""Smooth focus animation: interpolates bbox from screen edges to detected face."""

import time
from enum import Enum


def _lerp_box(a, b, t):
    return [a[i] + (b[i] - a[i]) * t for i in range(4)]


def _ease_out_cubic(t):
    return 1 - (1 - t) ** 3


def _smooth_ema(current, target, alpha=0.25):
    if current is None:
        return list(target)
    return [alpha * t + (1 - alpha) * c for c, t in zip(current, target)]


class FocusState(Enum):
    SEARCHING = "searching"
    FOCUSING  = "focusing"
    LOCKED    = "locked"


class FocusAnimator:
    """Animate a bounding box from screen edges toward the detected face."""

    def __init__(self, focus_duration: float = 0.6, smooth_alpha: float = 0.25):
        self.focus_duration = focus_duration
        self.smooth_alpha = smooth_alpha
        self.state = FocusState.SEARCHING
        self._focus_start: float | None = None
        self._animated_box: list | None = None   # текущий отображаемый box [x,y,w,h]
        self._smoothed_raw: list | None = None   # EMA поверх сырых данных детектора

    def update(self, raw_box, frame_shape) -> tuple[int, int, int, int] | None:
        """
        raw_box: (x, y, w, h) от детектора или None если лицо не найдено.
        frame_shape: (h, w, ...) — shape кадра.
        Возвращает (x, y, w, h) для отрисовки или None.
        """
        h, w = frame_shape[:2]
        full_box = [0, 0, w, h]

        if raw_box is None:
            self.state = FocusState.SEARCHING
            if self._animated_box is not None:
                # Плавно расширяемся обратно к границам экрана
                self._animated_box = _lerp_box(self._animated_box, full_box, 0.05)
            return None  # не рисуем квадрат пока нет лица

        # EMA-сглаживание сырого bbox
        self._smoothed_raw = _smooth_ema(self._smoothed_raw, list(raw_box), self.smooth_alpha)
        target = self._smoothed_raw

        if self.state == FocusState.SEARCHING:
            self.state = FocusState.FOCUSING
            self._focus_start = time.monotonic()
            self._animated_box = list(full_box)  # стартуем от экрана

        if self.state == FocusState.FOCUSING:
            elapsed = time.monotonic() - self._focus_start
            t = min(elapsed / self.focus_duration, 1.0)
            ease_t = _ease_out_cubic(t)
            self._animated_box = _lerp_box(self._animated_box, target, ease_t)
            if t >= 1.0:
                self.state = FocusState.LOCKED

        elif self.state == FocusState.LOCKED:
            # Мягко следим за лицом каждый кадр
            self._animated_box = _lerp_box(self._animated_box, target, self.smooth_alpha)

        x, y, bw, bh = [int(v) for v in self._animated_box]
        return x, y, bw, bh
