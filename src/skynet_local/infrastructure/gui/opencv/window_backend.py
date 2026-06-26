"""OpenCV-based low-latency preview backend used as the default GUI adapter."""

from __future__ import annotations

import cv2
import numpy as np

from skynet_local.domain.entities.scene import SceneState
from skynet_local.presentation.modes.base import BaseModeRenderer
from skynet_local.presentation.modes.registry import build_mode_renderer


class OpenCvWindowBackend:
    """Render scene states in a standard OpenCV window using pluggable presentation modes."""

    window_name: str
    mode_renderer: BaseModeRenderer

    def __init__(self, mode_name: str, window_name: str = "Skynet Local") -> None:
        self.window_name = window_name
        self.mode_renderer = build_mode_renderer(mode_name)

    def render(self, scene: SceneState) -> None:
        """Render the active frame with the current mode renderer and show it in a window."""
        rendered: np.ndarray = self.mode_renderer.render(scene)
        cv2.imshow(self.window_name, rendered)

        key = cv2.waitKey(1) & 0xFF
        scene.last_key = None if key == 255 else key

        if key == ord("q"):
            scene.should_exit = True
