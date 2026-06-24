"""OpenCV-based low-latency preview backend used as the default GUI adapter."""

import cv2

from skynet_local.presentation.modes.registry import build_mode_renderer


class OpenCvWindowBackend:
    """Render scene states in a standard OpenCV window using pluggable presentation modes."""

    def __init__(self, mode_name: str, window_name: str = "Skynet Local") -> None:
        self.window_name = window_name
        self.mode_renderer = build_mode_renderer(mode_name)

    def render(self, scene) -> None:
        """Render the active frame with the current mode renderer and show it in a window."""
        frame = self.mode_renderer.render(scene)
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            scene.should_exit = True
