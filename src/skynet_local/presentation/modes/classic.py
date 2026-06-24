"""Classic presentation mode with standard overlays and status text."""

import cv2

from skynet_local.presentation.modes.base import BaseModeRenderer


class ClassicModeRenderer(BaseModeRenderer):
    """Render a practical default overlay with boxes, labels and runtime messages."""

    def render(self, scene):
        """Render the classic scene view from the current frame and observations."""
        frame = self.draw_faces(scene.frame, scene)
        y = 30
        for message in scene.messages:
            cv2.putText(frame, message, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y += 30
        return frame
