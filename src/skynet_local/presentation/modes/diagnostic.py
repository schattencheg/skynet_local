"""Diagnostic mode for future confidence plots, pose stats and audio-state overlays."""

import cv2

from skynet_local.presentation.modes.base import BaseModeRenderer


class DiagnosticModeRenderer(BaseModeRenderer):
    """Render engineering-focused metrics on top of the live frame."""

    def render(self, scene):
        """Draw the scene and append a diagnostic footer block."""
        frame = self.draw_faces(scene.frame, scene)
        cv2.putText(frame, "DIAGNOSTIC MODE", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame
