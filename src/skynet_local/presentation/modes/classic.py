"""Classic presentation mode with standard overlays and status text."""

from __future__ import annotations

import cv2
import numpy as np

from skynet_local.domain.entities.scene import SceneState
from skynet_local.presentation.modes.base import BaseModeRenderer


class ClassicModeRenderer(BaseModeRenderer):
    """Render a practical default overlay with boxes, labels and runtime messages."""

    def render(self, scene: SceneState) -> np.ndarray:
        frame: np.ndarray = self.draw_faces(scene.frame, scene)
        y = 30
        for message in scene.messages:
            cv2.putText(
                frame, message, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )
            y += 30
        return frame
