"""Minimal presentation mode with restrained overlays and low visual noise."""

from skynet_local.presentation.modes.base import BaseModeRenderer


class MinimalModeRenderer(BaseModeRenderer):
    """Render only the essential face boxes and labels."""

    def render(self, scene):
        """Return a stripped-down video frame with minimal annotations."""
        return self.draw_faces(scene.frame, scene)
