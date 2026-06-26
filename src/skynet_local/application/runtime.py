"""Thin runtime shell that delegates frame processing to orchestrator and GUI backend."""

from __future__ import annotations

from skynet_local.application.orchestrator import SceneOrchestrator
from skynet_local.domain.protocols import FrameSource, GuiBackend


class SkynetRuntime:
    """Consume frames, build scene state and render it until exit is requested."""

    source: FrameSource
    orchestrator: SceneOrchestrator
    guibackend: GuiBackend

    def __init__(
        self,
        source: FrameSource,
        orchestrator: SceneOrchestrator,
        guibackend: GuiBackend,
    ) -> None:
        self.source = source
        self.orchestrator = orchestrator
        self.guibackend = guibackend

    def run(self) -> None:
        """Consume frames from the source and render the resulting scene."""
        last_key: int | None = None

        for frame in self.source.frames():
            scene = self.orchestrator.handle_frame(frame, last_key=last_key)
            self.guibackend.render(scene)
            last_key = getattr(scene, "last_key", None)
            if scene.should_exit:
                break
