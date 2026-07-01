"""Thin runtime shell that delegates frame processing to orchestrator and GUI backend.

If an AudioService is provided it is started alongside the frame loop and its
latest SpeakerObservation is merged into each SceneState.
"""

from __future__ import annotations

from skynet_local.application.orchestrator import SceneOrchestrator
from skynet_local.domain.protocols import FrameSource, GuiBackend


class SkynetRuntime:
    """Consume frames, build scene state, optionally pull audio observations and render."""

    def __init__(
        self,
        source: FrameSource,
        orchestrator: SceneOrchestrator,
        guibackend: GuiBackend,
        audio_service=None,  # AudioService | None — optional to avoid hard dependency
    ) -> None:
        self.source = source
        self.orchestrator = orchestrator
        self.guibackend = guibackend
        self._audio = audio_service

    def run(self) -> None:
        """Main loop: capture frame → build scene → merge audio → render."""
        if self._audio is not None:
            self._audio.start()

        try:
            last_key: int | None = None
            for frame in self.source.frames():
                scene = self.orchestrator.handle_frame(frame, last_key=last_key)

                # Merge latest audio observation into the scene (non-blocking read).
                if self._audio is not None:
                    scene.speaker = self._audio.latest_observation()

                self.guibackend.render(scene)
                last_key = getattr(scene, "last_key", None)
                if scene.should_exit:
                    break
        finally:
            if self._audio is not None:
                self._audio.stop()
