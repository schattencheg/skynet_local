"""Composition root that wires configuration, services, repositories and GUI backends."""

from pathlib import Path

from skynet_local.application.orchestrator import SceneOrchestrator
from skynet_local.config.settings import Settings
from skynet_local.infrastructure.capture.camera_source import CameraSource
from skynet_local.infrastructure.gui.opencv.window_backend import OpenCvWindowBackend
from skynet_local.infrastructure.storage.identity_repository import IdentityRepository
from skynet_local.infrastructure.vision.detectors.opencv_onnx_detector import OpenCvOnnxFaceDetector
from skynet_local.infrastructure.vision.recognizers.sface_recognizer import SFaceRecognizer


class SkynetRuntime:
    """Thin runtime shell that delegates processing to orchestrator and GUI backend."""

    def __init__(self, source, orchestrator, gui_backend) -> None:
        self.source = source
        self.orchestrator = orchestrator
        self.gui_backend = gui_backend

    def run(self) -> None:
        """Consume frames from the source and render the resulting scene."""
        for frame in self.source.frames():
            scene = self.orchestrator.handle_frame(frame)
            self.gui_backend.render(scene)
            if scene.should_exit:
                break

def find_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Project root not found")

def build_runtime() -> SkynetRuntime:
    """Create the default Windows-safe runtime graph."""
    PROJECT_ROOT = find_project_root(Path(__file__))
    CONFIG_PATH = PROJECT_ROOT / "configs" / "app.yaml"
    settings = Settings.load(CONFIG_PATH)
    source = CameraSource(index=settings.source.camera_index)
    detector = OpenCvOnnxFaceDetector()
    recognizer = SFaceRecognizer()
    repository = IdentityRepository(settings.storage.sqlite_url)
    orchestrator = SceneOrchestrator(settings=settings, detector=detector, recognizer=recognizer, repository=repository)
    gui_backend = OpenCvWindowBackend(mode_name=settings.gui.mode)
    return SkynetRuntime(source=source, orchestrator=orchestrator, gui_backend=gui_backend)
