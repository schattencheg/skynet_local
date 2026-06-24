"""Composition root that wires configuration, services, repositories and GUI backends."""

from pathlib import Path
import cv2

from skynet_local.application.orchestrator import SceneOrchestrator
from skynet_local.application.runtime import SkynetRuntime
from skynet_local.application.services.face_recognition_service import FaceRecognitionService
from skynet_local.config.settings import Settings
from skynet_local.infrastructure.capture.camera_source import CameraSource
from skynet_local.infrastructure.gui.opencv.window_backend import OpenCvWindowBackend
from skynet_local.infrastructure.storage.face_registry import FileFaceRegistry
from skynet_local.infrastructure.storage.identity_repository import IdentityRepository
from skynet_local.infrastructure.vision.detectors.opencv_onnx_detector import OpenCvOnnxFaceDetector
from skynet_local.application.services.unknown_face_enrollment_service import UnknownFaceEnrollmentService


def find_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Project root not found")


def build_face_detector(project_root: Path) -> OpenCvOnnxFaceDetector:
    models_dir = project_root / "models"
    yunet_model = str(models_dir / "face" / "detectors" / "face_detection_yunet_2023mar.onnx")
    sface_model = str(models_dir / "face" / "recognizers" / "face_recognition_sface_2021dec.onnx")
    registry_dir = project_root / "data" / "faces"

    detector_yn = cv2.FaceDetectorYN.create(
        yunet_model,
        "",
        (320, 320),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000,
    )

    recognizer_sf = cv2.FaceRecognizerSF.create(sface_model, "")

    registry = FileFaceRegistry(registry_dir)
    registry.load()

    recognition_service = FaceRecognitionService(
        registry=registry,
        recognizer_sf=recognizer_sf,
        match_threshold=0.363,
        ambiguity_margin=0.04,
    )

    return OpenCvOnnxFaceDetector(
        detector_yn=detector_yn,
        recognition_service=recognition_service,
        score_threshold=0.6,
        smooth_alpha=0.25,
        expand_ratio=0.25,
    )


def build_runtime() -> SkynetRuntime:
    project_root = find_project_root(Path(__file__))
    config_path = project_root / "configs" / "app.yaml"
    settings = Settings.load(config_path)

    source = CameraSource(index=settings.source.camera_index)
    detector = build_face_detector(project_root)
    
    unknown_face_enrollment_service = UnknownFaceEnrollmentService(
        recognition_service=detector.recognition_service,
        detector=detector,
        min_dwell_seconds=2.5,
        ignore_cooldown_seconds=20.0,
        min_face_width=120,
        samples_to_enroll=5,
    )
    repository = IdentityRepository(settings.storage.sqlite_url)

    orchestrator = SceneOrchestrator(
        settings=settings,
        detector=detector,
        repository=repository,
        unknown_face_enrollment_service=unknown_face_enrollment_service,
    )

    guibackend = OpenCvWindowBackend(mode_name=settings.gui.mode)
    return SkynetRuntime(
        source=source,
        orchestrator=orchestrator,
        guibackend=guibackend,
    )
