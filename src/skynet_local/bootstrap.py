"""Composition root — wires configuration, services, repositories and GUI/audio backends."""

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
from skynet_local.infrastructure.vision.attributes import EmotionAnalyzer
from skynet_local.infrastructure.vision.attributes import LandmarkEmotionDetector
from skynet_local.infrastructure.vision.attributes import EnsembleEmotionDetector
from skynet_local.infrastructure.vision.attributes import ChewingDetector
from skynet_local.infrastructure.vision.attributes import FerplusEmotionDetector
from skynet_local.infrastructure.vision.attributes import FerplusEmotionDetectorCalibrated

# Audio is optional — if vosk / sounddevice are not installed the app still runs.
try:
    from skynet_local.infrastructure.audio.audio_service import AudioService
    _AUDIO_AVAILABLE = True
except ImportError:
    _AUDIO_AVAILABLE = False


def find_project_root(start: Path) -> Path:
    """Walk up the directory tree until pyproject.toml is found."""
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Project root not found")


def build_face_detector(project_root: Path) -> OpenCvOnnxFaceDetector:
    """Construct the YuNet detector + SFace recogniser from model files."""
    models_dir = project_root / "models"
    yunet_model = str(models_dir / "face" / "detectors" / "face_detection_yunet_2026may.onnx")
    sface_model = str(models_dir / "face" / "recognizers" / "face_recognition_sface_2021dec.onnx")
    registry_dir = project_root / "data" / "faces"

    detector_yn = cv2.FaceDetectorYN.create(
        yunet_model, "", (320, 320),
        score_threshold=0.6, nms_threshold=0.3, top_k=5000,
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


def build_audio_service(project_root: Path, settings: Settings) -> "AudioService | None":
    """Build AudioService when dependencies are available; return None otherwise."""
    if not _AUDIO_AVAILABLE:
        print("[bootstrap] audio deps not installed — running without voice pipeline")
        return None

    # Look for a Vosk model directory; skip STT if absent rather than crashing.
    vosk_model_path = project_root / "models" / "audio" / "vosk-model-small-ru-0.22"
    if not vosk_model_path.exists():
        print(f"[bootstrap] Vosk model not found at {vosk_model_path} — STT disabled")
        vosk_model_path = None  # type: ignore[assignment]

    voiceprints_dir = project_root / "data" / "voiceprints"

    return AudioService(
        vosk_model_path=vosk_model_path,
        voiceprints_dir=voiceprints_dir,
        tts_rate=getattr(settings, "tts_rate", 175),
        enable_tts=True,
    )


def build_runtime() -> SkynetRuntime:
    """Assemble and return the fully wired SkynetRuntime."""
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

    emotion_model_path = project_root / settings.vision.emotion_model_path
    orchestrator = SceneOrchestrator(
        settings=settings,
        detector=detector,
        repository=repository,
        recognition_service=detector.recognition_service,
        unknown_face_enrollment_service=unknown_face_enrollment_service,
        emotion_analyzer=EmotionAnalyzer(
            detector=EnsembleEmotionDetector([
                (FerplusEmotionDetectorCalibrated(emotion_model_path, min_prob=settings.vision.emotion_min_prob), 0.65),
                (LandmarkEmotionDetector(), 0.35),
            ]),
            n_det=settings.vision.emotion_n_det,
            n_cooldown=settings.vision.emotion_n_cooldown,
        ),
        chewing_detector=ChewingDetector(),
    )

    audio_service = build_audio_service(project_root, settings)

    guibackend = OpenCvWindowBackend(mode_name=settings.gui.mode)
    return SkynetRuntime(
        source=source,
        orchestrator=orchestrator,
        guibackend=guibackend,
        audio_service=audio_service,
    )
