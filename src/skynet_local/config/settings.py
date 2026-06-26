"""Pydantic settings models loaded from YAML configuration files.

Each class below mirrors a top-level section in configs/app.yaml.
When adding a new parameter:
  1. Add a field + inline comment here.
  2. Add the matching key + comment in app.yaml.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel


class AppConfig(BaseModel):
    """General application metadata and environment flags."""

    name: str = "Skynet Local"
    """Human-readable application name shown in the title bar."""

    environment: str = "desktop"
    """Deployment context: 'desktop' | 'headless' | 'embedded'."""


class SourceConfig(BaseModel):
    """Video source selection and source-specific parameters."""

    kind: str = "camera"
    """Input source type: 'camera' | 'video' | 'image'."""

    camera_index: int = 0
    """OS camera index used when kind='camera' (0 = default webcam)."""

    video_path: str = ""
    """Absolute or relative path to a video file (used when kind='video')."""

    image_path: str = ""
    """Absolute or relative path to a still image (used when kind='image')."""


class VisionConfig(BaseModel):
    """Face pipeline backend selection and matching thresholds."""

    detector_backend: str = "opencv_onnx"
    """Face detector implementation: 'opencv_onnx' | 'mediapipe' | 'mock'."""

    detector_model_path: str = "models/face/detectors/face_detection_yunet_2026may.onnx"
    """Path to the ONNX face-detector weights file."""

    detector_score_threshold: float = 0.9
    """Minimum detection confidence to keep a face box (0–1).
    Lower values increase recall but also false positives."""

    detector_nms_threshold: float = 0.3
    """IoU threshold for Non-Maximum Suppression.
    Lower removes more overlapping boxes."""

    detector_top_k: int = 5000
    """Maximum number of face candidates kept before NMS.
    Increase for crowded scenes."""

    recognizer_backend: str = "sface"
    """Face-embedding backend: 'sface' | 'arcface' | 'mock'."""

    attributes_backend: str = "none"
    """Attribute-analysis backend: 'none' | 'onnx'."""

    emotion_model_path: str = "models/face/attributes/emotion_ferplus_8.onnx"
    """Path to the ONNX emotion-classification model (FER+ or compatible)."""

    emotion_n_det: int = 2
    """Number of consecutive detections required before an emotion is accepted as stable."""

    emotion_n_cooldown: int = 20
    """Number of frames an emotion result is held after the face disappears (prevents flicker)."""

    emotion_min_prob: float = 0.23
    """Minimum softmax probability to report an emotion.
    Tune this down if emotions are frequently missed."""

    pose_backend: str = "landmarks_pnp"
    """Head-pose estimation method: 'landmarks_pnp' | 'none'."""

    detection_interval: int = 1
    """Run full detector every N frames.
    1 = every frame; 2 = every other frame (halves CPU cost)."""

    matching_threshold: float = 0.45
    """Cosine-similarity threshold for face-identity match (0–1).
    Higher = stricter; lower = more permissive."""


class VoiceConfig(BaseModel):
    """Speech recognition, speaker recognition and TTS backend configuration."""

    stt_backend: str = "vosk"
    """Speech-to-text engine: 'vosk' | 'whisper' | 'none'."""

    speaker_backend: str = "onnx_speaker"
    """Speaker-identification backend: 'onnx_speaker' | 'none'."""

    tts_backend: str = "none"
    """Text-to-speech engine: 'pyttsx3' | 'coqui' | 'none'."""

    owner_only_commands: bool = True
    """When True, voice commands are processed only for the registered owner."""


class GuiConfig(BaseModel):
    """GUI backend and visual mode configuration."""

    backend: str = "opencv"
    """Rendering backend: 'opencv' | 'qt' | 'mock'."""

    mode: str = "classic"
    """Visual display mode: 'classic' | 'terminator' | 'minimal'."""

    red_tint_alpha: float = 0.18
    """Opacity of the red Terminator overlay tint (0.0–1.0)."""

    side_panel_width: int = 320
    """Width of the identity / info side panel in pixels."""


class StorageConfig(BaseModel):
    """Storage layer configuration for SQLite and local artifacts."""

    sqlite_url: str = "sqlite:///data/embeddings/skynet_local.db"
    """SQLAlchemy-compatible URL for the face-embedding database."""


class RuntimeConfig(BaseModel):
    """Runtime fallback and platform behavior toggles."""

    windows_safe: bool = True
    """When True, disables features known to be unstable on Windows
    (e.g. multiprocessing fork start method)."""

    enable_fallbacks: bool = True
    """When True, missing optional backends fall back to no-ops instead of raising errors."""


class Settings(BaseModel):
    """Root settings model aggregating all sub-configuration sections."""

    app: AppConfig = AppConfig()
    """General application metadata — see AppConfig."""

    source: SourceConfig = SourceConfig()
    """Input source selection — see SourceConfig."""

    vision: VisionConfig = VisionConfig()
    """Face pipeline and vision settings — see VisionConfig."""

    voice: VoiceConfig = VoiceConfig()
    """Voice and audio pipeline settings — see VoiceConfig."""

    gui: GuiConfig = GuiConfig()
    """GUI backend and display mode — see GuiConfig."""

    storage: StorageConfig = StorageConfig()
    """Persistence and storage settings — see StorageConfig."""

    runtime: RuntimeConfig = RuntimeConfig()
    """Runtime platform tweaks — see RuntimeConfig."""

    @classmethod
    def load(cls, path: Path) -> "Settings":
        """Load settings from a YAML file and validate them."""
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(data)
