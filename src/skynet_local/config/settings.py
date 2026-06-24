"""Pydantic settings models loaded from YAML configuration files."""

from pathlib import Path

import yaml
from pydantic import BaseModel


class AppConfig(BaseModel):
    """General application metadata and environment flags."""

    name: str = "Skynet Local"
    environment: str = "desktop"


class SourceConfig(BaseModel):
    """Video source selection and source-specific parameters."""

    kind: str = "camera"
    camera_index: int = 0
    video_path: str = ""
    image_path: str = ""


class VisionConfig(BaseModel):
    """Face pipeline backend selection and matching thresholds."""

    detector_backend: str = "opencv_onnx"
    recognizer_backend: str = "sface"
    attributes_backend: str = "none"
    pose_backend: str = "landmarks_pnp"
    detection_interval: int = 1
    matching_threshold: float = 0.45


class VoiceConfig(BaseModel):
    """Speech recognition, speaker recognition and TTS backend configuration."""

    stt_backend: str = "vosk"
    speaker_backend: str = "onnx_speaker"
    tts_backend: str = "none"
    owner_only_commands: bool = True


class GuiConfig(BaseModel):
    """GUI backend and visual mode configuration."""

    backend: str = "opencv"
    mode: str = "classic"
    red_tint_alpha: float = 0.18
    side_panel_width: int = 320


class StorageConfig(BaseModel):
    """Storage layer configuration for SQLite and local artifacts."""

    sqlite_url: str = "sqlite:///data/embeddings/skynet_local.db"


class RuntimeConfig(BaseModel):
    """Runtime fallback and platform behavior toggles."""

    windows_safe: bool = True
    enable_fallbacks: bool = True


class Settings(BaseModel):
    """Root settings model aggregating all sub-configuration sections."""

    app: AppConfig = AppConfig()
    source: SourceConfig = SourceConfig()
    vision: VisionConfig = VisionConfig()
    voice: VoiceConfig = VoiceConfig()
    gui: GuiConfig = GuiConfig()
    storage: StorageConfig = StorageConfig()
    runtime: RuntimeConfig = RuntimeConfig()

    @classmethod
    def load(cls, path: Path) -> "Settings":
        """Load settings from a YAML file and validate them."""
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(data)
