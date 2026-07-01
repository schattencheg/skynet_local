"""Audio infrastructure: microphone capture, STT, TTS, speaker recognition, VAD."""

from skynet_local.infrastructure.audio.audio_service import AudioService
from skynet_local.infrastructure.audio.microphone import MicrophoneStream

__all__ = ["AudioService", "MicrophoneStream"]
