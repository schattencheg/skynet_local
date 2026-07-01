"""Text-to-speech backends."""

from skynet_local.infrastructure.audio.tts.null_tts import NullTTS
from skynet_local.infrastructure.audio.tts.pyttsx3_tts import Pyttsx3TTS

__all__ = ["NullTTS", "Pyttsx3TTS"]
