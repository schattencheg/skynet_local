"""No-op voice activity detector for scaffolding and tests."""


class NullVoiceActivityDetector:
    """Always report no active speech until a real VAD backend is connected."""

    def is_active(self) -> bool:
        """Return false for every polling cycle."""
        return False
