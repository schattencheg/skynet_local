"""Offline speech recognition stub built around the Vosk backend contract."""


class VoskSpeechRecognizer:
    """Future Vosk adapter for local speech-to-text processing."""

    def poll_text(self) -> str:
        """Return an empty string until microphone streaming is implemented."""
        return ""
