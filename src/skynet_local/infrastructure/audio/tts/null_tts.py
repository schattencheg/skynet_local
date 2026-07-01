"""No-op TTS implementation — used when no audio output is desired or available."""


class NullTTS:
    """Silent TTS stub; satisfies the same interface as Pyttsx3TTS."""

    def start(self) -> None:
        """No-op: nothing to initialise."""

    def stop(self) -> None:
        """No-op: nothing to tear down."""

    def speak(self, text: str) -> None:  # noqa: ARG002
        """Silently discard the utterance."""
