"""No-op TTS backend used when voice synthesis is disabled."""


class NullTtsEngine:
    """Ignore spoken text requests while preserving the TTS interface."""

    def say(self, text: str) -> None:
        """Accept text without performing speech synthesis."""
        return None
