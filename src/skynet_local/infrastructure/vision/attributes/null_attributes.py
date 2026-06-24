"""No-op facial attribute analyzer used when optional attribute models are disabled."""


class NullAttributeAnalyzer:
    """Return observations unchanged without computing age, gender or emotion."""

    def analyze(self, frame, faces):
        """Provide a stable interface for pipelines where attributes are disabled."""
        return faces
