"""Speaker-recognition stub for future ONNX-based voiceprint matching."""

from skynet_local.domain.entities import SpeakerObservation


class OnnxSpeakerRecognizer:
    """Return a neutral speaker observation until voice embedding models are added."""

    def recognize(self) -> SpeakerObservation:
        """Provide a default empty speaker observation for integration tests."""
        return SpeakerObservation(speaker_id=None, label="no-speaker", confidence=0.0, is_owner=False)
