"""Fusion service combining face identity and speaker identity into one confidence model."""

from skynet_local.domain.entities import IdentityFusionResult


class IdentityFusionService:
    """Resolve a combined identity from independent face and voice observations."""

    def fuse(self, face_id: str | None, speaker_id: str | None) -> IdentityFusionResult:
        """Return a basic fusion result using simple equality-based confidence rules."""
        same = face_id is not None and face_id == speaker_id
        return IdentityFusionResult(face_id=face_id, speaker_id=speaker_id, fused_id=face_id if same else None, confidence=1.0 if same else 0.0, conflict=face_id is not None and speaker_id is not None and face_id != speaker_id)
