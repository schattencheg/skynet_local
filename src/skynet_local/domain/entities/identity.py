from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


@dataclass(slots=True)
class FaceSample:
    sample_id: str
    embedding: np.ndarray
    quality: float
    created_at: datetime
    source: str | None = None
    image_path: str | None = None


@dataclass(slots=True)
class FaceIdentity:
    person_id: str
    display_name: str
    samples: list[FaceSample] = field(default_factory=list[FaceSample])
    prototype: np.ndarray | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def add_sample(self, sample: FaceSample) -> None:
        self.samples.append(sample)
        self.updated_at = datetime.now()

    @property
    def sample_count(self) -> int:
        return len(self.samples)


@dataclass(slots=True)
class IdentityFusionResult:
    """Combined decision produced from face identity and speaker identity channels."""

    face_id: str | None
    speaker_id: str | None
    fused_id: str | None
    confidence: float
    conflict: bool = False
