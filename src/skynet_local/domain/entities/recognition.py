from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FaceCandidate:
    person_id: str
    display_name: str
    score: float


@dataclass(slots=True)
class FaceRecognitionResult:
    person_id: str | None
    display_name: str | None
    score: float
    is_match: bool
