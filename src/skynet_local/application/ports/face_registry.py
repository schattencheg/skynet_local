from __future__ import annotations

from typing import Protocol
import numpy as np

from skynet_local.domain.entities import (
    FaceCandidate,
    FaceIdentity,
    FaceSample,
)


class FaceRegistryPort(Protocol):
    def load(self) -> None: ...
    def save(self) -> None: ...

    def add_identity(self, person_id: str, display_name: str) -> FaceIdentity: ...
    def get_identity(self, person_id: str) -> FaceIdentity | None: ...
    def list_identities(self) -> list[FaceIdentity]: ...
    def remove_identity(self, person_id: str) -> None: ...

    def add_sample(
        self,
        person_id: str,
        embedding: np.ndarray,
        quality: float,
        source: str | None = None,
        image_path: str | None = None,
    ) -> FaceSample: ...

    def rebuild_prototype(self, person_id: str) -> None: ...

    def find_top_candidates(
        self,
        embedding: np.ndarray,
        limit: int = 3,
    ) -> list[FaceCandidate]: ...