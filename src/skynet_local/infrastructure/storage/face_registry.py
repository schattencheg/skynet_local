from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

from skynet_local.domain.entities import FaceCandidate, FaceIdentity, FaceSample


class FileFaceRegistry:
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.registry_file = self.base_dir / "registry.json"
        self.embeddings_dir = self.base_dir / "embeddings"
        self._identities: dict[str, FaceIdentity] = {}

    def load(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        if not self.registry_file.exists():
            self._identities = {}
            return

        payload = json.loads(self.registry_file.read_text(encoding="utf-8"))
        identities: dict[str, FaceIdentity] = {}

        for person in payload.get("people", []):
            person_id = person["person_id"]
            npz_path = self.embeddings_dir / f"{person_id}.npz"

            samples = []
            prototype = None

            if npz_path.exists():
                data = np.load(npz_path, allow_pickle=True)

                embeddings = data["embeddings"] if "embeddings" in data else np.array([])
                qualities = data["qualities"] if "qualities" in data else np.array([])
                sample_ids = data["sample_ids"] if "sample_ids" in data else np.array([], dtype=object)
                created_ats = data["created_ats"] if "created_ats" in data else np.array([], dtype=object)
                prototype_raw = data["prototype"] if "prototype" in data else np.array([])

                prototype = prototype_raw if prototype_raw.size else None

                for i in range(len(embeddings)):
                    samples.append(
                        FaceSample(
                            sample_id=str(sample_ids[i]),
                            embedding=np.asarray(embeddings[i], dtype=np.float32),
                            quality=float(qualities[i]),
                            created_at=datetime.fromisoformat(str(created_ats[i])),
                        )
                    )

            identities[person_id] = FaceIdentity(
                person_id=person_id,
                display_name=person["display_name"],
                samples=samples,
                prototype=prototype,
                created_at=datetime.fromisoformat(person["created_at"]) if person.get("created_at") else None,
                updated_at=datetime.fromisoformat(person["updated_at"]) if person.get("updated_at") else None,
            )

        self._identities = identities

    def save(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        people = []
        for identity in self._identities.values():
            people.append(
                {
                    "person_id": identity.person_id,
                    "display_name": identity.display_name,
                    "created_at": identity.created_at.isoformat() if identity.created_at else None,
                    "updated_at": identity.updated_at.isoformat() if identity.updated_at else None,
                    "samples": len(identity.samples),
                }
            )

            np.savez_compressed(
                self.embeddings_dir / f"{identity.person_id}.npz",
                embeddings=np.array([s.embedding for s in identity.samples], dtype=np.float32),
                qualities=np.array([s.quality for s in identity.samples], dtype=np.float32),
                sample_ids=np.array([s.sample_id for s in identity.samples], dtype=object),
                created_ats=np.array([s.created_at.isoformat() for s in identity.samples], dtype=object),
                prototype=identity.prototype if identity.prototype is not None else np.array([]),
            )

        self.registry_file.write_text(
            json.dumps({"version": 1, "people": people}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add_identity(self, person_id: str, display_name: str) -> FaceIdentity:
        if person_id in self._identities:
            return self._identities[person_id]

        now = datetime.utcnow()
        identity = FaceIdentity(
            person_id=person_id,
            display_name=display_name,
            created_at=now,
            updated_at=now,
        )
        self._identities[person_id] = identity
        return identity

    def get_identity(self, person_id: str) -> FaceIdentity | None:
        return self._identities.get(person_id)

    def list_identities(self) -> list[FaceIdentity]:
        return list(self._identities.values())

    def add_sample(
        self,
        person_id: str,
        embedding: np.ndarray,
        quality: float,
    ) -> FaceSample:
        identity = self.get_identity(person_id)
        if identity is None:
            raise ValueError(f"Unknown identity: {person_id}")

        sample = FaceSample(
            sample_id=str(uuid.uuid4()),
            embedding=self._normalize(embedding),
            quality=quality,
            created_at=datetime.utcnow(),
        )
        identity.samples.append(sample)
        identity.prototype = self._build_prototype(identity.samples)
        identity.updated_at = datetime.utcnow()
        return sample

    def find_top_candidates(
        self,
        embedding: np.ndarray,
        limit: int = 3,
    ) -> list[FaceCandidate]:
        query = self._normalize(embedding)
        candidates = []

        for identity in self._identities.values():
            if identity.prototype is None:
                continue
            score = float(np.dot(query, identity.prototype))
            candidates.append(
                FaceCandidate(
                    person_id=identity.person_id,
                    display_name=identity.display_name,
                    score=score,
                )
            )

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:limit]

    def _build_prototype(self, samples: list[FaceSample]) -> np.ndarray | None:
        if not samples:
            return None
        matrix = np.stack([self._normalize(s.embedding) for s in samples], axis=0)
        proto = matrix.mean(axis=0)
        return self._normalize(proto)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(arr)
        return arr if norm == 0 else arr / norm
