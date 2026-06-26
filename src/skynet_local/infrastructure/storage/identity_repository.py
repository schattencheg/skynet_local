"""SQLite repository skeleton for face and voice identity profiles."""

from pathlib import Path


class IdentityRepository:
    """Store and retrieve registered identities and embeddings from SQLite."""

    sqlite_url: str
    db_path: Path

    def __init__(self, sqlite_url: str) -> None:
        self.sqlite_url = sqlite_url
        self.db_path = Path(sqlite_url.replace("sqlite:///", ""))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def save_face_profile(self, profile_id: str, name: str) -> None:
        """Persist a face profile placeholder for future implementation."""
        return None
