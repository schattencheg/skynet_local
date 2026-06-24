"""Identifier generation helpers for profiles, tracks and snapshots."""

from uuid import uuid4


def new_id() -> str:
    """Return a new hex identifier suitable for local entities and artifacts."""
    return uuid4().hex
