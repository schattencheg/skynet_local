"""Enumerations used across identity, GUI and runtime state models."""

from enum import Enum


class FaceCategory(str, Enum):
    """Categories assigned to detected or registered faces."""

    KNOWN = "known"
    ENEMY = "enemy"
    UNKNOWN = "unknown"


class GuiMode(str, Enum):
    """Supported presentation modes for the active scene."""

    CLASSIC = "classic"
    MINIMAL = "minimal"
    DIAGNOSTIC = "diagnostic"
    TERMINATOR = "terminator"
