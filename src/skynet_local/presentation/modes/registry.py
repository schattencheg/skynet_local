"""Factory for constructing presentation mode renderers by configuration name."""

from __future__ import annotations

from skynet_local.presentation.modes.base import BaseModeRenderer
from skynet_local.presentation.modes.classic import ClassicModeRenderer
from skynet_local.presentation.modes.diagnostic import DiagnosticModeRenderer
from skynet_local.presentation.modes.minimal import MinimalModeRenderer
from skynet_local.presentation.modes.terminator import TerminatorModeRenderer


def build_mode_renderer(mode_name: str) -> BaseModeRenderer:
    """Return the renderer implementation for the requested mode name."""
    mapping: dict[str, type[BaseModeRenderer]] = {
        "classic":    ClassicModeRenderer,
        "minimal":    MinimalModeRenderer,
        "diagnostic": DiagnosticModeRenderer,
        "terminator": TerminatorModeRenderer,
    }
    renderer_cls = mapping.get(mode_name, ClassicModeRenderer)
    return renderer_cls()
