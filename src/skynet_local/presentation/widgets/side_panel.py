"""Side-panel widget placeholder used by advanced GUI modes such as Terminator mode."""


class SidePanelWidget:
    """Represent a reusable metadata panel for recognized targets."""

    def build_lines(self, face) -> list[str]:
        """Convert a face observation into textual rows for panel rendering."""
        return [face.label]
