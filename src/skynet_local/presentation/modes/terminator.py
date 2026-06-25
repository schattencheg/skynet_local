"""Terminator-style cinematic mode with red tint, side panel and HUD placeholders."""

import cv2

from skynet_local.presentation.modes.base import BaseModeRenderer


class TerminatorModeRenderer(BaseModeRenderer):
    """Render a red-tinted HUD with side panel, target metadata and scanline feel."""

    def render(self, scene):
        """Compose the Terminator-inspired overlay from the current scene state."""
        base = self.draw_faces(scene.frame, scene)
        overlay = base.copy()
        overlay[:, :, 2] = cv2.add(overlay[:, :, 2], 60)
        frame = cv2.addWeighted(overlay, 0.25, base, 0.75, 0)
        h, w = frame.shape[:2]
        panel_x = max(w - 320, 0)
        cv2.rectangle(frame, (panel_x, 0), (w, h), (0, 0, 80), -1)
        cv2.putText(frame, "TARGET ANALYSIS", (panel_x + 16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y = 70
        if scene.faces:
            face = scene.faces[0]
            from skynet_local.infrastructure.vision.attributes.emotion_analyzer import emotion_to_emoticon
            emotion_str = face.emotion or "unknown"
            emoticon = emotion_to_emoticon(face.emotion)
            if emoticon:
                emotion_str = f"{emotion_str} {emoticon}"
            chew_str = "YES" if getattr(face, "is_chewing", False) else "NO"
            for line in [
                f"ID: {face.label}",
                f"EMOTION: {emotion_str}",
                f"CHEWING: {chew_str}",
                f"AGE: {face.age if face.age is not None else 'n/a'}",
                f"GENDER: {face.gender or 'n/a'}",
                f"YAW: {face.yaw if face.yaw is not None else 'n/a'}",
            ]:
                cv2.putText(frame, line, (panel_x + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 255), 2)
                y += 28
        for scan_y in range(0, h, 4):
            cv2.line(frame, (0, scan_y), (w, scan_y), (0, 0, 40), 1)
        cv2.putText(frame, "TERMINATOR MODE", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 255), 2)
        return frame
