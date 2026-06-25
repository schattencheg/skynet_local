"""Base renderer primitives shared by all visual modes."""

import cv2
from skynet_local.presentation.animations.focus_animator import FocusAnimator
from skynet_local.infrastructure.vision.attributes.emotion_analyzer import emotion_to_emoticon


class BaseModeRenderer:
    """Common OpenCV drawing helpers shared by concrete mode renderers."""

    def __init__(self):
        self._focus_animators: dict[str, FocusAnimator] = {}
        # Tracks how long to keep the "Bon appétit!" banner visible (frames)
        self._bon_appetit_frames_left: int = 0
        self._bon_appetit_text: str = ""
        _BON_APPETIT_DURATION_FRAMES = 90  # ~3 s at 30 fps
        self._BON_APPETIT_DURATION = _BON_APPETIT_DURATION_FRAMES

    def _get_animator(self, track_id: str) -> FocusAnimator:
        if track_id not in self._focus_animators:
            self._focus_animators[track_id] = FocusAnimator(focus_duration=0.6)
        return self._focus_animators[track_id]

    @staticmethod
    def format_face_label(face) -> str:
        """Build display label: 'name emoticon' — no chewing indicator in normal label."""
        label = face.label
        emoticon = emotion_to_emoticon(face.emotion)
        if emoticon:
            label += f" {emoticon}"
        return label

    def _draw_bon_appetit(self, frame, scene):
        """Show a large centred 'Bon appétit!' banner when the eating event fires."""
        # Trigger
        name = getattr(scene, "bon_appetit_name", None)
        if name:
            person = name if name != "unknown" else ""
            self._bon_appetit_text = (
                f"Bon appétit, {person}!" if person else "Bon appétit!"
            )
            self._bon_appetit_frames_left = self._BON_APPETIT_DURATION

        if self._bon_appetit_frames_left <= 0:
            return frame

        self._bon_appetit_frames_left -= 1

        h, w = frame.shape[:2]
        font       = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.4
        thickness  = 2
        (tw, th), baseline = cv2.getTextSize(
            self._bon_appetit_text, font, font_scale, thickness
        )
        x = (w - tw) // 2
        y = h // 3

        # Semi-transparent dark pill behind text
        pad = 18
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x - pad, y - th - pad),
            (x + tw + pad, y + baseline + pad),
            (0, 0, 0),
            -1,
        )
        alpha = 0.65
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.putText(
            frame,
            self._bon_appetit_text,
            (x, y),
            font,
            font_scale,
            (0, 230, 120),
            thickness,
            cv2.LINE_AA,
        )
        return frame

    def draw_faces(self, frame, scene):
        output = frame.copy()
        for face in scene.faces:
            x1, y1, x2, y2 = face.bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                output,
                self.format_face_label(face),
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        self._draw_bon_appetit(output, scene)

        prompt = getattr(scene, "pending_unknown_prompt", None)
        if prompt:
            cv2.rectangle(output, (20, 20), (760, 70), (0, 0, 0), -1)
            cv2.rectangle(output, (20, 20), (760, 70), (0, 255, 255), 2)
            cv2.putText(
                output,
                prompt,
                (30, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        return output
