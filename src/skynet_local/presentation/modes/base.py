"""Base renderer primitives shared by all visual modes."""

import cv2
from skynet_local.presentation.animations.focus_animator import FocusAnimator


class BaseModeRenderer:
    """Common OpenCV drawing helpers shared by concrete mode renderers."""

    def __init__(self):
        # Один аниматор на каждый рендерер (хранит состояние между кадрами)
        self._focus_animators: dict[str, FocusAnimator] = {}

    def _get_animator(self, track_id: str) -> FocusAnimator:
        """Получить или создать аниматор для конкретного трека лица."""
        if track_id not in self._focus_animators:
            self._focus_animators[track_id] = FocusAnimator(focus_duration=0.6)
        return self._focus_animators[track_id]
    
    def draw_faces(self, frame, scene):
        output = frame.copy()
        for face in scene.faces:
            x1, y1, x2, y2 = face.bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                output,
                face.label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

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
    