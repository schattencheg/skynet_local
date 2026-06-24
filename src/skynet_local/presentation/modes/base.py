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
        """Draw animated face boxes and labels for all detected targets."""
        output = frame.copy()
        h, w = frame.shape[:2]

        active_ids = {face.track_id for face in scene.faces}

        # Убираем аниматоры исчезнувших лиц
        for old_id in list(self._focus_animators):
            if old_id not in active_ids:
                del self._focus_animators[old_id]

        for face in scene.faces:
            # face.bbox сейчас хранит (x1, y1, x2, y2) — конвертируем в (x, y, w, h)
            x1, y1, x2, y2 = face.bbox
            raw_box = (x1, y1, x2 - x1, y2 - y1)

            animator = self._get_animator(face.track_id)
            result = animator.update(raw_box, frame.shape)

            if result is None:
                continue

            ax, ay, aw, ah = result
            # Обратно в (x1, y1, x2, y2) для cv2.rectangle
            cv2.rectangle(output, (ax, ay), (ax + aw, ay + ah), (0, 255, 0), 2)
            cv2.putText(
                output, face.label,
                (ax, max(20, ay - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

        return output
    