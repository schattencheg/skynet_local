"""Base renderer primitives shared by all visual modes."""

import cv2


class BaseModeRenderer:
    """Common OpenCV drawing helpers shared by concrete mode renderers."""

    def draw_faces(self, frame, scene):
        """Draw common face boxes and labels for all detected targets."""
        output = frame.copy()
        for face in scene.faces:
            x1, y1, x2, y2 = face.bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output, face.label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return output
