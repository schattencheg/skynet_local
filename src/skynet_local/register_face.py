from __future__ import annotations

from pathlib import Path
import cv2

from skynet_local.application.services.face_recognition_service import FaceRecognitionService
from skynet_local.infrastructure.storage.face_registry import FileFaceRegistry


def main():
    project_root = Path(__file__).resolve().parents[2]
    models_dir = project_root / "models"
    registry_dir = project_root / "data" / "faces"

    yunet_model = str(models_dir / "face" / "detectors" / "face_detection_yunet_2023mar.onnx")
    sface_model = str(models_dir / "face" / "recognizers" / "face_recognition_sface_2021dec.onnx")

    detector = cv2.FaceDetectorYN.create(
        yunet_model,
        "",
        (320, 320),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000,
    )

    recognizer = cv2.FaceRecognizerSF.create(sface_model, "")
    registry = FileFaceRegistry(registry_dir)
    registry.load()

    service = FaceRecognitionService(registry=registry, recognizer_sf=recognizer)

    cap = cv2.VideoCapture(0)
    person_id = input("person_id: ").strip()
    display_name = input("display_name: ").strip()

    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)

        if faces is not None:
            for face_row in faces[:1]:
                x, y, bw, bh = face_row[:4].astype(int)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"Saved: {saved} | Press S to save sample | Q to quit",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Register Face", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if key == ord("s") and faces is not None and len(faces) > 0:
            service.enroll_detection(
                person_id=person_id,
                display_name=display_name,
                frame=frame,
                face_row=faces[0],
                quality=1.0,
            )
            saved += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()