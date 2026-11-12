"""Milestone 2: live face detection from the webcam using OpenCV's YuNet model."""
import sys

import cv2

from config import CAMERA_INDEX, DETECTION_THRESHOLD, DETECTOR_MODEL


def create_detector(width, height):
    if not DETECTOR_MODEL.exists():
        sys.exit("Model not found. Run: python scripts/download_models.py")
    return cv2.FaceDetectorYN.create(
        str(DETECTOR_MODEL), "", (width, height),
        score_threshold=DETECTION_THRESHOLD, nms_threshold=0.3, top_k=5000,
    )


def draw_faces(frame, faces):
    for face in faces:
        x, y, w, h = face[:4].astype(int)
        score = face[-1]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(frame, f"{score:.2f}", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        # 5 landmarks: right eye, left eye, nose tip, mouth corners
        landmarks = face[4:14].reshape(5, 2).astype(int)
        for (lx, ly) in landmarks:
            cv2.circle(frame, (lx, ly), 2, (255, 0, 0), -1)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        sys.exit("Could not open webcam. Try changing CAMERA_INDEX in config.py")

    ok, frame = cap.read()
    if not ok:
        sys.exit("Could not read from webcam")
    detector = create_detector(frame.shape[1], frame.shape[0])

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = detector.detect(frame)
        count = 0 if faces is None else len(faces)
        if faces is not None:
            draw_faces(frame, faces)

        cv2.putText(frame, f"Faces: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Face detection - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
