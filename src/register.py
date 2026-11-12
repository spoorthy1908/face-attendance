"""Milestone 3: register a person by capturing face samples from the webcam.

Usage:  python src/register.py
Press SPACE to capture a sample, q to cancel.
"""
import re
import sys

import cv2
import numpy as np

from config import EMBEDDINGS_DIR, SAMPLES_PER_PERSON
from face_utils import create_detector, create_recognizer, detect_faces, get_embedding, open_camera

GREEN, RED, ORANGE = (0, 200, 0), (0, 0, 255), (0, 165, 255)


def clean_name(name):
    """Keep letters, numbers, spaces, - and _, and turn spaces into underscores."""
    name = re.sub(r"[^A-Za-z0-9 _-]", "", name).strip()
    return name.replace(" ", "_")


def main():
    name = clean_name(input("Enter the person's name: "))
    if not name:
        sys.exit("Name can't be empty.")

    save_path = EMBEDDINGS_DIR / f"{name}.npy"
    if save_path.exists():
        answer = input(f"{name} is already registered. Overwrite? (y/n): ")
        if answer.strip().lower() != "y":
            sys.exit("Cancelled.")

    cap, frame = open_camera()
    detector = create_detector(frame.shape[1], frame.shape[0])
    recognizer = create_recognizer()
    samples = []

    print("Click the video window, then press SPACE to capture. Move your head slightly between captures.")

    while len(samples) < SAMPLES_PER_PERSON:
        ok, frame = cap.read()
        if not ok:
            break
        faces = detect_faces(detector, frame)
        display = frame.copy()

        for face in faces:
            x, y, w, h = face[:4].astype(int)
            cv2.rectangle(display, (x, y), (x + w, y + h), GREEN, 2)

        if len(faces) == 1:
            message, colour = "Press SPACE to capture", GREEN
        elif len(faces) == 0:
            message, colour = "No face found", RED
        else:
            message, colour = "Only one person in view please", ORANGE

        cv2.putText(display, f"{name}: {len(samples)}/{SAMPLES_PER_PERSON}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
        cv2.putText(display, message, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        cv2.imshow("Register face - SPACE to capture, q to cancel", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and len(faces) == 1:
            samples.append(get_embedding(recognizer, frame, faces[0]))
            print(f"Captured sample {len(samples)}/{SAMPLES_PER_PERSON}")

    cap.release()
    cv2.destroyAllWindows()

    if len(samples) < SAMPLES_PER_PERSON:
        sys.exit("Registration cancelled - nothing was saved.")

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(save_path, np.vstack(samples))
    print(f"Saved {len(samples)} face samples for {name} to {save_path}")


if __name__ == "__main__":
    main()
