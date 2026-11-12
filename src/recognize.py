"""Milestone 4: recognise registered people from the webcam in real time.

Usage:  python src/recognize.py
Press q to quit.
"""
import sys

import cv2
import numpy as np

from config import EMBEDDINGS_DIR, RECOGNITION_THRESHOLD
from face_utils import create_detector, create_recognizer, detect_faces, get_embedding, open_camera

GREEN, RED = (0, 200, 0), (0, 0, 255)


def load_known_faces():
    """Load every saved embedding. Returns (names, embeddings) with one name per row."""
    names, all_embeddings = [], []
    for path in sorted(EMBEDDINGS_DIR.glob("*.npy")):
        samples = np.load(path)
        names.extend([path.stem] * len(samples))
        all_embeddings.append(samples)

    if not all_embeddings:
        sys.exit("No one is registered yet. Run: python src/register.py")

    known = np.vstack(all_embeddings)
    # Normalise each row to length 1, so a dot product equals cosine similarity
    known = known / np.linalg.norm(known, axis=1, keepdims=True)
    return names, known


def identify(embedding, names, known):
    """Return (name, score) of the best match, or ("Unknown", score)."""
    embedding = embedding / np.linalg.norm(embedding)
    scores = known @ embedding          # cosine similarity with every saved sample
    best = int(np.argmax(scores))
    score = float(scores[best])
    if score >= RECOGNITION_THRESHOLD:
        return names[best], score
    return "Unknown", score


def main():
    names, known = load_known_faces()
    print(f"Loaded {len(set(names))} registered people: {', '.join(sorted(set(names)))}")

    cap, frame = open_camera()
    detector = create_detector(frame.shape[1], frame.shape[0])
    recognizer = create_recognizer()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        for face in detect_faces(detector, frame):
            name, score = identify(get_embedding(recognizer, frame, face), names, known)
            colour = GREEN if name != "Unknown" else RED
            x, y, w, h = face[:4].astype(int)
            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
            label = f"{name.replace('_', ' ')} ({score:.2f})"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

        cv2.imshow("Face recognition - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
