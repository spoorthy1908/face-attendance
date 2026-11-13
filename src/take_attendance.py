"""Run the webcam and automatically mark recognised people present.

A person is marked present only when:
  1. they are recognised in CONFIRM_FRAMES frames in a row, and
  2. they pass the liveness check by turning their head side to side.

Usage:  python src/take_attendance.py
Press q to quit.
"""
from datetime import date, datetime

import cv2

from attendance import get_attendance, mark_present
from config import CONFIRM_FRAMES, LIVENESS_CHECK
from face_utils import create_detector, create_recognizer, detect_faces, get_embedding, open_camera
from liveness import LivenessTracker
from recognize import identify, load_known_faces

GREEN, RED, ORANGE = (0, 200, 0), (0, 0, 255), (0, 165, 255)


def main():
    names, known = load_known_faces()
    present = {name for name, _ in get_attendance(date.today().isoformat())}
    streak = {}   # how many frames in a row each person has been recognised
    liveness = LivenessTracker()
    print(f"{len(present)} already marked present today. Press q in the video window to stop.")
    if LIVENESS_CHECK:
        print("Liveness check is on: people must turn their head side to side.")

    cap, frame = open_camera()
    detector = create_detector(frame.shape[1], frame.shape[0])
    recognizer = create_recognizer()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        seen_this_frame = set()
        for face in detect_faces(detector, frame):
            name, score = identify(get_embedding(recognizer, frame, face), names, known)
            display_name = name.replace("_", " ")

            if name == "Unknown":
                label, colour = "Unknown", RED
            elif name in present:
                label, colour = f"{display_name} - present", GREEN
            else:
                seen_this_frame.add(name)
                streak[name] = streak.get(name, 0) + 1
                if LIVENESS_CHECK:
                    is_live, ratio = liveness.update(name, face)
                else:
                    is_live, ratio = True, 0.5

                if streak[name] >= CONFIRM_FRAMES and is_live:
                    if mark_present(name):
                        print(f"[{datetime.now():%H:%M:%S}] {display_name} marked present")
                    present.add(name)
                    label, colour = f"{display_name} - present", GREEN
                elif not is_live:
                    label, colour = f"{display_name} - turn head side to side ({ratio:.2f})", ORANGE
                else:
                    label, colour = f"{display_name} - checking...", ORANGE

            x, y, w, h = face[:4].astype(int)
            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

        # Anyone not seen in this frame has to start their streak again
        for name in streak:
            if name not in seen_this_frame:
                streak[name] = 0

        cv2.putText(frame, f"Present today: {len(present)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
        cv2.imshow("Attendance - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. {len(present)} people present today.")


if __name__ == "__main__":
    main()
