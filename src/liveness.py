"""Milestone 7: a simple liveness check to stop people using a photo.

Idea: when a real person turns their head, their nose moves sideways relative
to their eyes, because a face is 3D. A flat photo can be moved or tilted, but
the nose stays in the same relative position between the eyes.
"""
import time

from config import LIVENESS_TIMEOUT, TURN_HIGH, TURN_LOW


def head_turn_ratio(face):
    """Where the nose sits between the eyes: about 0.5 when facing the camera,
    lower when turned one way, higher when turned the other way.

    YuNet face row: [x, y, w, h, right_eye_x, right_eye_y, left_eye_x, left_eye_y,
                     nose_x, nose_y, mouth_right_x, mouth_right_y, mouth_left_x, mouth_left_y, score]
    """
    right_eye_x, left_eye_x, nose_x = face[4], face[6], face[8]
    eye_gap = left_eye_x - right_eye_x
    if abs(eye_gap) < 1e-6:
        return 0.5
    return float((nose_x - right_eye_x) / eye_gap)


class LivenessTracker:
    """Remembers, for each person, whether they have turned their head both ways."""

    def __init__(self):
        self.state = {}

    def update(self, name, face):
        """Record this frame for a person. Returns (is_live, ratio)."""
        now = time.time()
        person = self.state.get(name)
        # Start again if we haven't seen this person for a while
        if person is None or now - person["last_seen"] > LIVENESS_TIMEOUT:
            person = {"turned_low": False, "turned_high": False}

        ratio = head_turn_ratio(face)
        person["turned_low"] = person["turned_low"] or ratio < TURN_LOW
        person["turned_high"] = person["turned_high"] or ratio > TURN_HIGH
        person["last_seen"] = now
        self.state[name] = person
        return person["turned_low"] and person["turned_high"], ratio
