import numpy as np

from liveness import LivenessTracker, head_turn_ratio


def make_face(nose_x):
    """A fake YuNet face row with eyes at x=100 and x=200."""
    return np.array([0, 0, 0, 0, 100, 100, 200, 100, nose_x, 150, 0, 0, 0, 0, 0.9])


def test_facing_forward_gives_half():
    assert head_turn_ratio(make_face(150)) == 0.5


def test_turning_changes_ratio():
    assert head_turn_ratio(make_face(120)) < 0.35
    assert head_turn_ratio(make_face(180)) > 0.65


def test_live_only_after_turning_both_ways():
    tracker = LivenessTracker()
    assert tracker.update("Ammu", make_face(150))[0] is False
    assert tracker.update("Ammu", make_face(120))[0] is False
    assert tracker.update("Ammu", make_face(180))[0] is True


def test_still_photo_never_passes():
    tracker = LivenessTracker()
    for _ in range(50):
        is_live, _ = tracker.update("Photo", make_face(150))
    assert is_live is False
