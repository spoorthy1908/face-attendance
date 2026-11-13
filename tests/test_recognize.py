import numpy as np

from recognize import identify


def normalise(rows):
    rows = np.asarray(rows, dtype=float)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_identifies_closest_person():
    names = ["Ammu", "Ravi"]
    known = normalise([[1, 0, 0], [0, 1, 0]])
    name, score = identify(np.array([0.9, 0.1, 0.0]), names, known)
    assert name == "Ammu"
    assert score > 0.9


def test_unknown_when_nobody_is_similar():
    names = ["Ammu", "Ravi"]
    known = normalise([[1, 0, 0], [0, 1, 0]])
    name, _ = identify(np.array([0.0, 0.0, 1.0]), names, known)
    assert name == "Unknown"
