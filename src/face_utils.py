"""Shared helpers for the camera, face detection and face recognition."""
import sys

import cv2
import numpy as np

from config import CAMERA_INDEX, DETECTION_THRESHOLD, DETECTOR_MODEL, RECOGNIZER_MODEL


def open_camera():
    """Open the webcam and return (camera, first_frame)."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        sys.exit("Could not open webcam. Check camera permissions or CAMERA_INDEX in config.py")
    # The first few frames can fail while the camera warms up, so try a few times
    for _ in range(10):
        ok, frame = cap.read()
        if ok:
            return cap, frame
    sys.exit("Could not read from webcam")


def create_detector(width, height):
    """Load the YuNet face detector."""
    if not DETECTOR_MODEL.exists():
        sys.exit("Detector model not found. Run: python scripts/download_models.py")
    return cv2.FaceDetectorYN.create(
        str(DETECTOR_MODEL), "", (width, height),
        score_threshold=DETECTION_THRESHOLD, nms_threshold=0.3, top_k=5000,
    )


def create_recognizer():
    """Load the SFace face recognizer."""
    if not RECOGNIZER_MODEL.exists():
        sys.exit("Recognizer model not found. Run: python scripts/download_models.py")
    return cv2.FaceRecognizerSF.create(str(RECOGNIZER_MODEL), "")


def detect_faces(detector, frame):
    """Return a list of detected faces (each row: box, 5 landmarks, score)."""
    detector.setInputSize((frame.shape[1], frame.shape[0]))
    _, faces = detector.detect(frame)
    return [] if faces is None else faces


def get_embedding(recognizer, frame, face):
    """Align the face and turn it into a 128-number embedding."""
    aligned = recognizer.alignCrop(frame, face)
    return recognizer.feature(aligned).flatten()
