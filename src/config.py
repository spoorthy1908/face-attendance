"""Central place for paths and settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

DETECTOR_MODEL = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = MODEL_DIR / "face_recognition_sface_2021dec.onnx"

CAMERA_INDEX = 0            # change to 1, 2... if you have multiple cameras
DETECTION_THRESHOLD = 0.9   # minimum confidence to count as a face
