"""Central place for paths and settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

EMBEDDINGS_DIR = DATA_DIR / "embeddings"
ATTENDANCE_DB = DATA_DIR / "attendance.db"
DETECTOR_MODEL = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = MODEL_DIR / "face_recognition_sface_2021dec.onnx"

CAMERA_INDEX = 0            # change to 1, 2... if you have multiple cameras
DETECTION_THRESHOLD = 0.9   # minimum confidence to count as a face
SAMPLES_PER_PERSON = 5      # how many photos to capture when registering someone
RECOGNITION_THRESHOLD = 0.363  # how similar a face must be to count as a match (0 to 1)
CONFIRM_FRAMES = 5             # frames in a row a face must be recognised before marking present