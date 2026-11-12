"""Download the pre-trained OpenCV face models into models/."""
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = {
    "face_detection_yunet_2023mar.onnx": f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    for name, url in MODELS.items():
        dest = MODEL_DIR / name
        if dest.exists() and dest.stat().st_size > 100_000:
            print(f"[ok] {name} already downloaded")
            continue
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(url, dest)
        size_kb = dest.stat().st_size // 1024
        if size_kb < 100:
            print(f"[warning] {name} is only {size_kb} KB - download may have failed. "
                  f"Try downloading it manually from:\n  {url}")
        else:
            print(f"[ok] saved {name} ({size_kb} KB)")


if __name__ == "__main__":
    main()
