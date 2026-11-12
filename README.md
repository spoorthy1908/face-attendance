# Face Attendance System

A computer vision attendance system that recognises faces from a webcam and logs attendance automatically.

Built with Python and OpenCV (YuNet for face detection, SFace for face recognition).

## Features (roadmap)
- [x] Live webcam face detection
- [ ] Register people with a few photos
- [ ] Real-time face recognition
- [ ] Automatic attendance logging (once per person per day)
- [ ] Web dashboard to view attendance
- [ ] Export reports

## Setup
```bash
git clone https://github.com/<your-username>/face-attendance.git
cd face-attendance
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_models.py
```

## Usage
```bash
python src/detect.py            # live face detection demo (press q to quit)
```

## Privacy
Face data is biometric personal data. Only register people who have given consent.
The `data/` folder is git-ignored so faces and attendance records are never pushed to GitHub.
