"""MediaPipe HolisticLandmarker 모델 번들을 mediapipe_models/ 에 내려받는다.
requirements.txt 설치 후 한 번만 실행하면 된다.
"""
import urllib.request
from pathlib import Path

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
MODEL_PATH = Path(__file__).resolve().parent / "mediapipe_models" / "holistic_landmarker.task"


def main():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        print(f"이미 존재합니다: {MODEL_PATH}")
        return

    print(f"다운로드 중: {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"저장됨: {MODEL_PATH}")


if __name__ == "__main__":
    main()
