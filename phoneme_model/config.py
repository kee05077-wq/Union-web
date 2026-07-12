from pathlib import Path

# Union-web(app.py)에 배포된 gesture_classifier.h5와 동일한 클래스 순서/인덱싱
ACTIONS = [
    'ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
    'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
    'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ',
]

SEQ_LENGTH = 10  # Union-web과 동일 (10프레임 시퀀스)

BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "dataset" / "videos"
SEQUENCE_DIR = BASE_DIR / "dataset" / "sequences"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "phoneme_model.h5"
