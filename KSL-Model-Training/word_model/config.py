from pathlib import Path

# 예시 단어 목록 (필요에 따라 자유롭게 추가/수정)
ACTIONS = ['나는', '맛있다', '고양이', '식사']

SEQ_LENGTH = 30       # 단어 하나를 표현하는 데 필요한 프레임 수
FEATURE_DIM = 150     # pose 6점(x,y,z,visibility)=24 + 왼손 21x3=63 + 오른손 21x3=63

BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "dataset" / "videos"
SEQUENCE_DIR = BASE_DIR / "dataset" / "sequences"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "word_model.h5"
