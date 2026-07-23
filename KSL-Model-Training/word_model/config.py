from pathlib import Path

ACTIONS = [
    "나",
    "만나다",
    "반갑다",
    "밥 먹었어?",
    "배",
    "아프다",
    "오늘",
    "날씨",
    "좋다",
]

SEQ_LENGTH = 30
FEATURE_DIM = 150
DEFAULT_RECORD_SECONDS = 7
DEFAULT_TAKES = 5
DEFAULT_TOTAL_PEOPLE = 4

BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "dataset" / "videos"
SEQUENCE_DIR = BASE_DIR / "dataset" / "sequences"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "word_model.h5"
