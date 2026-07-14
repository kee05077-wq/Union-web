"""
[방법 1: 직접촬영] record_dataset.py로 녹화한 영상을 시퀀스 npy로 변환한다.

각 영상(정확히 SEQ_LENGTH 프레임)에서 pose+양손 특징(150차원)을 추출해
dataset/sequences/<단어>_<n>.npy (shape: [SEQ_LENGTH, 150]) 로 저장한다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from config import SEQ_LENGTH, VIDEO_DIR, SEQUENCE_DIR
from modules.holistic_module import HolisticDetector
from modules.utils import createDirectory, extract_word_keypoints


def process_video(video_path: Path, detector: HolisticDetector) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frame_features = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = detector.process(frame)
        frame_features.append(extract_word_keypoints(results))

    cap.release()
    return np.array(frame_features)


def fit_to_seq_length(frames: np.ndarray) -> np.ndarray:
    """녹화 오차로 프레임 수가 SEQ_LENGTH와 살짝 다를 경우 자르거나 마지막 프레임으로 채운다."""
    if len(frames) == SEQ_LENGTH:
        return frames
    if len(frames) > SEQ_LENGTH:
        return frames[:SEQ_LENGTH]
    pad = np.repeat(frames[-1:], SEQ_LENGTH - len(frames), axis=0)
    return np.concatenate([frames, pad], axis=0)


def main():
    createDirectory(str(SEQUENCE_DIR))
    detector = HolisticDetector(min_detection_confidence=0.5)

    video_paths = sorted(VIDEO_DIR.glob("*/*.avi"))
    if not video_paths:
        print(f"{VIDEO_DIR} 에 녹화된 영상이 없습니다. 먼저 record_dataset.py를 실행하세요.")
        return

    for video_path in video_paths:
        action = video_path.parent.name
        print(f"처리 중: {video_path}")
        frames = process_video(video_path, detector)
        if len(frames) == 0:
            print("  경고: 프레임을 읽지 못해 건너뜁니다.")
            continue

        sequence = fit_to_seq_length(frames)
        out_path = SEQUENCE_DIR / f"{action}_{video_path.stem}.npy"
        np.save(out_path, sequence)
        print(f"  저장됨: {out_path} (shape={sequence.shape})")

    detector.close()


if __name__ == "__main__":
    main()
