"""
[방법 1: 직접촬영] record_dataset.py로 녹화한 영상을 시퀀스 npy로 변환한다.

dataset/videos/<자모>/<자모>_<n>.avi 의 각 프레임에서 오른손 특징(55차원)을 추출하고,
SEQ_LENGTH(10) 프레임 단위 슬라이딩 윈도우로 잘라 dataset/sequences/<자모>_<n>.npy 에 저장한다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from config import SEQ_LENGTH, VIDEO_DIR, SEQUENCE_DIR
from modules.holistic_module import HolisticDetector
from modules.utils import createDirectory, extract_phoneme_keypoints


def process_video(video_path: Path, detector: HolisticDetector) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frame_features = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = detector.process(frame)
        feat = extract_phoneme_keypoints(results.right_hand_landmarks)
        if feat is not None:
            frame_features.append(feat)

    cap.release()
    return np.array(frame_features)


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

        if len(frames) < SEQ_LENGTH:
            print(f"  경고: 인식된 프레임이 {len(frames)}개뿐이라 시퀀스를 만들 수 없습니다. 건너뜁니다.")
            continue

        sequences = [frames[i:i + SEQ_LENGTH] for i in range(len(frames) - SEQ_LENGTH + 1)]
        out_path = SEQUENCE_DIR / f"{action}_{video_path.stem}.npy"
        np.save(out_path, np.array(sequences))
        print(f"  저장됨: {out_path} (시퀀스 {len(sequences)}개)")

    detector.close()


if __name__ == "__main__":
    main()
