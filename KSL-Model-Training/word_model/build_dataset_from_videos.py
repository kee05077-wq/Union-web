"""
[?? 1: ????] record_dataset.py? ??? ??? ??? npy? ????.

?? ???? ?? ??? ???? ??? ?? ??(ACTIONS)? ?????,
???? <??>, <??>_2, <??>_3 ... ? ?? ???(_??)? ??? ??? ????.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from config import SEQ_LENGTH, VIDEO_DIR, SEQUENCE_DIR
from modules.holistic_module import HolisticDetector
from modules.utils import createDirectory, extract_word_keypoints

PARTICIPANT_SUFFIX_RE = re.compile(r"^(?P<action>.+?)(?:_(?P<person>\d+))?$")


def split_action_and_person(folder_name: str) -> tuple[str, int]:
    match = PARTICIPANT_SUFFIX_RE.match(folder_name)
    if not match:
        return folder_name, 1
    action = match.group("action")
    person = int(match.group("person") or 1)
    return action, person


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
    """?? ??? ??? ?? SEQ_LENGTH? ?? ?? ?? ???? ??? ????? ???."""
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
        print(f"{VIDEO_DIR} ? ??? ??? ????. ?? record_dataset.py? ?????.")
        return

    for video_path in video_paths:
        folder_name = video_path.parent.name
        action, person_index = split_action_and_person(folder_name)
        print(f"?? ?: {video_path} -> ?? '{action}' (??? {person_index})")
        frames = process_video(video_path, detector)
        if len(frames) == 0:
            print("  ??: ???? ?? ?? ?????.")
            continue

        sequence = fit_to_seq_length(frames)
        out_path = SEQUENCE_DIR / f"{action}__p{person_index}__{video_path.stem}.npy"
        np.save(out_path, sequence)
        print(f"  ???: {out_path} (shape={sequence.shape})")

    detector.close()


if __name__ == "__main__":
    main()
