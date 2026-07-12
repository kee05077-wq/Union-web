"""
[방법 2: AIHub 공개 데이터] AIHub "수어 영상" 데이터셋에서 단어 구간을 잘라
시퀀스 npy로 변환한다.

출처: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=103
(로그인 및 이용 신청이 필요해 이 코드로 자동 다운로드는 하지 않는다. AIHub 홈페이지에서
 직접 내려받은 뒤, 영상(mp4)과 라벨링(json) 파일 경로를 아래 인자로 지정해서 실행한다.)

주의: AIHub 라벨링 JSON의 정확한 키 구조는 데이터 갱신에 따라 바뀔 수 있다.
      아래 `iter_segments()`는 공개된 설명 기준(URL/Duration/start/end/name)으로 작성했으니,
      실제로 내려받은 JSON을 한 번 열어 필드명이 다르면 이 함수만 맞춰 고치면 된다.

각 단어 구간은 프레임 수가 제각각이므로, SEQ_LENGTH(30) 프레임으로 균등 리샘플링한다.

사용법:
    python build_dataset_from_aihub.py --annotations-dir <라벨링 json 폴더> --videos-dir <mp4 폴더>
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from config import ACTIONS, SEQ_LENGTH, SEQUENCE_DIR
from modules.holistic_module import HolisticDetector
from modules.utils import createDirectory, extract_word_keypoints

# AIHub 라벨(name)이 우리 ACTIONS 표기와 다를 경우 여기에 매핑을 추가한다.
NAME_ALIASES: dict[str, str] = {}


def resolve_action_name(raw_name: str) -> str | None:
    name = NAME_ALIASES.get(raw_name, raw_name)
    return name if name in ACTIONS else None


def iter_segments(annotation_json: dict):
    entries = annotation_json if isinstance(annotation_json, list) else annotation_json.get("data", [annotation_json])
    for entry in entries:
        url = entry.get("URL") or entry.get("url")
        segments = entry.get("morphemes") or entry.get("segments") or [entry]
        for seg in segments:
            start = seg.get("start")
            end = seg.get("end")
            name = seg.get("name")
            if url is not None and start is not None and end is not None and name is not None:
                yield url, float(start), float(end), name


def extract_segment_sequence(video_path: Path, start_sec: float, end_sec: float, detector: HolisticDetector):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    start_frame = int(start_sec * fps)
    end_frame = max(int(end_sec * fps), start_frame + 1)

    # 구간을 SEQ_LENGTH개 프레임으로 균등 샘플링
    frame_indices = np.linspace(start_frame, end_frame - 1, SEQ_LENGTH).astype(int)

    features = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            features.append(np.zeros(150))
            continue
        results = detector.process(frame)
        features.append(extract_word_keypoints(results))

    cap.release()
    return np.array(features)


def main():
    parser = argparse.ArgumentParser(description="AIHub 수어 영상 -> 단어 시퀀스 npy 변환")
    parser.add_argument("--annotations-dir", required=True, help="AIHub 라벨링 json 파일들이 있는 폴더")
    parser.add_argument("--videos-dir", required=True, help="AIHub mp4 원본 영상이 있는 폴더")
    args = parser.parse_args()

    annotations_dir = Path(args.annotations_dir)
    videos_dir = Path(args.videos_dir)
    createDirectory(str(SEQUENCE_DIR))

    detector = HolisticDetector(min_detection_confidence=0.5)
    unmatched_names: set[str] = set()
    saved_count = 0

    for json_path in sorted(annotations_dir.glob("*.json")):
        annotation = json.loads(json_path.read_text(encoding="utf-8"))

        for video_url, start_sec, end_sec, raw_name in iter_segments(annotation):
            action = resolve_action_name(raw_name)
            if action is None:
                unmatched_names.add(raw_name)
                continue

            video_path = videos_dir / Path(video_url).name
            if not video_path.exists():
                print(f"영상 파일을 찾을 수 없습니다: {video_path}")
                continue

            print(f"처리 중: {video_path.name} [{start_sec:.2f}s ~ {end_sec:.2f}s] -> '{action}'")
            sequence = extract_segment_sequence(video_path, start_sec, end_sec, detector)

            out_path = SEQUENCE_DIR / f"{action}_aihub_{json_path.stem}_{start_sec:.2f}.npy"
            np.save(out_path, sequence)
            saved_count += 1
            print(f"  저장됨: {out_path}")

    detector.close()
    print(f"\n총 {saved_count}개 시퀀스 파일 저장 완료.")
    if unmatched_names:
        print("\n[참고] ACTIONS 목록과 매칭되지 않아 건너뛴 라벨명들 (필요하면 NAME_ALIASES에 추가):")
        print(sorted(unmatched_names))


if __name__ == "__main__":
    main()
