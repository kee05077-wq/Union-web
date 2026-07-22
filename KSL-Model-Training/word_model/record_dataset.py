"""
Record word-sign videos from webcam.

Defaults:
- 7 seconds per take
- 5 takes per action
- participant folders: <action>, <action>_2, <action>_3 ...

Controls:
- SPACE: start the current take
- ESC: cancel current screen or exit
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from config import (
    ACTIONS,
    DEFAULT_RECORD_SECONDS,
    DEFAULT_TAKES,
    DEFAULT_TOTAL_PEOPLE,
    SEQ_LENGTH,
    VIDEO_DIR,
)
from modules.utils import createDirectory, put_text_kr

WINDOW_NAME = "record_dataset"


def participant_folder_name(action: str, person_index: int) -> str:
    return action if person_index <= 1 else f"{action}_{person_index}"


def next_take_number(class_dir: Path) -> int:
    existing = sorted(class_dir.glob("*.avi"))
    if not existing:
        return 1
    return int(existing[-1].stem.rsplit("_", 1)[-1]) + 1


def show_message(frame, lines: list[str]):
    y = 10
    for idx, line in enumerate(lines):
        color = (0, 255, 0) if idx == 0 else (255, 255, 0)
        frame = put_text_kr(frame, line, (10, y), color=color)
        y += 42
    return frame


def wait_for_start(cap, action: str, person_index: int, take: int) -> bool:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라 프레임을 읽지 못했습니다.")
            return False
        frame = show_message(frame, [
            f"단어: {action} | 참여자 {person_index} | take {take}",
            "SPACE를 누르면 녹화를 시작합니다.",
            "ESC를 누르면 종료합니다.",
        ])
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return False
        if key == 32:
            return True


def countdown_screen(cap, action: str, person_index: int, take: int, seconds: int = 3) -> bool:
    for remaining in range(seconds, 0, -1):
        ret, frame = cap.read()
        if not ret:
            print("카메라 프레임을 읽지 못했습니다.")
            return False
        frame = show_message(frame, [
            f"'{action}' take {take} 준비",
            f"참여자 {person_index}",
            f"{remaining}초 후 시작",
        ])
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1000) & 0xFF
        if key == 27:
            return False
    return True


def record_take(cap, action: str, person_index: int, take: int, duration: float, fourcc) -> bool:
    folder_name = participant_folder_name(action, person_index)
    class_dir = VIDEO_DIR / folder_name
    createDirectory(str(class_dir))
    out_path = class_dir / f"{folder_name}_{take}.avi"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0

    if not wait_for_start(cap, action, person_index, take):
        return False

    if not countdown_screen(cap, action, person_index, take):
        return False

    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"영상 저장 파일을 열 수 없습니다: {out_path}")

    start_time = time.time()
    total_frames = max(int(round(duration * fps)), SEQ_LENGTH)

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print("카메라 프레임을 읽지 못했습니다.")
            break

        writer.write(frame)
        elapsed = time.time() - start_time
        remaining = max(0.0, duration - elapsed)
        frame = show_message(frame, [
            f"REC '{action}' take {take}",
            f"참여자 {person_index}",
            f"남은 시간: {remaining:.1f}초 | 프레임 {frame_idx + 1}/{total_frames}",
        ])
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            writer.release()
            out_path.unlink(missing_ok=True)
            return False
        if elapsed >= duration and frame_idx + 1 >= SEQ_LENGTH:
            break

    writer.release()
    print(f"저장됨: {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="단어 수어 학습용 영상 녹화")
    parser.add_argument("--takes", type=int, default=DEFAULT_TAKES, help="각 단어별 반복 촬영 횟수")
    parser.add_argument("--duration", type=float, default=DEFAULT_RECORD_SECONDS, help="각 촬영 길이(초)")
    parser.add_argument("--person-index", type=int, default=1, help="참여자 번호")
    parser.add_argument("--total-people", type=int, default=DEFAULT_TOTAL_PEOPLE, help="전체 참여 인원 수")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    if args.person_index < 1:
        raise SystemExit("--person-index는 1 이상이어야 합니다.")
    if args.takes < 1:
        raise SystemExit("--takes는 1 이상이어야 합니다.")
    if args.duration <= 0:
        raise SystemExit("--duration은 0보다 커야 합니다.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("카메라를 열 수 없습니다. 다른 앱이 사용 중인지 확인하세요.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")

    print(f"참여자 {args.person_index}/{args.total_people} 녹화를 시작합니다.")
    print(f"각 단어별 {args.duration}초 x {args.takes}회씩 촬영합니다.")
    print("미리보기 창에서 SPACE를 누르면 현재 take 녹화가 시작됩니다.")

    try:
        for action in ACTIONS:
            folder_name = participant_folder_name(action, args.person_index)
            class_dir = VIDEO_DIR / folder_name
            start_take = next_take_number(class_dir) if class_dir.exists() else 1
            end_take = start_take + args.takes - 1
            print(f"[{folder_name}] {start_take}~{end_take}번째 촬영")
            for take in range(start_take, start_take + args.takes):
                should_continue = record_take(cap, action, args.person_index, take, args.duration, fourcc)
                if not should_continue:
                    print("녹화를 종료합니다.")
                    return
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
