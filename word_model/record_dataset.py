"""
[방법 1: 직접촬영] 단어 수어 학습용 영상을 웹캠으로 녹화한다.

각 단어(ACTIONS)마다 --takes 번(기본 20) 반복 촬영하며, 한 번 촬영(take)은
정확히 SEQ_LENGTH(30) 프레임으로 고정한다 (한 번의 단어 동작 = 한 개의 학습 샘플).

사용법:
    python record_dataset.py --takes 20
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from config import ACTIONS, SEQ_LENGTH, VIDEO_DIR
from modules.utils import createDirectory


def next_take_number(class_dir: Path) -> int:
    existing = sorted(class_dir.glob("*.avi"))
    if not existing:
        return 1
    return int(existing[-1].stem.rsplit("_", 1)[-1]) + 1


def record_take(cap, action: str, take: int, fourcc) -> None:
    class_dir = VIDEO_DIR / action
    createDirectory(str(class_dir))
    out_path = class_dir / f"{action}_{take}.avi"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    ret, frame = cap.read()
    if ret:
        cv2.putText(frame, f"'{action}' {take}번째 - 준비되면 아무 키나 (ESC=건너뛰기)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("record_dataset", frame)
        if cv2.waitKey(0) == 27:
            writer.release()
            out_path.unlink(missing_ok=True)
            return

    for i in range(SEQ_LENGTH):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        cv2.putText(frame, f"REC '{action}' {i + 1}/{SEQ_LENGTH}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("record_dataset", frame)
        cv2.waitKey(1)

    writer.release()
    print(f"저장됨: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="단어 수어 학습용 영상 녹화")
    parser.add_argument("--takes", type=int, default=20, help="단어당 반복 촬영 횟수")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("카메라를 열 수 없습니다.")

    fourcc = cv2.VideoWriter_fourcc(*"DIVX")

    try:
        for action in ACTIONS:
            class_dir = VIDEO_DIR / action
            start_take = next_take_number(class_dir) if class_dir.exists() else 1
            for take in range(start_take, start_take + args.takes):
                record_take(cap, action, take, fourcc)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
