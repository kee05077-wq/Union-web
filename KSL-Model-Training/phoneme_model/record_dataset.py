"""
[방법 1: 직접촬영] 지문자(자음/모음) 학습용 영상을 웹캠으로 녹화한다.

사용법:
    python record_dataset.py --seconds 20 --hands right

각 클래스(ACTIONS)마다 안내 문구를 보여준 뒤 지정한 시간(초) 동안 녹화하고,
dataset/videos/<자모>/<자모>_<n>.avi 로 저장한다. 여러 번 실행해서 촬영을 반복하면
매번 새로운 take 번호로 이어서 저장된다. 녹화 중 ESC를 누르면 그 클래스 촬영을 중단한다.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from config import ACTIONS, VIDEO_DIR
from modules.utils import createDirectory


def next_take_number(class_dir) -> int:
    existing = sorted(class_dir.glob("*.avi"))
    if not existing:
        return 1
    last = existing[-1].stem  # 예: "ㄱ_3"
    return int(last.rsplit("_", 1)[-1]) + 1


def record_class(cap, action: str, seconds: float, fourcc) -> None:
    class_dir = VIDEO_DIR / action
    createDirectory(str(class_dir))
    take = next_take_number(class_dir)
    out_path = class_dir / f"{action}_{take}.avi"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    print(f"[{action}] {seconds}초간 녹화합니다. 준비되면 아무 키나 누르세요 (ESC=건너뛰기)")
    ret, frame = cap.read()
    if ret:
        cv2.putText(frame, f"'{action}' 준비 - 아무 키나 누르세요", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("record_dataset", frame)
        if cv2.waitKey(0) == 27:
            writer.release()
            out_path.unlink(missing_ok=True)
            return

    start = time.time()
    while time.time() - start < seconds:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

        remaining = seconds - (time.time() - start)
        cv2.putText(frame, f"REC '{action}' {remaining:0.1f}s", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow("record_dataset", frame)
        if cv2.waitKey(1) == 27:
            break

    writer.release()
    print(f"저장됨: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="지문자 학습용 영상 녹화")
    parser.add_argument("--seconds", type=float, default=20.0, help="클래스당 녹화 시간(초)")
    parser.add_argument("--camera", type=int, default=0, help="cv2.VideoCapture 카메라 인덱스")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        sys.exit(1)

    fourcc = cv2.VideoWriter_fourcc(*"DIVX")

    try:
        for action in ACTIONS:
            record_class(cap, action, args.seconds, fourcc)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
