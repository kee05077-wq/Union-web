"""
gesture_classifier.h5 모델을 웹캠으로 직접 테스트한다.
app.py의 sign_frame 처리 로직과 동일한 파이프라인(오른손 21점 -> 55차원 특징 -> 10프레임 시퀀스)을 사용한다.

사용법:
    python test_gesture_classifier.py
    (ESC로 종료)
"""
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from modules.hand_module import HandDetector
from modules.utils import Vector_Normalization

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "gesture_classifier.h5"

ACTIONS = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
           'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ']
SEQ_LENGTH = 10
CONFIDENCE_THRESHOLD = 0.5


def extract_feature(right_hand_lmList):
    """오른손 랜드마크(리스트) -> 55차원 특징 벡터. 손이 없으면 None."""
    if right_hand_lmList is None:
        return None

    joint = np.zeros((42, 2))
    for j, lm in enumerate(right_hand_lmList):
        joint[j] = [lm.x, lm.y]

    vector, angle_label = Vector_Normalization(joint)
    return np.concatenate([vector.flatten(), angle_label.flatten()])


def main():
    print(f"모델 로드 중: {MODEL_PATH}")
    model = load_model(MODEL_PATH, compile=False)
    detector = HandDetector(min_detection_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("카메라를 열 수 없습니다.")

    seq = deque(maxlen=SEQ_LENGTH)
    label, confidence = "-", 0.0

    print("ESC를 누르면 종료합니다.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detector.findHands(frame)
        right_hand_lmList = detector.findRightHandLandmark(frame)

        feature = extract_feature(right_hand_lmList)
        if feature is not None:
            seq.append(feature)
            if len(seq) == SEQ_LENGTH:
                input_data = np.expand_dims(np.array(seq, dtype=np.float32), axis=0)
                y_pred = model.predict(input_data, verbose=0)[0]
                class_id = int(np.argmax(y_pred))
                confidence = float(y_pred[class_id])
                label = ACTIONS[class_id] if confidence > CONFIDENCE_THRESHOLD else "-"

        cv2.rectangle(frame, (0, 0), (320, 40), (245, 117, 16), -1)
        cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow("gesture_classifier test", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
