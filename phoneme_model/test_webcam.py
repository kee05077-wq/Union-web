"""
웹캠으로 음운(지문자) 모델을 직접 테스트한다.

사용법:
    python test_webcam.py
    (ESC로 종료)
"""
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from config import ACTIONS, MODEL_PATH, SEQ_LENGTH
from modules.holistic_module import HolisticDetector
from modules.utils import extract_phoneme_keypoints

CONFIDENCE_THRESHOLD = 0.6


def main():
    model = load_model(MODEL_PATH, compile=False)
    detector = HolisticDetector(min_detection_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("카메라를 열 수 없습니다.")

    seq = deque(maxlen=SEQ_LENGTH)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.process(frame)
        frame = detector.draw(frame)

        feat = extract_phoneme_keypoints(results.right_hand_landmarks)
        label, confidence = "-", 0.0

        if feat is not None:
            seq.append(feat)
            if len(seq) == SEQ_LENGTH:
                input_data = np.expand_dims(np.array(seq, dtype=np.float32), axis=0)
                y_pred = model.predict(input_data, verbose=0)[0]
                class_id = int(np.argmax(y_pred))
                confidence = float(y_pred[class_id])
                if confidence > CONFIDENCE_THRESHOLD:
                    label = ACTIONS[class_id]

        cv2.rectangle(frame, (0, 0), (320, 40), (245, 117, 16), -1)
        cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow("phoneme_model test", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
