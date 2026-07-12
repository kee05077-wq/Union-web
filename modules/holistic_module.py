from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HolisticLandmarker, HolisticLandmarkerOptions, RunningMode

# 어깨(11,12), 팔꿈치(13,14), 손목(15,16) - MediaPipe Pose 랜드마크 인덱스
POSE_LANDMARK_INDICES = [11, 12, 13, 14, 15, 16]

MODEL_PATH = Path(__file__).resolve().parent.parent / "mediapipe_models" / "holistic_landmarker.task"

class HolisticDetector():
    """MediaPipe Tasks API(HolisticLandmarker) 래퍼.

    음운(지문자) 모델은 오른손 21개 랜드마크만 사용하고,
    단어 모델은 양손 + 위 6개 pose 랜드마크를 모두 사용한다.
    """

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # HolisticLandmarkerOptions에는 min_tracking_confidence가 없다 (IMAGE 모드라 프레임간 추적을 하지 않음).
        options = HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            min_face_detection_confidence=min_detection_confidence,
            min_pose_detection_confidence=min_detection_confidence,
            min_hand_landmarks_confidence=min_detection_confidence,
            running_mode=RunningMode.IMAGE,
        )
        self.landmarker = HolisticLandmarker.create_from_options(options)
        self.results = None

    def process(self, img_bgr):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        self.results = self.landmarker.detect(mp_image)
        return self.results

    def draw(self, img_bgr):
        if self.results is None:
            return img_bgr

        h, w = img_bgr.shape[:2]
        for lm in self.results.left_hand_landmarks:
            cv2.circle(img_bgr, (int(lm.x * w), int(lm.y * h)), 3, (121, 22, 76), -1)
        for lm in self.results.right_hand_landmarks:
            cv2.circle(img_bgr, (int(lm.x * w), int(lm.y * h)), 3, (245, 117, 66), -1)
        if self.results.pose_landmarks:
            for idx in POSE_LANDMARK_INDICES:
                lm = self.results.pose_landmarks[idx]
                cv2.circle(img_bgr, (int(lm.x * w), int(lm.y * h)), 5, (80, 22, 10), -1)
        return img_bgr

    def close(self):
        self.landmarker.close()
