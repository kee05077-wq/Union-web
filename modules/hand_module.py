from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"

class HandDetector():
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=RunningMode.IMAGE,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self.result = None

    def findHands(self, img):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.result = self.landmarker.detect(mp_image)
        return img

    def findRightHandLandmark(self, img):
        if not self.result or not self.result.hand_landmarks:
            return None

        for hand_landmarks, handedness in zip(self.result.hand_landmarks, self.result.handedness):
            if handedness[0].category_name == "Right":
                return hand_landmarks

        return None
