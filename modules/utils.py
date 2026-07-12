import os
import numpy as np

from .holistic_module import POSE_LANDMARK_INDICES

def createDirectory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# ==========================================
# 음운(지문자) 모델용 특징 추출
# 오른손 21개 랜드마크 -> 관절 벡터 20개 + 관절 각도 15개 = 55차원
# (Union-web에 이미 배포된 gesture_classifier.h5와 동일한 특징 공식)
# ==========================================
def Vector_Normalization(joint):
    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :2]
    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :2]
    v = v2 - v1
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    angle = np.arccos(np.einsum(
        'nt,nt->n',
        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :],
    ))
    angle = np.degrees(angle)
    angle_label = np.array([angle], dtype=np.float32)

    return v, angle_label

def extract_phoneme_keypoints(right_hand_landmarks):
    """오른손 랜드마크(mediapipe results.right_hand_landmarks, 랜드마크 리스트) -> 55차원 특징 벡터.
    손이 인식되지 않았으면(빈 리스트) None을 반환한다."""
    if not right_hand_landmarks:
        return None

    joint = np.zeros((21, 2))
    for i, lm in enumerate(right_hand_landmarks):
        joint[i] = [lm.x, lm.y]

    vector, angle_label = Vector_Normalization(joint)
    return np.concatenate([vector.flatten(), angle_label.flatten()])

# ==========================================
# 단어 모델용 특징 추출
# pose 6점(x,y,z,visibility) + 왼손 21점(x,y,z) + 오른손 21점(x,y,z) = 24 + 63 + 63 = 150차원
# ==========================================
def extract_word_keypoints(results):
    if results.pose_landmarks:
        pose = np.array([
            [lm.x, lm.y, lm.z, lm.visibility or 0.0]
            for i, lm in enumerate(results.pose_landmarks)
            if i in POSE_LANDMARK_INDICES
        ]).flatten()
    else:
        pose = np.zeros(len(POSE_LANDMARK_INDICES) * 4)

    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks]).flatten()
    else:
        lh = np.zeros(21 * 3)

    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks]).flatten()
    else:
        rh = np.zeros(21 * 3)

    return np.concatenate([pose, lh, rh])
