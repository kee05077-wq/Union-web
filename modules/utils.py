import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

# ==========================================
# 한글 텍스트 오버레이
# cv2.putText는 Hershey 폰트만 지원해서 한글 글리프가 없어 '?'로 깨진다.
# PIL로 렌더링한 뒤 다시 OpenCV 이미지로 변환하는 방식으로 우회한다.
# ==========================================
_KOREAN_FONT_PATH = Path("C:/Windows/Fonts/malgun.ttf")
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}

def _get_korean_font(font_size: int) -> ImageFont.FreeTypeFont:
    if font_size not in _font_cache:
        _font_cache[font_size] = ImageFont.truetype(str(_KOREAN_FONT_PATH), font_size)
    return _font_cache[font_size]

def put_text_kr(img_bgr, text, org, font_size=28, color=(255, 255, 255)):
    """한글이 섞인 텍스트를 OpenCV BGR 이미지에 그려서 반환한다.
    org는 cv2.putText와 달리 텍스트 좌상단 좌표, color는 (B, G, R) 순서."""
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(org, text, font=_get_korean_font(font_size), fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# vector normalization
def Vector_Normalization(joint):
    # Compute angles between joints
    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :2] # Parent joint
    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :2] # Child joint
    v = v2 - v1
    # Normalize v
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    # Get angle using arcos of dot product
    angle = np.arccos(np.einsum('nt,nt->n',
        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))

    angle = np.degrees(angle) # Convert radian to degree

    angle_label = np.array([angle], dtype=np.float32)

    return v, angle_label

# ==========================================
# 시간 기반 프레임 리샘플링
# 소켓으로 들어오는 프레임은 도착 간격이 일정하지 않다(네트워크 지연, 서버 부하에 따라
# 29ms~200ms+ 로 들쭉날쭉함). 학습 데이터(record_dataset.py, 고정 fps 웹캠 녹화)와
# 같은 시간축을 모델에 넣어주기 위해, "최근 도착한 프레임 N개"가 아니라
# "최근 window_seconds 동안"을 seq_length개의 균일한 시점으로 리샘플링한다.
# 각 가상 시점에는 그 시점 이전에 가장 최근 관측된 값을 사용한다(zero-order hold).
# ==========================================
def resample_time_buffer(buffer, seq_length: int, window_seconds: float, now: float):
    """buffer: (timestamp, feature_vector) 튜플의 시퀀스(시간 오름차순).
    window_seconds 만큼의 실제 경과 시간이 아직 안 쌓였으면 None을 반환한다."""
    if not buffer:
        return None
    oldest_ts = buffer[0][0]
    if now - oldest_ts < window_seconds:
        return None

    timestamps = np.array([ts for ts, _ in buffer])
    features = np.stack([feat for _, feat in buffer])

    target_times = now - window_seconds + np.linspace(0, window_seconds, seq_length)
    resampled = np.empty((seq_length,) + features.shape[1:], dtype=np.float32)
    for i, t in enumerate(target_times):
        idx = np.searchsorted(timestamps, t, side="right") - 1
        idx = max(idx, 0)
        resampled[i] = features[idx]
    return resampled
