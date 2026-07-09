import cv2
from ultralytics import YOLO

# 1. 학습된 최적의 가중치 로드
model_path = 'runs/detect/train/weights/best.pt'
try:
    model = YOLO(model_path)
except Exception as e:
    print(f"[오류] 모델 로드 실패. 경로를 확인하십시오: {model_path}")
    print(f"상세 에러: {e}")
    exit()

# 2. 웹캠 연결 (기본 카메라: 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[오류] 웹캠을 인식할 수 없습니다. 연결 상태를 확인하십시오.")
    exit()

print("웹캠 추론을 시작합니다. 활성화된 창에서 'q'를 누르면 종료됩니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[오류] 프레임을 읽어올 수 없습니다.")
        break

    # 직관적인 테스트를 위한 좌우 반전 (거울 모드)
    frame = cv2.flip(frame, 1)

    # 3. 객체 탐지 수행 (신뢰도 60% 이상인 결과만 표시, 콘솔 출력 생략)
    results = model(frame, conf=0.6, verbose=False)

    # 4. Bounding Box 및 라벨이 렌더링된 프레임 추출
    annotated_frame = results[0].plot()

    # 5. 화면 출력
    cv2.imshow('YOLOv8 Real-time Inference', annotated_frame)

    # 'q' 키를 누르면 루프 탈출
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6. 자원 해제
cap.release()
cv2.destroyAllWindows()
