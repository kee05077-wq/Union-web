from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')

    results = model.train(
        data='data.yaml',
        epochs=10,
        batch=16,
        imgsz=640,
        device=0,
        workers=2,         # 추가: CPU 워커 수를 2로 제한하여 메모리 초과 방지
        fliplr=0.5,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        mosaic=0.0,
        mixup=0.0
    )
