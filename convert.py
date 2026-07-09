import os
import cv2
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

target_classes = range(31) # 0~30번
input_base_dir = 'merged_data'
output_base_dir = 'dataset'

# 1. YOLO 학습용 디렉토리 구조 생성
for split in ['train', 'val']:
    os.makedirs(f'{output_base_dir}/images/{split}', exist_ok=True)
    os.makedirs(f'{output_base_dir}/labels/{split}', exist_ok=True)

print("YOLO 포맷 변환 및 데이터 분할 작업을 시작합니다...\n")

for sign_num in target_classes:
    csv_path = f'{input_base_dir}/{sign_num}/merged_capstone_sign_{sign_num}.csv'
    source_img_dir = f'{input_base_dir}/{sign_num}/'
    
    if not os.path.exists(csv_path):
        continue

    df = pd.read_csv(csv_path)
    
    # 데이터가 너무 적을 경우 분할 오류 방지
    if len(df) < 2:
        print(f"[경고] Class {sign_num}: 데이터가 부족하여 건너뜁니다.")
        continue

    # 2. Train / Val 분할 (8:2 비율)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    def process_data(data_frame, split_name):
        for _, row in data_frame.iterrows():
            img_name = row['image']
            label = row['label']
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            
            img_path = os.path.join(source_img_dir, img_name)
            
            # 3. 이미지 복사 및 정규화 좌표 계산
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img_height, img_width = img.shape[:2]
            
            # YOLO 상대 좌표 (0~1) 계산
            x_center = (xmin + xmax) / (2.0 * img_width)
            y_center = (ymin + ymax) / (2.0 * img_height)
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            # 지정된 split(train/val) 폴더로 이미지 복사
            dst_img_path = os.path.join(output_base_dir, 'images', split_name, img_name)
            shutil.copy(img_path, dst_img_path)
            
            # 동일한 이름의 .txt 라벨 파일 생성
            txt_name = img_name.replace('.jpg', '.txt')
            dst_label_path = os.path.join(output_base_dir, 'labels', split_name, txt_name)
            
            with open(dst_label_path, 'w') as f:
                f.write(f"{label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    # 분할된 데이터 처리 실행
    process_data(train_df, 'train')
    process_data(val_df, 'val')
    print(f"Class {sign_num}: 변환 완료 (Train {len(train_df)}장, Val {len(val_df)}장)")

print("\n모든 변환 작업이 완료되었습니다. 'dataset' 폴더 내부를 확인하십시오.")
input("작업 내역을 확인하신 후 엔터 키를 누르면 창이 닫힙니다...")
