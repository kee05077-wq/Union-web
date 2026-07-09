import os
import pandas as pd
import shutil

team_folders = ['personA', 'personB', 'personC', 'personD']
target_classes = range(31) 
output_base_dir = 'merged_data'

os.makedirs(output_base_dir, exist_ok=True)
success_count = 0

print("데이터 병합 작업을 시작합니다...\n")

for sign_num in target_classes:
    merged_csv_list = []
    output_sign_dir = os.path.join(output_base_dir, str(sign_num))
    os.makedirs(output_sign_dir, exist_ok=True)
    
    for person_name in team_folders:
        csv_path = f'{person_name}/{sign_num}/capstone_sign_{sign_num}.csv'
        img_dir = f'{person_name}/{sign_num}'
        
        if not os.path.exists(csv_path):
            print(f"[경고] 데이터를 찾을 수 없습니다: {csv_path}")
            continue
            
        df = pd.read_csv(csv_path)
        prefix = f"{person_name}_" 
        
        for index, row in df.iterrows():
            old_img_name = row['image']
            new_img_name = prefix + old_img_name 
            
            old_img_path = os.path.join(img_dir, old_img_name)
            new_img_path = os.path.join(output_sign_dir, new_img_name)
            
            if os.path.exists(old_img_path):
                shutil.copy(old_img_path, new_img_path)
                
                row['image'] = new_img_name
                merged_csv_list.append(row)
    
    if merged_csv_list:
        merged_df = pd.DataFrame(merged_csv_list)
        merged_df.to_csv(f'{output_sign_dir}/merged_capstone_sign_{sign_num}.csv', index=False)
        print(f"Class {sign_num}: 데이터 병합 완료 (총 {len(merged_df)}개)")
        success_count += 1

print(f"\n총 {success_count}개 클래스의 데이터 병합 처리가 종료되었습니다.")
input("작업 내역을 확인하신 후 엔터 키를 누르면 창이 닫힙니다...")
