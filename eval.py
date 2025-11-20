import os
import random
import shutil

# 원본 폴더와 새 폴더 경로 설정
source_folder = "./traces_train/malware"     # 예: "./data/json_all"
destination_folder = "./eval_mal"  # 예: "./data/json_sample"

# 새 폴더가 없으면 생성
os.makedirs(destination_folder, exist_ok=True)

# 폴더 내 모든 JSON 파일 목록 가져오기
json_files = [f for f in os.listdir(source_folder) if f.endswith(".json")]

# 100개 랜덤 샘플링 (100개 미만이면 가능한 만큼만 이동)
sample_size = min(100, len(json_files))
sampled_files = random.sample(json_files, sample_size)

# 선택된 파일 이동
for filename in sampled_files:
    src_path = os.path.join(source_folder, filename)
    dst_path = os.path.join(destination_folder, filename)
    shutil.move(src_path, dst_path)  # 복사가 아니라 이동

print(f"✅ {sample_size}개의 JSON 파일이 '{destination_folder}' 폴더로 이동되었습니다.")
