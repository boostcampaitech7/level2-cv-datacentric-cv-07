#canny를 이용해서 마스킹 처리한 이미지를 원본 이미지에 대체시키는 코드

import os
import shutil

# 경로 설정
data_dir = "/data/ephemeral/home/Jeongseon/data"  # 원본 이미지가 있는 폴더
result_dir = "/data/ephemeral/home/Jeongseon/output/result"  # 마스킹된 이미지가 저장된 폴더

# 언어별 폴더 설정
language_folders = {
    "china": "chinese_receipt",
    "japan": "japanese_receipt",
    "thai": "thai_receipt",
    "viet": "vietnamese_receipt"
}

# 각 언어별 폴더에 대해 파일 대체 작업 수행
for lang_key, data_subfolder in language_folders.items():
    # 언어별 result와 data 경로 설정
    result_lang_dir = os.path.join(result_dir, lang_key)
    data_lang_dir = os.path.join(data_dir, data_subfolder, "img", "train")

    # result 폴더의 언어별 파일 목록 가져오기
    if not os.path.exists(result_lang_dir):
        print(f"Result directory for {lang_key} not found: {result_lang_dir}")
        continue

    result_files = [f for f in os.listdir(result_lang_dir) if f.startswith("masked_")]

    # 각 이미지 파일 대체
    for result_file in result_files:
        # 마스킹된 이미지 파일 경로
        result_file_path = os.path.join(result_lang_dir, result_file)
        
        # 원본 이미지 파일 이름에서 "masked_" 제거
        original_file_name = result_file.replace("masked_", "")
        
        # 대체할 원본 파일 경로
        target_file_path = os.path.join(data_lang_dir, original_file_name)
        
        # 원본 파일 대체
        if os.path.exists(target_file_path):
            shutil.copy2(result_file_path, target_file_path)
            print(f"Replaced {target_file_path} with {result_file_path}")
        else:
            print(f"Original file not found: {target_file_path}")
