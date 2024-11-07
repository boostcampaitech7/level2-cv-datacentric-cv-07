#illegibility의 값에 따라 bbox의 색상을 다르게하여 분석하기 위한 코드
#bbox id도 시각화

import json
import os
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np

# JSON 파싱
def load_ufo_annotation(annotation_path):
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    return data

# 상자 그리기 및 번호 표시
def draw_bounding_boxes(image, words_data, font_size=150):
    draw = ImageDraw.Draw(image)
    
    # 기본 폰트 설정 (크기 조정 가능)
    font = ImageFont.load_default()

    for word_id, word_data in words_data.items():
        bbox = word_data['points']
        bbox = [(point[0], point[1]) for point in bbox]
        
        # illegibility 상태에 따른 색상 설정
        color = "blue" if word_data['illegibility'] else "red"
        
        # Bounding box 그리기
        for i in range(len(bbox)):
            next_i = (i + 1) % len(bbox)
            draw.line([bbox[i], bbox[next_i]], fill=color, width=2)
        
        # Bounding box 왼쪽 상단에 ID 텍스트 표시
        top_left_x, top_left_y = bbox[0]  # 왼쪽 상단 좌표
        draw.text((top_left_x, top_left_y), word_id, fill=color, font=font)
    
    return image

# 상자 표시하고 폴더에 저장
def process_images_with_bboxes(image_dir, annotation_path, output_dir, font_size=20):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    annotations = load_ufo_annotation(annotation_path)

    for image_name, annotation_data in annotations['images'].items():
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Image {image_name} not found in {image_dir}")
            continue

        # 이미지 로드
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")

        # 단어 데이터 사용해서 상자 그리기
        image_with_bboxes = draw_bounding_boxes(image, annotation_data['words'], font_size)

        # 상자 표시된 이미지 저장
        output_path = os.path.join(output_dir, f"bbox_{image_name}")
        image_with_bboxes.save(output_path)
        print(f"Saved image with bounding boxes to {output_path}")

# 경로 설정
image_dir = "/data/ephemeral/home/Jeongseon/data/vietnamese_receipt/img/train"                       # 이미지 경로
annotation_path = "/data/ephemeral/home/Jeongseon/data/vietnamese_receipt/ufo/train.json"            # 어노테이션 경로
output_dir = "/data/ephemeral/home/Jeongseon/output/viet_total"                                      # bounding box 표시된 이미지 저장할 경로

# 함수 호출
process_images_with_bboxes(image_dir, annotation_path, output_dir, font_size=20)
