import json
import os
from PIL import Image, ImageDraw, ImageOps
import numpy as np

#json 파싱
def load_ufo_annotation(annotation_path):
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    return data

#상자 그리기
def draw_bounding_boxes(image, words_data):
    draw = ImageDraw.Draw(image)
    for _, word_data in words_data.items():
        bbox = word_data['points']
        bbox = [(point[0], point[1]) for point in bbox]
        
        # Draw bounding box as a polygon
        for i in range(len(bbox)):
            next_i = (i + 1) % len(bbox)
            draw.line([bbox[i], bbox[next_i]], fill="red", width=2)
    return image


#상자 표시하고 폴더에 저장
def process_images_with_bboxes(image_dir, annotation_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    annotations = load_ufo_annotation(annotation_path)

    for i in range(len(annotations['images'].items())):

        # 이미지 로드
        image_name = f"cord_{i+1}.jpg"
        image = image_dir[i]

        # 단어 데이터 사용해서 상자 그리기
        image_with_bboxes = draw_bounding_boxes(image, annotations['images'][image_name]['words'])

        # 상자 표지된 이미지 저장
        output_path = os.path.join(output_dir, f"bbox_{image_name}")
        image_with_bboxes.save(output_path)
        print(f"Saved image with bounding boxes to {output_path}")

from datasets import load_dataset
ds = load_dataset("naver-clova-ix/cord-v2")

# 경로
train_img_data = ds['train']['image']
test_img_data = ds['test']['image']
validation_img_data = ds['validation']['image']

image_dir = train_img_data + test_img_data + validation_img_data
annotation_path = "/data/ephemeral/home/Jungyeon/new_dataset/cord.json"
output_dir = "/data/ephemeral/home/Jungyeon/new_dataset/bbox_img"

# 함수 호출
process_images_with_bboxes(image_dir, annotation_path, output_dir)
