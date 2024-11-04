import streamlit as st
import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import albumentations as A
import cv2  # OpenCV for dilation

# 화면을 넓게 설정
st.set_page_config(layout="wide")

# 데이터 경로 설정 (예시 경로입니다. 실제 경로에 맞게 수정하세요)
data_paths = {
    "중국": "/data/ephemeral/home/data/chinese_receipt/img/train",
    "일본": "/data/ephemeral/home/data/japanese_receipt/img/train",
    "태국": "/data/ephemeral/home/data/thai_receipt/img/train",
    "베트남": "/data/ephemeral/home/data/vietnamese_receipt/img/train"
}

# 어노테이션 파일 불러오기 (파일명 예시)
annotations = {
    "중국": "/data/ephemeral/home/data/chinese_receipt/ufo/train.json",
    "일본": "/data/ephemeral/home/data/japanese_receipt/ufo/train.json",
    "태국": "/data/ephemeral/home/data/thai_receipt/ufo/train.json",
    "베트남": "/data/ephemeral/home/data/vietnamese_receipt/ufo/train.json"
}

# 어노테이션 로딩 함수 
def load_annotations(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 이미지 로딩 함수 (EXIF 방향 정보 자동 적용)
def load_image_with_orientation(image_path):
    image = ImageOps.exif_transpose(Image.open(image_path))
    return image.convert("RGB")

# 상자 그리기 함수 정의
def draw_bounding_boxes(image, words_data):
    draw = ImageDraw.Draw(image)
    for word_id, word_data in words_data.items():
        bbox = word_data['points']
        bbox = [(point[0], point[1]) for point in bbox]
        for i in range(len(bbox)):
            next_i = (i + 1) % len(bbox)
            draw.line([bbox[i], bbox[next_i]], fill="red", width=2)
    return image

def convert_to_pascal_voc_format(bboxes, image_width, image_height):
    voc_bboxes = []
    for bbox in bboxes:
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        
        x_min = max(0.0, min(1.0, min(x_coords) / image_width))
        y_min = max(0.0, min(1.0, min(y_coords) / image_height))
        x_max = max(0.0, min(1.0, max(x_coords) / image_width))
        y_max = max(0.0, min(1.0, max(y_coords) / image_height))
        
        if x_min >= x_max or y_min >= y_max:
            print(f"Warning: Invalid bbox detected: {[x_min, y_min, x_max, y_max]}")
            continue
            
        voc_bboxes.append([x_min, y_min, x_max, y_max])
    
    return voc_bboxes

def convert_from_pascal_voc(bbox, image_width, image_height):
    x_min, y_min, x_max, y_max = bbox
    return [
        [x_min * image_width, y_min * image_height],
        [x_max * image_width, y_min * image_height],
        [x_max * image_width, y_max * image_height],
        [x_min * image_width, y_max * image_height]
    ]

def augment_image(image, bboxes, h_flip, v_flip, brightness, blur, crop, hue_shift, sat_shift, val_shift,
                  alpha, sigma, scale, rotate_limit, dilation_kernel_size):
    height, width = image.shape[:2]
    voc_bboxes = convert_to_pascal_voc_format(bboxes, width, height)

    if not voc_bboxes:
        print("Warning: No valid bounding boxes found after conversion")
        return image, []

    # Augmentation transforms
    transforms = []
    if h_flip:
        transforms.append(A.HorizontalFlip(p=1.0))
    if v_flip:
        transforms.append(A.VerticalFlip(p=1.0))
    if brightness > 0:
        transforms.append(A.CLAHE(clip_limit=max(brightness, 1), p=1.0))
    if blur > 0:
        transforms.append(A.GaussianBlur(blur_limit=(blur, blur), p=1.0))
    if crop > 0:
        transforms.append(A.CenterCrop(height=crop, width=crop, p=1.0))
    if hue_shift != 0 or sat_shift != 0 or val_shift != 0:
        transforms.append(A.HueSaturationValue(
            hue_shift_limit=(hue_shift, hue_shift),
            sat_shift_limit=(sat_shift, sat_shift),
            val_shift_limit=(val_shift, val_shift),
            p=1.0
        ))

    # New augmentations
    if alpha > 0 and sigma > 0:
        transforms.append(A.ElasticTransform(alpha=alpha, sigma=sigma, alpha_affine=alpha * 0.1, p=1.0))
    if scale > 0:
        transforms.append(A.Perspective(scale=(scale, scale), p=1.0))
    if rotate_limit != 0:
        transforms.append(A.Rotate(limit=rotate_limit, p=1.0))

    # Custom dilation using Lambda and OpenCV
    if dilation_kernel_size > 0:
        kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        transforms.append(A.Lambda(image=lambda img, **kwargs: cv2.dilate(img, kernel), p=1.0))

    # Apply transformations
    if not transforms:
        return image, voc_bboxes

    transform = A.Compose(
        transforms,
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=[], min_area=0, min_visibility=0)
    )

    try:
        augmented = transform(image=image, bboxes=voc_bboxes)
        height, width = augmented['image'].shape[:2]
        original_format_bboxes = [convert_from_pascal_voc(bbox, width, height) for bbox in augmented['bboxes']]
        return augmented['image'], original_format_bboxes
    except ValueError as e:
        print(f"Augmentation failed: {str(e)}")
        return image, bboxes

# Streamlit UI
st.title("Data Augmentation and Annotation Editor")

# 나라 선택
country = st.selectbox("나라 선택", options=list(data_paths.keys()))
image_folder = data_paths[country]
annotation_file = annotations[country]

# 어노테이션 데이터 불러오기
annotation_data = load_annotations(annotation_file)

# 이미지 선택
image_files = os.listdir(image_folder)
selected_image = st.selectbox("이미지 선택", options=image_files)
image_path = os.path.join(image_folder, selected_image)

# 원본 이미지 로딩 및 복사
original_image = load_image_with_orientation(image_path)

# 어노테이션 정보 로딩
image_annotations = annotation_data["images"].get(selected_image, {}).get("words", {})

# 수정 가능한 어노테이션 정보
updated_annotations = {}
for word_id, word_data in image_annotations.items():
    st.write(f"ID: {word_id}")
    text = st.text_input(f"Text for {word_id}", value=word_data["transcription"])
    points = st.text_area(
        f"Points for {word_id}",
        value=str(word_data["points"]),
        help="Enter points as [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]"
    )
    updated_annotations[word_id] = {
        "transcription": text,
        "points": eval(points)
    }

# 바운딩 박스 그리기 (원본 이미지)
drawn_original_image = draw_bounding_boxes(original_image.copy(), updated_annotations)

# 증강 파라미터 설정
st.sidebar.title("Data Augmentation Parameters")
h_flip = st.sidebar.checkbox("Horizontal Flip")
v_flip = st.sidebar.checkbox("Vertical Flip")
brightness = st.sidebar.slider("Brightness (Contrast Adjustment)", min_value=1, max_value=4, step=1, value=1)
blur = st.sidebar.slider("Blur Level", min_value=1, max_value=31, step=2, value=1)

apply_crop = st.sidebar.checkbox("Center Crop 적용")
crop = st.sidebar.slider("Crop Size", min_value=50, max_value=min(original_image.size), step=10, value=100) if apply_crop else 0

hue_shift = st.sidebar.slider("색조 변화", min_value=-30, max_value=30, step=1, value=0)
sat_shift = st.sidebar.slider("채도 변화", min_value=-50, max_value=50, step=1, value=0)
val_shift = st.sidebar.slider("밝기 변화", min_value=-50, max_value=50, step=1, value=0)

# Additional Augmentation Parameters
#이미지를 고무처럼 늘리고 구부려서 자연스러운 왜곡을 추가(필기체 인식처럼 손으로 작성된 글씨가 휘어지거나 삐뚤어질 수 있는 상황에서 유용
apply_elastic = st.sidebar.checkbox("Elastic Transformation 적용")
alpha = st.sidebar.slider("Elastic Alpha", min_value=1, max_value=1000, step=10, value=50) if apply_elastic else 0 #변형의 강도
sigma = st.sidebar.slider("Elastic Sigma", min_value=1, max_value=100, step=1, value=10) if apply_elastic else 0   #변형을 부드럽게 하는 정도

apply_perspective = st.sidebar.checkbox("Perspective Transformation 적용(이미지가 특정 각도에서 촬영된 것처럼)")
scale = st.sidebar.slider("Perspective Scale", min_value=0.1, max_value=0.5, step=0.05, value=0.2) if apply_perspective else 0

apply_rotation = st.sidebar.checkbox("Rotation 적용")
rotate_limit = st.sidebar.slider("Rotation Angle", min_value=-90, max_value=90, step=1, value=0) if apply_rotation else 0

apply_dilation = st.sidebar.checkbox("Dilation(팽창) 적용")
dilation_kernel_size = st.sidebar.slider("Dilation Kernel Size", min_value=1, max_value=31, step=2, value=3) if apply_dilation else 0

# 증강 적용을 위해 이미지를 numpy 배열로 변환
image_np = np.array(original_image)
bboxes = [word['points'] for word in updated_annotations.values()]

# 증강 적용
augmented_image_np, augmented_bboxes = augment_image(
    image=image_np,
    bboxes=bboxes,
    h_flip=h_flip,
    v_flip=v_flip,
    brightness=brightness,
    blur=blur,
    crop=crop,
    hue_shift=hue_shift,
    sat_shift=sat_shift,
    val_shift=val_shift,
    alpha=alpha,
    sigma=sigma,
    scale=scale,
    rotate_limit=rotate_limit,
    dilation_kernel_size=dilation_kernel_size
)

# 증강된 이미지의 바운딩 박스 정보 업데이트
augmented_annotations = {}
for (word_id, word_data), new_bbox in zip(updated_annotations.items(), augmented_bboxes):
    augmented_annotations[word_id] = {
        "transcription": word_data["transcription"],
        "points": new_bbox
    }

# 증강된 이미지 Pillow로 변환 후 바운딩 박스 그리기
augmented_image = Image.fromarray(augmented_image_np)
drawn_augmented_image = draw_bounding_boxes(augmented_image.copy(), augmented_annotations)

# 원본 이미지와 증강된 이미지 나란히 표시
col1, col2 = st.columns(2)

with col1:
    st.subheader("원본 이미지")
    st.image(drawn_original_image)

with col2:
    st.subheader("증강 적용된 이미지")
    st.image(drawn_augmented_image)

# 어노테이션 정보 표시
st.subheader("Updated Annotations")
st.json(augmented_annotations)

