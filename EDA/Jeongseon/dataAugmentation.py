import streamlit as st
import os
import numpy as np
from PIL import Image, ImageOps
import albumentations as A

# 화면을 넓게 설정
st.set_page_config(layout="wide")

# 데이터 경로 설정 (예시 경로입니다. 실제 경로에 맞게 수정하세요)
data_paths = {
    "중국": "/data/ephemeral/home/data/chinese_receipt/img/train",
    "일본": "/data/ephemeral/home/data/japanese_receipt/img/train",
    "태국": "/data/ephemeral/home/data/thai_receipt/img/train",
    "베트남": "/data/ephemeral/home/data/vietnamese_receipt/img/train"
}

# 이미지 로딩 함수 (EXIF 방향 정보 자동 적용)
def load_image_with_orientation(image_path):
    # EXIF 정보를 자동으로 처리하여 이미지를 올바른 방향으로 로드
    image = ImageOps.exif_transpose(Image.open(image_path))
    return image.convert("RGB")

def augment_image(image, h_flip, v_flip, brightness, blur, crop, hue_shift, sat_shift, val_shift):
    # 이미지 크기 얻기
    height, width = image.shape[:2]
    
    # 증강 리스트 생성
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
        transforms.append(
            A.HueSaturationValue(
                hue_shift_limit=(hue_shift, hue_shift),
                sat_shift_limit=(sat_shift, sat_shift),
                val_shift_limit=(val_shift, val_shift),
                p=1.0
            )
        )

    # transforms가 비어있으면 원본 이미지 반환
    if not transforms:
        return image

    transform = A.Compose(transforms)
    
    try:
        augmented = transform(image=image)
        return augmented['image']
    except ValueError as e:
        print(f"Augmentation failed: {str(e)}")
        return image

# Streamlit UI
st.title("Data Augmentation Editor")

# 나라 선택
country = st.selectbox("나라 선택", options=list(data_paths.keys()))
image_folder = data_paths[country]

# 이미지 선택
image_files = os.listdir(image_folder)
selected_image = st.selectbox("이미지 선택", options=image_files)
image_path = os.path.join(image_folder, selected_image)

# 원본 이미지 로딩
original_image = load_image_with_orientation(image_path)

# 증강 파라미터 설정
st.sidebar.title("Data Augmentation Parameters")
h_flip = st.sidebar.checkbox("Horizontal Flip")
v_flip = st.sidebar.checkbox("Vertical Flip")
brightness = st.sidebar.slider("Brightness (Contrast Adjustment)", min_value=1, max_value=4, step=1, value=1)
blur = st.sidebar.slider("Blur Level", min_value=1, max_value=31, step=2, value=1)

# Crop 설정
apply_crop = st.sidebar.checkbox("Center Crop 적용")
if apply_crop:
    crop = st.sidebar.slider("Crop Size", min_value=50, max_value=min(original_image.size), step=10, value=100)
else:
    crop = 0

# 색조, 채도, 명도 조절 슬라이더
hue_shift = st.sidebar.slider("색조 변화", min_value=-30, max_value=30, step=1, value=0)
sat_shift = st.sidebar.slider("채도 변화", min_value=-50, max_value=50, step=1, value=0)
val_shift = st.sidebar.slider("밝기 변화", min_value=-50, max_value=50, step=1, value=0)

# 증강 적용을 위해 이미지를 numpy 배열로 변환
image_np = np.array(original_image)

# 증강 적용
augmented_image_np = augment_image(
    image=image_np,
    h_flip=h_flip,
    v_flip=v_flip,
    brightness=brightness,
    blur=blur,
    crop=crop,
    hue_shift=hue_shift,
    sat_shift=sat_shift,
    val_shift=val_shift
)

# 증강된 이미지를 Pillow로 변환
augmented_image = Image.fromarray(augmented_image_np)

# 원본 이미지와 증강된 이미지 나란히 표시
col1, col2 = st.columns(2)

with col1:
    st.subheader("원본 이미지")
    st.image(original_image)

with col2:
    st.subheader("증강 적용된 이미지")
    st.image(augmented_image)