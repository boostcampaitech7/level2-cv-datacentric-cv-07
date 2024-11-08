import os
import json
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from trdg.generators import GeneratorFromStrings
import argparse
from scipy.ndimage import gaussian_filter, map_coordinates

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Data synthesis for Japanese receipt")
    parser.add_argument("--output_dir", type=str, default="./synth_data/japanese_receipt", help="Output directory")
    parser.add_argument("--json_path", type=str, default="./train.json", help="Path to JSON file")
    parser.add_argument("--language", type=str, default="ja", help="Language for text generation")
    parser.add_argument("--fonts", nargs='+', help="List of font paths")
    parser.add_argument("--language_save", type=str, default="zh", help="Language for save name")
    return parser.parse_args()

# Set up directories for output
args = parse_args()
output_dir = args.output_dir
ufo_dir = output_dir
image_dir = os.path.join(output_dir, "img")
os.makedirs(ufo_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

# Elastic transformation function
def apply_elastic_transform(image, alpha=20, sigma=4):
    random_state = np.random.RandomState(None)
    shape = image.shape[:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    if image.ndim == 3:
        transformed_image = np.zeros_like(image)
        for i in range(image.shape[2]):
            transformed_image[..., i] = map_coordinates(image[..., i], indices, order=1, mode='reflect').reshape(shape)
    else:
        transformed_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

    return transformed_image

# Adding noise to the background for areas outside the document mask
def add_noise_to_background(image_array, mask):
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    noise = np.random.randint(0, 100, image_array.shape, dtype=np.uint8)
    blurred_noise = cv2.GaussianBlur(noise, (5, 5), 0)
    background_with_noise = np.where(mask == 0, blurred_noise, image_array)
    return background_with_noise

# Text file creation function
def create_text_file(json_path, idx):
    arrs = []
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    for image_name, image_info in json_data['images'].items():
        for word_id, word_info in image_info['words'].items():
            text = word_info.get('transcription', "")
            if text is None or text == "":
                text = "-" * 40
            arrs.append(text)

    arrs = [text for text in arrs if text is not None]

    if len(arrs) < 100:
        print(f"Warning: Not enough text data for {idx}. Only {len(arrs)} items available.")

    arrs_permuted = np.random.permutation(arrs)
    result_text = '  '.join(arrs_permuted[:100]) if len(arrs_permuted) >= 100 else '  '.join(arrs_permuted)

    file_name = os.path.join(image_dir, f"words_{idx}.txt")
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(result_text)

    return file_name

# Function to generate word images
def get_words(text_file, count=5):
    with open(text_file, "r", encoding="utf-8") as f:
        paragraphs = f.readlines()

    sentences = " ".join([x.strip() for x in paragraphs]).split()
    sizes = [55, 60, 64, 68]
    fonts = args.fonts

    words = []
    for sentence in sentences:
        if sentence == '-':
            continue
        
        size = 40 if '---' in sentence else random.choice(sizes)
        font_path = random.choice(fonts)
        
        if "---" in sentence:
            width, height = 500, size
            image = Image.new("RGB", (width, height), color=(255, 255, 255))
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(font_path, size)
            draw.text((10, 0), sentence, fill="black", font=font)
            words.append({"patch": image, "text": sentence, "size": (width, height), "margin": 10})
            continue

        generator = GeneratorFromStrings(
            [sentence],
            language=args.language,
            size=size,
            count=1,
            fonts=[font_path],
            margins=(10, 10, 10, 10),
            character_spacing=random.randint(15, 25),
            fit=True,
            blur=2,
            random_blur=True,
            background_type=2
        )

        for patch, text in generator:
            max_width = 1080
            if patch.size[0] > max_width:
                aspect_ratio = patch.size[1] / patch.size[0]
                patch = patch.resize((max_width, int(max_width * aspect_ratio)), Image.ANTIALIAS)

            words.append({"patch": patch, "text": text, "size": patch.size, "margin": 10})
            if len(words) >= count:
                return words
    return words

# 문서 생성 함수
def make_document(words, width=1080, height=1920):
    image = Image.fromarray(np.random.normal(230, 6, (height, width, 3)).astype(np.uint8))
    x, y = 0, 0

    for word in words:
        patch = word["patch"]
        size = word["size"]
        margin = word["margin"]

        # x 좌표가 너비를 넘으면 다음 줄로 이동
        if x + size[0] > width:
            x = 0
            y += size[1] + margin

        # y 좌표가 높이를 넘지 않도록 제한
        if y + size[1] > height:
            print("Warning: Text goes beyond document height, stopping further placement.")
            break  # 이미지 하단을 넘어갈 경우 텍스트 추가 중단

        word["bbox"] = [x, y, x + size[0], y, x + size[0], y + size[1], x, y + size[1]]
        image.paste(patch, (x, y))
        x += size[0] + margin

    # Bounding box가 설정된 단어들만 반환
    words_with_bbox = [word for word in words if "bbox" in word]
    return {"image": image, "words": words_with_bbox}

# Document augmentation with elastic and perspective transformations
def perturb_document_inplace(document, pad=0, color=(64, 64, 64), add_noise=True, apply_elastic=True):
    width, height = document["image"].size
    src = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    
    # Perspective Transformation
    magnitude_lb, magnitude_ub = 0, 200
    perturb = np.random.uniform(magnitude_lb, magnitude_ub, (4, 2)).astype(np.float32)
    perturb *= np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    dst = src + perturb
    M = cv2.getPerspectiveTransform(src, dst)

    transformed_image = cv2.warpPerspective(np.array(document["image"]), M, (width, height), flags=cv2.INTER_LINEAR, borderValue=color)
    
    if add_noise:
        mask = cv2.warpPerspective(np.ones((height, width), dtype=np.uint8), M, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
        transformed_image = add_noise_to_background(transformed_image, mask)

    # Apply Elastic Transformation
    if apply_elastic:
        transformed_image = apply_elastic_transform(transformed_image, alpha=20, sigma=4)

    document["image"] = Image.fromarray(transformed_image)

    # 'bbox'가 있는 단어들에 대해서만 변환 적용
    for word in [w for w in document["words"] if "bbox" in w]:
        word["bbox"] = transform_bbox(word["bbox"], M)

    return document

# Transform bounding box with perspective matrix
def transform_bbox(bbox, M):
    v = np.array(bbox).reshape(-1, 2).astype(np.float32).T
    v = np.vstack([v, np.ones((1, 4))])
    transformed_v = M @ v
    transformed_v = transformed_v[:2] / transformed_v[2]
    return transformed_v.T.flatten().tolist()

# Save to UFO format with only the filename in JSON
def save_to_ufo_format(document, img_name, ufo_data):
    file_name = os.path.basename(img_name)  # Extract just the filename
    ufo_data["images"][file_name] = {
        "paragraphs": {},
        "words": {},
        "chars": {},
        "img_w": document["image"].width,
        "img_h": document["image"].height,
        "tags": [],
        "relations": {}
    }

    for i, word in enumerate(document["words"]):
        ufo_data["images"][file_name]["words"][f"{i+1:04}"] = {
            "transcription": word["text"],
            "points": [[word["bbox"][j], word["bbox"][j+1]] for j in range(0, 8, 2)]
        }

# JSON path and setup for data synthesis
json_path = args.json_path
ufo_data = {"images": {}}

for idx in range(40):
    text_file = create_text_file(json_path, idx)
    words = get_words(text_file, count=100)
    document = make_document(words)
    language_save = args.language_save
    
    # 기본 이미지 저장 및 UFO 형식 저장
    img_name = os.path.join(image_dir, f"{language_save}_document_{idx}.jpg")
    document["image"].save(img_name)
    save_to_ufo_format(document, f"{language_save}_document_{idx}.jpg", ufo_data)  # 기본 이미지에 대한 bbox 저장

    # Augmented 이미지 생성 및 저장
    for angle in range(4):
        apply_elastic = angle < 2  # 첫 두 장에만 elastic 적용
        aug_doc = perturb_document_inplace(make_document(words), apply_elastic=apply_elastic)
        aug_img_name = os.path.join(image_dir, f"{language_save}_document_{idx}_aug_{angle}.jpg")
        aug_doc["image"].save(aug_img_name)
        save_to_ufo_format(aug_doc, f"{language_save}_document_{idx}_aug_{angle}.jpg", ufo_data)

# Save UFO JSON data with bbox information for all images
ufo_json_path = os.path.join(ufo_dir, "ufo_data.json")
with open(ufo_json_path, "w", encoding="utf-8") as f:
    json.dump(ufo_data, f, ensure_ascii=False, indent=4)
