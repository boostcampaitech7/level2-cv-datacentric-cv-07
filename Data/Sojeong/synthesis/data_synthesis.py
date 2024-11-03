import os
import json
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from trdg.generators import GeneratorFromStrings

# Set up directories for output
output_dir = "./synth_data/chinese_receipt"
ufo_dir = os.path.join(output_dir, "ufo")
image_dir = os.path.join(output_dir, "images")
os.makedirs(ufo_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

# Text file creation function
def create_text_file(json_path, idx):
    arrs = []
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    for image_name, image_info in json_data['images'].items():
        for word_id, word_info in image_info['words'].items():
            text = word_info.get('transcription', "")
            if text is None or text == "":
                text = "-" * 10  # Default value for empty text or None
            arrs.append(text)

    arrs = [text for text in arrs if text is not None]

    if len(arrs) < 100:
        print(f"Warning: Not enough text data for {idx}. Only {len(arrs)} items available.")

    arrs_permuted = np.random.permutation(arrs)
    result_text = '  '.join(arrs_permuted[:100]) if len(arrs_permuted) >= 100 else '  '.join(arrs_permuted)

    file_name = os.path.join(image_dir, f"chinese_words_{idx}.txt")
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(result_text)

    return file_name

# Function to generate word images
def get_words(text_file, count=5):
    with open(text_file, "r", encoding="utf-8") as f:
        paragraphs = f.readlines()

    sentences = " ".join([x.strip() for x in paragraphs]).split()
    sizes = [55, 60, 64, 68]
    fonts = [
        "/data/ephemeral/home/Sojeong/level2-cv-datacentric-cv-07/Data/Sojeong/synthesis/font/cn/SourceHanSans-Normal.ttf", 
        "/data/ephemeral/home/Sojeong/level2-cv-datacentric-cv-07/Data/Sojeong/synthesis/font/cn/ChillHuoFangKai_F_Regular.otf", 
        "/data/ephemeral/home/Sojeong/level2-cv-datacentric-cv-07/Data/Sojeong/synthesis/font/cn/ChillHuoKai_Regular.otf",
        "/data/ephemeral/home/Sojeong/level2-cv-datacentric-cv-07/Data/Sojeong/synthesis/font/cn/Slideyouran-Regular.ttf",
        "/data/ephemeral/home/Sojeong/level2-cv-datacentric-cv-07/Data/Sojeong/synthesis/font/cn/ChillLongCangKaiShu_Regular.otf"
    ]

    words = []
    for sentence in sentences:
        size = 40 if '-' in sentence else random.choice(sizes)
        font_path = random.choice(fonts)

        if all(c == '-' for c in sentence):
            width, height = 300, size
            image = Image.new("RGB", (width, height), color=(255, 255, 255))
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(font_path, size)
            draw.text((10, 0), sentence, fill="black", font=font)
            words.append({"patch": image, "text": sentence, "size": (width, height), "margin": 10})
            continue

        generator = GeneratorFromStrings(
            [sentence],
            language="cn",
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

# Document creation function
def make_document(words, width=1080, height=1920):
    image = Image.fromarray(np.random.normal(230, 6, (height, width, 3)).astype(np.uint8))
    x, y = 0, 0

    for word in words:
        patch = word["patch"]
        size = word["size"]
        margin = word["margin"]

        if x + size[0] > width:
            x = 0
            y += size[1] + margin

        word["bbox"] = [x, y, x + size[0], y, x + size[0], y + size[1], x, y + size[1]]
        image.paste(patch, (x, y))
        x += size[0] + margin

    return {"image": image, "words": words}

# Adding noise to the background
def add_noise_to_background(image_array, mask):
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    noise = np.random.randint(0, 100, image_array.shape, dtype=np.uint8)
    blurred_noise = cv2.GaussianBlur(noise, (5, 5), 0)
    background_with_noise = np.where(mask == 0, blurred_noise, image_array)
    return background_with_noise

# Document augmentation with perspective transformation
def perturb_document_inplace(document, pad=0, color=(64, 64, 64), add_noise=True):
    width, height = document["image"].size
    src = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    
    magnitude_lb, magnitude_ub = 0, 200
    perturb = np.random.uniform(magnitude_lb, magnitude_ub, (4, 2)).astype(np.float32)
    perturb *= np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    dst = src + perturb

    M = cv2.getPerspectiveTransform(src, dst)
    transformed_image = cv2.warpPerspective(np.array(document["image"]), M, (width, height), flags=cv2.INTER_LINEAR, borderValue=color)
    if add_noise:
        mask = cv2.warpPerspective(np.ones((height, width), dtype=np.uint8), M, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
        transformed_image = add_noise_to_background(transformed_image, mask)

    document["image"] = Image.fromarray(transformed_image)

    for word in document["words"]:
        word["bbox"] = transform_bbox(word["bbox"], M)

    return document

# Transform bounding box with perspective matrix
def transform_bbox(bbox, M):
    v = np.array(bbox).reshape(-1, 2).astype(np.float32).T
    v = np.vstack([v, np.ones((1, 4))])
    transformed_v = M @ v
    transformed_v = transformed_v[:2] / transformed_v[2]
    return transformed_v.T.flatten().tolist()

# Save to UFO format
def save_to_ufo_format(document, img_name, ufo_data):
    ufo_data["images"][img_name] = {
        "paragraphs": {},
        "words": {},
        "chars": {},
        "img_w": document["image"].width,
        "img_h": document["image"].height,
        "tags": [],
        "relations": {}
    }

    for i, word in enumerate(document["words"]):
        ufo_data["images"][img_name]["words"][f"{i+1:04}"] = {
            "transcription": word["text"],
            "points": [[word["bbox"][j], word["bbox"][j+1]] for j in range(0, 8, 2)]
        }

# Path and setup for JSON
json_path = "/data/ephemeral/home/data/chinese_receipt/ufo/train.json"
ufo_data = {"images": {}}

for idx in range(1):
    text_file = create_text_file(json_path, idx)
    words = get_words(text_file, count=100)
    document = make_document(words)

    img_name = os.path.join(image_dir, f"document_{idx}.jpg")
    document["image"].save(img_name)

    for angle in range(25):
        aug_doc = perturb_document_inplace(make_document(words))
        aug_img_name = os.path.join(image_dir, f"document_{idx}_aug_{angle}.jpg")
        aug_doc["image"].save(aug_img_name)
        save_to_ufo_format(aug_doc, aug_img_name, ufo_data)

# Save UFO JSON data
ufo_json_path = os.path.join(ufo_dir, "ufo_data.json")
with open(ufo_json_path, "w", encoding="utf-8") as f:
    json.dump(ufo_data, f, ensure_ascii=False, indent=4)


# python /data/ephemeral/home/Sojeong/level2-cv-datacentric-cv-07/Data/Sojeong/synthesis/data_synthesis.py