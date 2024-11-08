# !pip install datasets
from datasets import load_dataset
import json
import os

def load_image(dataset):
    train_img_data = dataset['train']['image']
    test_img_data = dataset['test']['image']
    validation_img_data = dataset['validation']['image']
    img_data = train_img_data + test_img_data + validation_img_data
    
    return img_data

def load_ground_truth(dataset):
    train_img_gt = dataset['train']['ground_truth']
    test_img_gt = dataset['test']['ground_truth']
    validation_img_gt = dataset['validation']['ground_truth']
    img_gt = train_img_gt + test_img_gt + validation_img_gt

    return img_gt

def save_images(dataset):
    img_data = load_image(dataset)
    for i in range(len(img_data)):
        output_path = os.path.join('img', f"cord_{i+1}.jpg")
        img_data[i].save(output_path, "JPEG")
    return
    
    
def convert_to_UFO(dataset):
    img_data = load_image(dataset)
    img_gt = load_ground_truth(dataset)
    cord = {}
    cord['images'] = {}
    images = cord['images']

    for i in range(len(img_data)):
        ground_truth = json.loads(img_gt[i])
        images[f"cord_{i+1}.jpg"] = {}
        curr_image = images[f"cord_{i+1}.jpg"]
        curr_image["paragraphs"] = {}
        curr_image["words"] = {}
        image_count = 0
    
        valid_line = ground_truth['valid_line']
        for j in range(len(valid_line)):
            for k in range(len(valid_line[j]["words"])):
                image_count += 1
                curr_word = valid_line[j]['words'][k]
                curr_quad = curr_word['quad']
                curr_image["words"][image_count] = {
                    "transcription": curr_word['text'],
                    "points":[ [curr_quad['x1'], curr_quad['y1']], 
                            [curr_quad['x2'], curr_quad['y2']], 
                            [curr_quad['x3'], curr_quad['y3']], 
                            [curr_quad['x4'], curr_quad['y4']] ]
                }
        curr_image["chars"] = {}
        curr_image["img_w"] = ground_truth['meta']['image_size']['width']
        curr_image["img_h"] = ground_truth['meta']['image_size']['height']
        curr_image["num_patches"] = None
        curr_image["tags"] = []
        curr_image["relations"] = {}
        curr_image["annotation_log"] = {
            "worker": "worker",
            "timestamp": "2022-07-20",
            "tool_version": "",
            "source": None
        }
        curr_image["license_tag"] = {
        "usability": True,
        "public": True,
        "commercial": True,
        "type": None,
        "holder": "Naver Clova"
        }
        
    with open('your annotation path', 'w', encoding='utf-8') as f:
        json.dump(cord, f, indent=4)
    return

if __name__ == "__main__" :
    dataset = load_dataset("naver-clova-ix/cord-v2")
    save_images(dataset)
    convert_to_UFO(dataset)