import json

# Load the JSON file
with open('/data/ephemeral/home/Jihwan/data/chinese_receipt/ufo/train.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Loop through each image and add new fields
for image in data['images'].values():
    for word in image['words'].values():
        word['illegibility'] = False
        word['language'] = None
        word['tags'] = None
        word['orientation'] = "Horizontal"

# Save the modified JSON file
with open('/data/ephemeral/home/Jihwan/data/chinese_receipt/ufo/train_modified.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print("수정된 JSON 파일이 저장되었습니다.")