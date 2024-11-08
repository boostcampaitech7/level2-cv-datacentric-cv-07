import json

def update_illegibility(data):
    """
    Updates illegibility field to True for items with empty transcription in nested JSON structure
    
    Args:
        data (dict): Nested JSON data containing images and words
    
    Returns:
        dict: Updated JSON data
    """
    # Iterate through all images
    for image_name, image_data in data.get("images", {}).items():
        # Process words in each image
        words = image_data.get("words", {})
        for word_id, word_info in words.items():
            if isinstance(word_info, dict):
                if word_info["transcription"] == "" or word_info["transcription"] is None or (isinstance(word_info["transcription"], str) and "-----" in word_info["transcription"]):
                        word_info["illegibility"] = True
    
    
    return data

# Example usage
if __name__ == "__main__":
    # Load your JSON data
    with open('/data/ephemeral/home/Jihwan/data/vietnamese_receipt/ufo/train_modified_vietnamese.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Update illegibility
    updated_data = update_illegibility(json_data)
    
    # Save updated JSON
    with open('/data/ephemeral/home/Jihwan/data/vietnamese_receipt/ufo/train.json', 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, ensure_ascii=False, indent=4)