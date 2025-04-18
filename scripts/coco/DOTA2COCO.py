import json
import numpy as np
import os

class_mapping = {'CFCBK': 0, 'FCBK': 1, 'Zigzag': 2}

def parse_dota_file(file_path, is_prediction=False):
    """Parses a single DOTA annotation file into COCO format.
    
    Args:
        file_path (str): Path to the DOTA annotation file.
        is_prediction (bool): Whether the file contains predictions.
    
    Returns:
        list: List of annotations in COCO format.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    annotations = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 9:
            category_id = class_mapping[parts[8]] + 1  # COCO category ID (1-based index)
            score = float(parts[9]) if len(parts) == 10 else 1.0  # If available, use confidence score
            annotations.append([float(x) for x in parts[:8]] + [category_id, score])
    
    return annotations

def dota_to_coco(dota_dir, is_prediction=False):
    """Converts DOTA format annotations into COCO JSON format for GT or predictions."""
    images = []
    annotations = []
    categories = [{"id": i+1, "name": name} for i, name in enumerate(class_mapping.keys())]

    ann_id = 1  # Annotation ID counter (only for GT)
    
    for img_id, file_name in enumerate(sorted(os.listdir(dota_dir))):
        if file_name.endswith('.txt'):
            file_path = os.path.join(dota_dir, file_name)
            image_name = os.path.splitext(file_name)[0] + ".png"  
            images.append({"id": img_id, "file_name": image_name})

            dota_annotations = parse_dota_file(file_path, is_prediction=is_prediction)
            for ann in dota_annotations:
                x1, y1, x2, y2, x3, y3, x4, y4, category_id, score = ann
                x_min = min(x1, x2, x3, x4)
                y_min = min(y1, y2, y3, y4)
                x_max = max(x1, x2, x3, x4)
                y_max = max(y1, y2, y3, y4)
                width = x_max - x_min
                height = y_max - y_min

                if width <= 0 or height <= 0:
                    print(f"⚠️ Warning: Invalid bbox {x_min, y_min, width, height} in image {image_name}")

                annotation = {
                    "image_id": img_id,
                    "category_id": int(category_id),
                    "bbox": [x_min, y_min, width, height],
                    "segmentation": [[x1, y1, x2, y2, x3, y3, x4, y4]],  
                    "area": width * height
                }

                if is_prediction:
                    annotation["score"] = score  # Predictions need score
                else:
                    annotation["id"] = ann_id  # Add ID for ground truth
                    annotation["iscrowd"] = 0  # Required for GT
                    ann_id += 1

                annotations.append(annotation)

    if is_prediction:
        return annotations  # Predictions should be a list
    else:
        return {
            "images": images,
            "annotations": annotations,
            "categories": categories
        }


def save_coco_json(coco_data, save_path):
    """Saves a COCO JSON file."""
    with open(save_path, 'w') as f:
        json.dump(coco_data, f, indent=4)




train_dota_dirs = [
    'data/bihar/train',
    'data/haryana/train',
    'data/m0/train',
    'data/swinir/bihar_same_class_count_10_120_1000_4x',
    'data/swinir/haryana_same_class_count_10_120_1000_4x'
    ]


val_dota_dirs = [
    'data/bihar/val',
    'data/haryana/val_bihar',
    'data/m0/val',
    'data/swinir/test_bihar_same_class_count_10_120_1000_4x'
    ]


inf_dota_dirs = [
    'results/train_bihar_test_bihar', 
    'results/train_haryana_test_bihar',
    'results/train_m0_test_m0',
    'results/train_swinirbihar_test_swinirbihar',
    'results/train_swinirharyana_test_swinirbihar'
    ]



for dota_dirs in [train_dota_dirs, val_dota_dirs]:
    for dota_dir in  dota_dirs:
        ann_path = os.path.join(dota_dir, 'annfiles')
        coco_data = dota_to_coco(ann_path)
        save_coco_json(coco_data, os.path.join(dota_dir, 'coco.json'))


for dota_dir in inf_dota_dirs:
    ann_path = os.path.join(dota_dir, 'annfiles')
    coco_data = dota_to_coco(ann_path, True)
    save_coco_json(coco_data, os.path.join(dota_dir, 'coco.json'))
