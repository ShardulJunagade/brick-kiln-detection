# Description: This script saves the inference results in the DOTA format WITHOUT applying NMS.


import os
import cv2
import mmcv
import numpy as np
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet.apis import inference_detector, init_detector
from mmrotate.visualization import RotLocalVisualizer
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

# Configuration and checkpoint files
config_file = 'configs-mine/rhino/rhino_phc_haus-4scale_r50_2xb2-36e_m0.py'
checkpoint_file = 'work_dirs/rhino_phc_haus-4scale_r50_2xb2-36e_m0/epoch_50.pth'
image_dir = 'data/m0/val/images'
inf_img_dir = 'results/m0/val/images'
inf_ann_dir = 'results/m0/val/annfiles'
save_images = True
save_annotations = True
standardize_points = False
img_height = 640
img_width = 640
replace_class_with_label = False
class_mapping = {0: "CFCBK", 1: "FCBK", 2: "Zigzag"}

# Load the configuration file
cfg = Config.fromfile(config_file)
init_default_scope(cfg.get('default_scope', 'mmrotate'))
# Initialize the model
model = init_detector(cfg, checkpoint_file, device='cuda:0')
# Initialize the visualizer
visualizer = RotLocalVisualizer()

def save_inference(image_dir, inf_img_dir, inf_ann_dir, model, visualizer, class_mapping, standardize_points=False, img_size=640, confidence_threshold=0.25, save_images=True):
    os.makedirs(inf_img_dir, exist_ok=True)
    os.makedirs(inf_ann_dir, exist_ok=True)

    image_files = os.listdir(image_dir)
    image_files = sorted(image_files)
    num_images = len(image_files)

    for i, image_name in enumerate(image_files):
        print(f'Processing image {i + 1}/{num_images}...')
        image_path = os.path.join(image_dir, image_name)
        result = inference_detector(model, image_path)

        # Save the visualized test images if save_images is True
        if save_images:
            visualizer.add_datasample(
                image_name,
                mmcv.imread(image_path),
                result,
                show=False,
                draw_gt=False,
                out_file=os.path.join(inf_img_dir, image_name)
            )

        # Save annotations in DOTA format
        if inf_ann_dir:
            ann_file = os.path.join(inf_ann_dir, f'{os.path.splitext(image_name)[0]}.txt')
            
            with open(ann_file, 'w') as f:
                bboxes = result.pred_instances.bboxes.cpu().numpy()
                scores = result.pred_instances.scores.cpu().numpy()
                labels = result.pred_instances.labels.cpu().numpy()

                for bbox, score, label in zip(bboxes, scores, labels):
                    if score < confidence_threshold:  # Confidence threshold to filter low-confidence detections
                        continue

                    # Convert (cx, cy, w, h, angle) to DOTA format
                    cx, cy, w, h, angle = bbox  # Rotated bbox format: (center_x, center_y, width, height, angle)
                    points = cv2.boxPoints(((cx, cy), (w, h), np.degrees(angle)))  # Get 4 corner points
                    points = points.flatten()
                    if standardize_points:
                        points = points / img_size

                    label_name = class_mapping.get(label, 'Unknown')
                    f.write(f"{' '.join(map(str, points))} {label_name} {score:.2f}\n")

    print('Done.')

# Call the function
save_inference(image_dir, inf_img_dir, inf_ann_dir, model, visualizer, class_mapping, standardize_points=standardize_points, img_size=img_height, save_images=save_images)
