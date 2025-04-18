import os
import cv2
import mmcv
import torch
import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet.apis import inference_detector, init_detector
from mmrotate.visualization import RotLocalVisualizer
from mmcv.ops import nms_rotated
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def apply_rotated_nms(bboxes, scores, labels, nms_iou_threshold=0.1):
    """Applies Rotated Non-Maximum Suppression (R-NMS) to filter overlapping detections."""
    print(f'Applying rotated NMS with {len(bboxes)} bboxes')
    if len(bboxes) == 0:
        print('No bounding boxes to apply NMS on.')
        return [], [], []

    # Convert numpy arrays to tensors
    bboxes = torch.tensor(bboxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64)

    # Apply rotated NMS
    _, keep_indices = nms_rotated(bboxes, scores, nms_iou_threshold)
    print(f'Keep indices after NMS: {keep_indices}')

    # Convert tensors back to numpy
    bboxes = bboxes[keep_indices].numpy()
    scores = scores[keep_indices].numpy()
    labels = labels[keep_indices].numpy()

    print(f'Kept {len(bboxes)} detections after NMS.')
    return bboxes, scores, labels


def save_inference(image_dir, model, inf_dir, class_mapping, confidence_threshold=0.25, apply_nms=True, nms_iou_threshold=0.1, standardize_points=False, img_size=640, save_images=False, save_ann=False, save_ann_format='dota'):
    os.makedirs(inf_dir, exist_ok=True)
    image_files = os.listdir(image_dir)
    image_files = sorted(image_files)
    num_images = len(image_files)
    print(f'Found {num_images} images in {image_dir}')

    visualizer = RotLocalVisualizer()

    for i, image_name in enumerate(image_files):
        print(f'Processing image {i + 1}/{num_images}: {image_name}')
        image_path = os.path.join(image_dir, image_name)
        result = inference_detector(model, image_path)

        bboxes = result.pred_instances.bboxes.cpu().numpy()
        scores = result.pred_instances.scores.cpu().numpy()
        labels = result.pred_instances.labels.cpu().numpy()

        print(f'Initial detections: {len(bboxes)} bboxes, {len(scores)} scores, {len(labels)} labels')

        # Filter detections based on confidence threshold
        mask = scores >= confidence_threshold
        bboxes, scores, labels = bboxes[mask], scores[mask], labels[mask]
        print(f'Detections after confidence threshold: {len(bboxes)} bboxes, {len(scores)} scores, {len(labels)} labels')

        if apply_nms:
            print(f'Applying NMS with IOU threshold of {nms_iou_threshold}')
            bboxes, scores, labels = apply_rotated_nms(bboxes, scores, labels, nms_iou_threshold=nms_iou_threshold)
            print(f'Detections after NMS: {len(bboxes)} bboxes, {len(scores)} scores, {len(labels)} labels')

        # Save the visualized test images with detections
        if save_images:
            inf_img_dir = os.path.join(inf_dir, 'images')
            os.makedirs(inf_img_dir, exist_ok=True)
            visualizer.add_datasample(
                image_name,
                mmcv.imconvert(mmcv.imread(image_path), 'bgr', 'rgb'),
                result,
                show=False,
                draw_gt=False,
                out_file=os.path.join(inf_img_dir, image_name)
            )
            print(f'Saved visualized image: {os.path.join(inf_img_dir, image_name)}')

        # Save annotations in DOTA format
        if save_ann:
            inf_ann_dir = os.path.join(inf_dir, 'annfiles')
            os.makedirs(inf_ann_dir, exist_ok=True)
            ann_file = os.path.join(inf_ann_dir, f'{os.path.splitext(image_name)[0]}.txt')
            print(f'Saving annotation file: {ann_file}')
            with open(ann_file, 'w') as f:
                for bbox, score, label in zip(bboxes, scores, labels):
                    # Convert (cx, cy, w, h, angle) to DOTA format
                    cx, cy, w, h, angle = bbox  # Rotated bbox format: (center_x, center_y, width, height, angle)
                    points = cv2.boxPoints(((cx, cy), (w, h), np.degrees(angle)))  # Get 4 corner points
                    points = points.flatten()

                    if standardize_points:
                        points = points / img_size
                        print(f'Standardized points: {points}')

                    if replace_class_with_label:
                        label = class_mapping.get(label, 'Unknown')
                    else:
                        label = str(label)
                    
                    if save_ann_format == 'dota':
                        f.write(f"{' '.join(map(str, points))} {label} {score}\n")
                    elif save_ann_format == 'supervision':
                        f.write(f"{label} {' '.join(map(str, points))} {score}\n")
            print(f'Saved annotation file: {ann_file}')

    print('Done.')


# Configuration files
model_configs = [
    # {   # Bihar to Bihar
    #     'train': 'Train Bihar',
    #     'test': 'Test Bihar',
    #     'backbone': 'swint',
    #     'head': 'rhino',
    #     'config_file': 'configs-mine/rhino-swint-dota2config/rhino_phc_haus-4scale_swint_2xb2-36e_bihar.py',
    #     'checkpoint_folder': 'work_dirs/rhino_phc_haus-4scale_swint_2xb2-36e_bihar',
    #     'val_dir': 'data/bihar/val',
    #     'inf_dir': 'results-swint/train_bihar_test_bihar',
    #     'img_height': 640,
    #     'epoch': 50,
    # },
    #     {
    #     'train': 'Delhi NCR',
    #     'test': 'West Bengal',
    #     'backbone': 'swint',
    #     'head': 'rhino',
    #     'config_file': '/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/RHINO/configs-mine/rhino-swint-dota2config/rhino_phc_haus-4scale_swint_2xb2-36e_delhi_ncr.py',
    #     'checkpoint_folder': '/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/RHINO/work_dirs/rhino_phc_haus-4scale_swint_2xb2-36e_delhi_ncr',
    #     'checkpoint_file': '/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/RHINO/work_dirs/rhino_phc_haus-4scale_swint_2xb2-36e_delhi_ncr/epoch_50.pth',
    #     'val_dir': '/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/RHINO/data/grid_data/wb_small_airshed',
    #     'inf_dir': '/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/RHINO/results-swint/delhi_to_wb',
    #     'img_height': 640,
    #     'epoch': 50,
    # },
    {
        'train': 'Gen Delhi NCR (CG)',
        'test': 'West Bengal',
        'backbone': 'swint',
        'head': 'rhino',
        'config_file': '/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/RHINO/configs-mine/rhino-swint-dota2config/rhino_phc_haus-4scale_swint_2xb2-36e_gen_delhi_CG_bks.py',
        'checkpoint_folder': '/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/RHINO/work_dirs/rhino_phc_haus-4scale_swint_2xb2-36e_gen_delhi_CG_bks',
        'checkpoint_file': '/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/RHINO/work_dirs/rhino_phc_haus-4scale_swint_2xb2-36e_gen_delhi_CG_bks/epoch_50.pth',
        'val_dir': '/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/RHINO/data/grid_data/wb_small_airshed',
        'inf_dir': '/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/RHINO/results-swint/gen_delhi_to_wb',
        'img_height': 640,
        'epoch': 50,
    }
]


save_images = False
save_ann = True
save_ann_format = 'supervision'
standardize_points = True
replace_class_with_label = False
class_mapping = {0: "CFCBK", 1: "FCBK", 2: "Zigzag"}
apply_nms = True
nms_iou_threshold = 0.33
confidence_threshold = 0.01


for model_config in model_configs:
    num_epochs = model_config['epoch']
    for i in range(1, num_epochs + 1)[::-1]:
        print(f'Epoch {i}/{num_epochs}')

        # Load the configuration file
        cfg = Config.fromfile(model_config['config_file'])
        init_default_scope(cfg.get('default_scope', 'mmrotate'))
        
        # Initialize the model
        model_path = os.path.join(model_config['checkpoint_folder'], f'epoch_{i}.pth')
        print(f'Loading model from {model_path}')
        model = init_detector(cfg, model_path, device='cuda:0')
        
        # Set the save directory
        save_dir = os.path.join(model_config['inf_dir'], f'epoch_{i}_supervision_conf_{confidence_threshold}_nms_{nms_iou_threshold}')
        os.makedirs(save_dir, exist_ok=True)
        
        # Perform inference and save results
        save_inference(
            image_dir = os.path.join(model_config['val_dir'], 'images'),
            model = model,
            inf_dir = save_dir,
            class_mapping = class_mapping,
            confidence_threshold = confidence_threshold,
            apply_nms = apply_nms,
            nms_iou_threshold = nms_iou_threshold,
            standardize_points = standardize_points,
            img_size = model_config['img_height'],
            save_images = save_images,
            save_ann = save_ann,
            save_ann_format = save_ann_format
        )
        
        # Check the number of files in image and annotation directories
        img_files_dir = os.path.join(model_config['val_dir'], 'images')
        print(f'Number of images in {img_files_dir}: {len(os.listdir(img_files_dir))}')
        if save_images:
            inf_imgs_dir = os.path.join(save_dir, 'images')
            print(f'Number of saved images in {inf_imgs_dir}: {len(os.listdir(inf_imgs_dir))}')
        if save_ann:
            ann_files_dir = os.path.join(save_dir, 'annfiles')
            print(f'Number of annotation files in {ann_files_dir}: {len(os.listdir(ann_files_dir))}')
        
        print('Done processing epoch:', i)
        print('-----------------------------------')
