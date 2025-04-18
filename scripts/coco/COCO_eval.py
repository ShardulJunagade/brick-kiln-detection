val_dota_dirs = [
    'data/bihar/val',
    'data/haryana/val_bihar',
    'data/m0/val',
    'data/swinir/test_bihar_same_class_count_10_120_1000_4x',
    'data/swinir/test_bihar_same_class_count_10_120_1000_4x'
    ]


inf_dota_dirs = [
    'results/train_bihar_test_bihar', 
    'results/train_haryana_test_bihar',
    'results/train_m0_test_m0',
    'results/train_swinirbihar_test_swinirbihar',
    'results/train_swinirharyana_test_swinirbihar'
    ]

output_files = [
    'mAPs/rhino-coco/bihar_on_bihar.json',
    'mAPs/rhino-coco/haryana_on_bihar.json',
    'mAPs/rhino-coco/m0_on_m0.json',
    'mAPs/rhino-coco/swinirbihar_on_bihar.json',
    'mAPs/rhino-coco/swinirharyana_on_bihar.json'
]


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

def compute_map(gt_json_path, pred_json_path, output_file='mAP_results.json'):
    """
    Computes mean Average Precision (mAP) between ground truth and predictions.
    
    Args:
        gt_json_path (str): Path to ground truth COCO JSON file.
        pred_json_path (str): Path to predictions COCO JSON file.
    
    Returns:
        dict: mAP scores for different IoU thresholds.
    """
    # Load COCO-formatted ground truth and predictions
    coco_gt = COCO(gt_json_path)
    coco_pred = coco_gt.loadRes(pred_json_path)

    # Initialize COCO evaluation object
    coco_eval = COCOeval(coco_gt, coco_pred, "bbox")

    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract mAP results
    map_results = {
        "AP @ IoU=0.50:0.95": coco_eval.stats[0],  # mAP
        "AP @ IoU=0.50": coco_eval.stats[1],       # AP50
        "AP @ IoU=0.75": coco_eval.stats[2],       # AP75
        "AP @ small objects": coco_eval.stats[3],
        "AP @ medium objects": coco_eval.stats[4],
        "AP @ large objects": coco_eval.stats[5],
        "AR @ IoU=0.50:0.95 (maxDets=1)": coco_eval.stats[6],
        "AR @ IoU=0.50:0.95 (maxDets=10)": coco_eval.stats[7],
        "AR @ IoU=0.50:0.95 (maxDets=100)": coco_eval.stats[8],
        "AR @ small objects": coco_eval.stats[9],
        "AR @ medium objects": coco_eval.stats[10],
        "AR @ large objects": coco_eval.stats[11],
    }
    
    with open(output_file, "w") as f:
        json.dump(map_results, f, indent=4)
    
    print(f"✅ mAP results saved to {output_file}")
    return map_results



for i in range(5):
    gt_json_path = val_dota_dirs[i] + '/coco.json'
    pred_json_path = inf_dota_dirs[i] + '/coco.json'
    output_file = output_files[i]
    map_results = compute_map(gt_json_path, pred_json_path, output_file)
    print(json.dumps(map_results, indent=4))
