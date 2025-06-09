import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

from ultralytics import YOLO

# Load the last trained model
model = YOLO("models/yolo11m-obb-delhi-to-wb/weights/last.pt")

# 🧠 Note: YOLOv8 is smart enough to resume training if you load last.pt and keep the same project and name. It'll continue from epoch 50 to 100.

# Resume training for a total of 100 epochs
results = model.train(
    data="/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/yolo/yamls/delhi_to_wb.yaml",
    epochs=100,  # total epochs you want to reach
    imgsz=2560,
    batch=8,
    iou=0.33,
    conf=0.001,
    device=device,  # or 'cuda:0', whichever you prefer
    val=True,
    project="models",
    name="yolo11m-obb-delhi-to-wb"  # same name so it continues in the same folder
)


# conda activate shardul_env
# export CUDA_VISIBLE_DEVICES=0
# echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# cd yolo

# nohup python ./yolo_train.py > ./logs/