import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

from ultralytics import YOLO

# Load the YOLO11m-OBB model
model = YOLO("yolo11m-obb.pt")  # Load pre-trained model

# Train the model with custom parameters
results = model.train(
    data="/home/shardul.junagade/my-work/domain-adaptation-brick-kilns/yolo/yamls/haryana_to_test_bihar.yaml",  # Path to the dataset YAML file
    epochs=100,
    imgsz=2560,
    batch=8,
    iou=0.33,
    conf=0.001,
    device=device,
    val=True,  # Enables validation
    project = "models",
    name = "yolo11m-obb-bihar-to-haryana"
)


# conda activate shardul_env
# export CUDA_VISIBLE_DEVICES=0
# echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# cd yolo
# nohup python ./yolo_train.py > ./logs/haryana_to_test_bihar.log 2>&1 &