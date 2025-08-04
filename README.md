<!-- # brick-kilns-detection -->

# Brick Kiln Detection and Domain Adaptation from Satellite Imagery
**Sustainability Lab, IIT Gandhinagar** | [Submitted to NeurIPS 2025]

## Project Overview
This repository contains an end-to-end pipeline for detecting brick kilns in satellite imagery, supporting air pollution monitoring across the Indo-Gangetic Plain. The project leverages state-of-the-art object detection and domain adaptation techniques, with a focus on robust cross-region generalization.

## Key Components
- **Detection Models:**
	- [RHINO](./RHINO/): Rotation Detection Transformer (Hausdorff distance in DINO) for rotated object detection.
	- [YOLO](./yolo/): Baseline detector for comparison.
	- [MMRotate](./RHINO/mmrotate/): Core library for rotated object detection.
- **Super-Resolution:**
	- [Thera](./thera/): Deep learning-based super-resolution models (SwinIR, EDSR, RDN, etc.) to enhance image quality.
- **Evaluation & Mapping:**
	- [Scripts](./scripts/): Utilities for dataset conversion, evaluation, and visualization.
	- [mAP Calculation](./RHINO/map/), [YOLO mAP](./yolo/map/), [RFDETR Results](./rfdetr/): Performance metrics and results.

## Results
- **RHINO (Hausdorff-DINO):**
	- 69 mAP (same-region)
	- 53 mAP (cross-region)
	- Outperforms YOLOv11 baseline by up to 30%.
- **Super-Resolution (SwinIR, Thera):**
	- Boosts RHINO accuracy to 78 mAP (same-region) and 61 mAP (cross-region)
	- Gains of 12% and 7% respectively over non-SR pipeline.

## Folder Structure
- `RHINO/`: RHINO model configs, training scripts, evaluation, and results.
- `rfdetr/`: RFDETR experiments, COCO data, and evaluation notebooks.
- `thera/`: Super-resolution models and evaluation scripts.
- `yolo/`: YOLO training, evaluation, and configs.
- `scripts/`: Data conversion, evaluation, and visualization utilities.
- `poster/`: Project presentation materials.

## Getting Started
See individual folder READMEs for setup and usage instructions:
- [RHINO README](./RHINO/README.md)
- [RFDETR README](./rfdetr/README.md)
- [Thera README](./thera/README.md)
- [YOLO README](./yolo/README.md)

## Citation
If you use this codebase, please cite the NeurIPS 2025 submission (details to be updated).

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.