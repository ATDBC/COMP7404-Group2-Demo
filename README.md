# COMP7404-Group2-Demo
This is the Group2 demo of COMP7404.
# YOLO-World Demo - Open-Vocabulary Object Detection

This repository contains a demonstration of **YOLO-World**, a state-of-the-art real-time open-vocabulary object detector. Based on the paper "[YOLO-World: Real-Time Open-Vocabulary Object Detection](https://github.com/AILab-CVC/YOLO-World)" presented at CVPR, this implementation allows detection of arbitrary objects using natural language prompts without retraining.

## Setup Instructions

### 1. Environment Setup
```bash
conda create -n yolo_world python=3.10 -y
conda activate yolo_world
pip install -r requirements.txt
```
### 2. Download Pre-trained Weights
```bash
mkdir -p weights
wget https://github.com/AILab-CVC/YOLO-World/releases/download/v2.0/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth -P weights/
```

## Demo Execution
### Interactive Gradio Interface
```bash
python demo/gradio_demo.py \configs/pretrain/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py \weights/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth
```
### Process Images
Process all images in sample_images folder:
```bash
python demo/image_demo.py \configs/pretrain/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py \weights/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth \demo/sample_images/dog.jpeg \"person,dog,eye,nose,ear,tongue,backpack" \--topk 100 \--threshold 0.1 \--output-dir demo_outputs
```
### Process Videos
```bash
python demo/video_demo.py \configs/pretrain/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py \weights/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth \demo/video/film2.mp4 \"person,person wearing red cloth" \--out demo_outputs/film2.mp4 \--score-thr 0.3
```
