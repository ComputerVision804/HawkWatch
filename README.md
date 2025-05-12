# HawkWatch
Drone YOLO Object Detection 📦🛩️
This repository contains a full pipeline for training a YOLOv8 model on custom drone footage data for object detection and line-crossing analysis. The implementation uses the Ultralytics YOLOv8 framework and is tailored for lightweight and fast inference using the YOLOv8 Nano variant.

🧠 Project Overview
This project trains a YOLOv8 model to detect objects from drone footage. The detections can later be used for advanced tasks like object counting, trajectory tracking, and boundary-line crossing analytics.

📁 Directory Structure
The dataset is organized as per the YOLO format. Below is the visual layout:
![ChatGPT Image May 11, 2025, 06_34_02 AM](https://github.com/user-attachments/assets/c026a516-1940-40cf-8871-b1931c33e4b8)

drone_yolo_dataset/
├── train/
│   ├── images/   → Training images
│   └── labels/   → Corresponding YOLO labels
├── val/
│   ├── images/   → Validation images
│   └── labels/   → Corresponding YOLO labels
├── test/
│   ├── images/   → Test images
│   └── labels/   → Corresponding YOLO labels
data.yaml         → Configuration file linking datasets and class names
📄 data.yaml Configuration
This file defines the dataset paths and class names for the model:

yaml
train: drone_yolo_dataset/train/images
val: drone_yolo_dataset/val/images
test: drone_yolo_dataset/test/images

nc: 1  # Number of classes
names: ['drone_object']  # Class names
Make sure this file is in the project root and correctly references the dataset folder paths.

🚀 Training Script
A minimal script to train the model using Ultralytics:

from ultralytics import YOLO
import os

# Create dataset structure if not present
required_dirs = [
    "drone_yolo_dataset/train/images", "drone_yolo_dataset/train/labels",
    "drone_yolo_dataset/val/images", "drone_yolo_dataset/val/labels",
    "drone_yolo_dataset/test/images", "drone_yolo_dataset/test/labels"
]
for d in required_dirs:
    os.makedirs(d, exist_ok=True)

# Load YOLOv8 Nano model
model = YOLO("yolov8n.yaml")

# Train the model
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="drone_yolo_model",
    project="runs/train",
    workers=4,
    val=True
)

# Evaluate the model
metrics = model.val(data="data.yaml", split="test")
print("Test metrics:", metrics)

# Run predictions
results = model.predict(source="drone_yolo_dataset/test/images", save=True)
✅ Requirements

pip install ultralytics
Ensure that torch and opencv-python are also installed for smooth training and prediction.

📊 Output
Trained model weights: runs/train/drone_yolo_model/weights/best.pt

Prediction results saved in the current directory (with bounding boxes)

📌 Notes
Label format follows YOLO: [class_id x_center y_center width height], all normalized.

Ensure correct image-label pairing (same filenames, different extensions).![ChatGPT Image May 11, 2025, 06_36_32 AM](https://github.com/user-attachments/assets/75a7a068-c7a4-4d21-b397-61616553d27e)
![Figure_1](https://github.com/user-attachments/assets/8284a321-b1ce-4db2-aca0-f1297aee18d5)


You can visualize predictions using model.predict(...) or explore results using the YOLO UI.
