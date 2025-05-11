import os
import shutil
import random

# Paths
img_dir = "Anti-drone.v10i.yolov8/train/images"
lbl_dir = "Anti-drone.v10i.yolov8/train/labels"
img_dir = "Anti-drone.v10i.yolov8/test/images"
lbl_dir = "Anti-drone.v10i.yolov8/test/labels"
img_dir = "Anti-drone.v10i.yolov8/val/images"
lbl_dir = "Anti-drone.v10i.yolov8/val/labels"

split_root = "drone_yolo_dataset"
splits = ['train', 'val', 'test']
split_ratios = [0.7, 0.2, 0.1]  # 70% train, 20% val, 10% test

# Create folders
for split in splits:
    os.makedirs(os.path.join(split_root, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(split_root, 'labels', split), exist_ok=True)

# Get all image files
files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
random.shuffle(files)

# Split data
train_end = int(len(files) * split_ratios[0])
val_end = train_end + int(len(files) * split_ratios[1])

for i, file in enumerate(files):
    if i < train_end:
        split = 'train'
    elif i < val_end:
        split = 'val'
    else:
        split = 'test'

    img_src = os.path.join(img_dir, file)
    lbl_src = os.path.join(lbl_dir, file.replace('.jpg', '.txt').replace('.png', '.txt'))

    img_dst = os.path.join(split_root, 'images', split, file)
    lbl_dst = os.path.join(split_root, 'labels', split, os.path.basename(lbl_src))

    shutil.copy(img_src, img_dst)
    shutil.copy(lbl_src, lbl_dst if os.path.exists(lbl_src) else open(lbl_dst, 'w').close())
