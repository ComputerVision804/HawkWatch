# Train Model yolov8 from scratch
from ultralytics import YOLO

#  Load apretrained YOLOv8 model
model = YOLO('yolov8n.pt')  # Load a pretrained YOLOv8n model

#  Train the model on a custom dataset
model.train(
    data='IP Proj 2-Quadcopter.v19i.yolov8/data.yaml', 
            epochs=1,
            imgsz=640,
            batch=4, # Batch can adjust like orignal is 64 and you can adjust 32, 16, 8.
            cache=True,
            name='drone-detection',
            project='runs/train',
            device='cpu' # Use '0' for GPU or 'cpu' for CPU training
)  
