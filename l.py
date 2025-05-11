from ultralytics import YOLO

# Load a base model (nano/small/medium based on your system)
model = YOLO("yolov8n.pt")  # You can use yolov8s.pt or yolov8m.pt for better accuracy

# Train the model
model.train(data="path/to/data.yaml", epochs=50, imgsz=640, batch=16)
