# This code is for live interface and you can detect drones with live camera
from ultralytics import YOLO
import cv2
import csv
from datetime import datetime

# Load the YOLOv8 model
model = YOLO('runs/train/drone-detection/weights/best.pt')  # Load the best model from training

# Initialize the video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam or replace with video file path

# set better backend properties for faster processing (helps Intel Iris Xe graphics)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Reduce buffer size

# Set confidence threshold (example 0.5 means 50%)
confidence_threshold = 0.5

# CSV file to store detection results
csv_filename = 'detection_results.csv'

# Ensure the CSV file is created with headers if it doesn't exist
def create_csv_file(filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Label', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])  # Headers

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break

    # Perform inference on the frame
    results = model.predict(frame, conf=confidence_threshold, verbose=False)

    # Save the results to the CSV file
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)  # Class ID
            conf = float(box.conf)  # Confidence score
            label = model.names[cls_id]  # Class label

            if label == 'drone' and conf >= confidence_threshold:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                date = [current_time, label, conf]

                # Append to csv file
                with open(csv_filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(date)

                # Print the detection results
                print(f"saved detection: {date}")

    # Plot the results
    annotated_frame = results[0].plot()

    # Show the output 
    cv2.imshow('Drone Detection', annotated_frame)

    # Stop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Detection Stopped by User")
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
