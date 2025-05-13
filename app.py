from flask import Flask, render_template, Response, send_file, jsonify, url_for
from ultralytics import YOLO
import cv2
import csv
from datetime import datetime
import os

app = Flask(__name__)
model = YOLO('runs/train/drone-detection10/weights/best.pt')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

confidence_threshold = 0.5
csv_filename = 'detection_results.csv'
detection_count = 0

def create_csv_file(filename):
    if not os.path.exists(filename):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Label', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])

create_csv_file(csv_filename)

def generate_frames():
    global detection_count
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=confidence_threshold, verbose=False)
        detections_in_frame = 0

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                label = model.names[cls_id]

                if label == 'drone' and conf >= confidence_threshold:
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, xyxy)
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    with open(csv_filename, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([timestamp, label, f"{conf:.2f}", x1, y1, x2, y2])

                    detections_in_frame += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        detection_count += detections_in_frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/download_csv')
def download_csv():
    return send_file(csv_filename, as_attachment=True)

@app.route('/api/detections')
def get_detections():
    return jsonify({'count': detection_count})

if __name__ == '__main__':
    app.run(debug=True)
