from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2

app = Flask(__name__)
model = YOLO('yolov8n.pt')  # or yolov5s.pt, etc.

@app.route('/detect', methods=['POST'])
def detect():
    image_file = request.files['image']
    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    results = model(image)

    detections = []
    for box in results[0].boxes:
        detections.append({
            'label': results[0].names[int(box.cls)],
            'confidence': float(box.conf),
            'bbox': box.xyxy[0].tolist()
        })

    return jsonify(detections)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
