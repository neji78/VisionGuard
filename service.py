from flask import Flask, request, jsonify
from PIL import Image
from ultralytics import YOLO
import io

app = Flask(__name__)
model = YOLO('yolov8n.pt')  # or yolov5s.pt, etc.

@app.route('/detect', methods=['POST'])
def detect():
    image_file = request.files['image']
    image = Image.open(image_file.stream)

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
    app.run(host='0.0.0.0', port=5000)
