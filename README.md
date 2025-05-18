# VisionGuard

VisionGuard is a lightweight object detection service built with FastAPI. It enables real-time object recognition using a YOLOv8 model and serves predictions via a REST API. The service is designed for easy integration with other applications or platforms requiring computer vision capabilities.

## 🚀 Features

- 📦 RESTful API built with FastAPI
- 🔍 Real-time object detection via YOLOv8
- 📷 Image upload and detection endpoint (`/detect/`)
- 💾 Save uploaded files and inference results locally
- 🔧 Easily configurable and extensible

## 🖼️ Example

Send a POST request with an image file:

bash
```
curl -X POST http://localhost:8000/detect/ -F "file=@example.jpg"
```
Response:
```
{
  "detections": [
    {
      "class": "person",
      "confidence": 0.92,
      "box": [34, 50, 100, 200]
    },
    ...
  ]
}
```
🛠️ Installation
  1.Clone the repository
  ```
git clone https://github.com/neji78/object_detection_service.git
cd object_detection_service
  ```
  2.Create a virtual environment
  ```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
  3.Install dependencies
  ```
pip install -r requirements.txt
```
  4.Run the API
  ```
uvicorn main:app --reload
```

📁 Project Structure
```
.
├── main.py               # FastAPI app
├── model.py              # YOLOv8 model loading and inference
├── utils.py              # Helper functions
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation

```
