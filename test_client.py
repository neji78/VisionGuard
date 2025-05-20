import test_case
import cv2

# Load image with OpenCV
image_path = 'image.jpg'  # replace with your test image
image = cv2.imread(image_path)

# Encode image as JPEG
_, img_encoded = cv2.imencode('.jpg', image)
files = {'image': ('test.jpg', img_encoded.tobytes(), 'image/jpeg')}

# Send POST request to Flask server
response = test_case.post('http://localhost:5000/detect', files=files)

# Parse response
detections = response.json()
print("Detections:", detections)

# Draw bounding boxes
for det in detections:
    x1, y1, x2, y2 = map(int, det['bbox'])
    label = det['label']
    confidence = det['confidence']
    
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show result
cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
