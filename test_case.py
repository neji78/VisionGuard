import requests
import cv2
import threading
import queue

def send_frame(frame_queue, result_queue, stop_event):
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.05)
        except queue.Empty:
            continue
        _, img_encoded = cv2.imencode('.jpg', frame)
        files = {'image': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
        try:
            response = requests.post('http://localhost:5000/detect', files=files, timeout=10)
            detections = response.json()
        except Exception as e:
            detections = []
        result_queue.put(detections)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue()
    stop_event = threading.Event()
    detections = []

    thread = threading.Thread(target=send_frame, args=(frame_queue, result_queue, stop_event))
    thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        # Only keep the latest frame in the queue
        if frame_queue.empty():
            frame_queue.put(frame.copy())

        # Get latest detections if available
        try:
            while True:
                detections = result_queue.get_nowait()
        except queue.Empty:
            pass

        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            label = det['label']
            confidence = det['confidence']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Real-Time Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_event.set()
    thread.join()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()