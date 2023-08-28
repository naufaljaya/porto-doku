import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# Initialize video capture
video_path = "../Videos/cars.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize YOLO model
model_path = "../Yolo-Weights/yolov8l.pt"
model = YOLO(model_path)

# List of object class names
class_names = ["car", "truck", "bus", "motorbike"]

# Load mask image
mask = cv2.imread("mask.png")

# Initialize object tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Define threshold limits
threshold_limits = [400, 297, 673, 297]

# Initialize tracked object IDs
tracked_ids = []

while True:
    # Read a frame from the video
    success, frame = cap.read()

    # Apply mask to the frame
    masked_frame = cv2.bitwise_and(frame, mask)

    # Overlay graphics on the frame
    graphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    frame = cvzone.overlayPNG(frame, graphics, (0, 0))

    # Detect objects using YOLO
    detections = np.empty((0, 5))
    results = model(masked_frame, stream=True)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            current_class = class_names[cls]

            if current_class in class_names and conf > 0.3:
                current_array = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_array))

    # Update object tracking
    tracked_objects = tracker.update(detections)

    # Draw threshold line
    cv2.line(frame, (threshold_limits[0], threshold_limits[1]),
             (threshold_limits[2], threshold_limits[3]), (0, 0, 255), 5)

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        w, h = x2 - x1, y2 - y1

        # Draw rectangle and text for each tracked object
        cvzone.cornerRect(frame, (x1, y1, w, h), l=9,
                          rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(frame, f'ID: {int(obj_id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Check if object crossed the threshold
        if threshold_limits[0] < cx < threshold_limits[2] and threshold_limits[1] - 15 < cy < threshold_limits[1] + 15:
            if obj_id not in tracked_ids:
                tracked_ids.append(obj_id)
                cv2.line(frame, (threshold_limits[0], threshold_limits[1]), (
                    threshold_limits[2], threshold_limits[3]), (0, 255, 0), 5)

    # Display total count
    cv2.putText(frame, str(len(tracked_ids)), (255, 100),
                cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    # Display the annotated frame
    cv2.imshow("Object Detection and Tracking", frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
