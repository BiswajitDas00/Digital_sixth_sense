import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------
# Load YOLOv8 Model (COCO pretrained)
# ----------------------------
model = YOLO("yolov8n.pt")  # nano model for speed

# ----------------------------
# Video Capture
# ----------------------------
cap = cv2.VideoCapture(0)  # webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, stream=True)

    person_count = 0

    for r in results:
        boxes = r.boxes.cpu().numpy()
        for box in boxes:
            cls = int(box.cls[0])  # class id
            conf = float(box.conf[0])  # confidence
            if cls == 0 and conf > 0.5:  # "person" class
                person_count += 1
                x1, y1, x2, y2 = box.xyxy[0].astype(int)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw center point
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                # Label
                cv2.putText(frame, f"Person {person_count}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Show total count
    cv2.putText(frame, f"Total Persons: {person_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

    cv2.imshow("Multiple Person Detection & Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
