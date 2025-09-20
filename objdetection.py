import cv2
from ultralytics import YOLO

# COCO dataset classes
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli",
    "carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

# Video source: webcam or phone IP camera
VIDEO_SOURCE = 0  # Use 0 for local webcam, or "http://<IP>:8080/video" for phone

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Open video capture
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("❌ Cannot open video source")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = model(frame, verbose=False)

    for r in results:
        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            cls_id = int(cls)
            if cls_id == 0:  # Skip person
                continue
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{COCO_CLASSES[cls_id]}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show live detection
    cv2.imshow("Object Detection with Names", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
