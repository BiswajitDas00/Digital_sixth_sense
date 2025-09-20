import cv2

# Example: three phones streaming
urls = [
    "http://172.16.3.202:8080/video",  # phone 1
    "http://172.16.3.205:8080/video",  # phone 2
    "http://172.16.1.38:8080/video"   # phone 3
]

# Open video streams
caps = [cv2.VideoCapture(u) for u in urls]

while True:
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    # If all 3 frames are captured
    if len(frames) == 3:
        # Resize all frames to same height for proper concatenation
        h = min(f.shape[0] for f in frames)
        frames = [cv2.resize(f, (int(f.shape[1] * h / f.shape[0]), h)) for f in frames]

        # Combine horizontally (side-by-side view)
        stitched = cv2.hconcat(frames)
        cv2.imshow("3 Camera View", stitched)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release all cameras
for cap in caps:
    cap.release()
cv2.destroyAllWindows()


