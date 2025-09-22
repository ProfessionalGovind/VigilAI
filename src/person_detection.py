import cv2
from ultralytics import YOLO

# Load YOLOv8 model (assumes it's already in your env)
model = YOLO("yolov8n.pt")  # or yolov8s.pt, etc.

# Input video
video_path = "data/videos/test_video_720p_1.mp4"
cap = cv2.VideoCapture(video_path)

# Video info
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Running person count
total_persons_detected = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, verbose=False)

    # Extract detections (first batch only)
    detections = results[0].boxes

    # Count persons in current frame
    person_count = 0

    for box in detections:
        cls_id = int(box.cls[0])
        if cls_id == 0:  # person class
            person_count += 1

            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box (green)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Optionally show confidence
            conf = float(box.conf[0])
            cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Update running total
    total_persons_detected += person_count

    # Overlay counts
    cv2.putText(frame, f"Persons in frame: {person_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Running total: {total_persons_detected}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("VigilAI - Person Detection", frame)

    # Controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):  # Pause
        cv2.waitKey(-1)
    elif key == ord("s"):  # Save snapshot
        cv2.imwrite("snapshot.jpg", frame)
        print("Snapshot saved!")

cap.release()
cv2.destroyAllWindows()
