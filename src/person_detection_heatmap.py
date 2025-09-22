import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 nano (person detection)
model = YOLO("yolov8n.pt")

# Video path
video_path = "data/videos/test_video_720p_1.mp4"
cap = cv2.VideoCapture(video_path)

# Video info
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Grid setup (10x10)
GRID_X = 10
GRID_Y = 10
cell_w = width // GRID_X
cell_h = height // GRID_Y

# Running total
total_persons_detected = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO
    results = model(frame, verbose=False)
    detections = results[0].boxes

    # Person count per frame
    person_count = 0

    # Initialize grid density map
    density_map = np.zeros((GRID_Y, GRID_X), dtype=np.int32)

    for box in detections:
        cls_id = int(box.cls[0])
        if cls_id == 0:  # person
            person_count += 1

            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # Draw person box (green)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Person center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Grid cell index
            gx = min(cx // cell_w, GRID_X - 1)
            gy = min(cy // cell_h, GRID_Y - 1)

            density_map[gy, gx] += 1

    # Update running total
    total_persons_detected += person_count

    # Create heatmap image (scaled to frame size)
    heatmap = cv2.resize(density_map, (width, height), interpolation=cv2.INTER_NEAREST)
    heatmap = np.uint8(255 * heatmap / heatmap.max()) if heatmap.max() > 0 else heatmap
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap (semi-transparent)
    overlay = cv2.addWeighted(heatmap_color, 0.4, frame, 0.6, 0)

    # Add counters on overlay
    cv2.putText(overlay, f"Persons in frame: {person_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(overlay, f"Running total: {total_persons_detected}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show video
    cv2.imshow("VigilAI - Person Heatmap", overlay)

    # Controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):  # Pause
        cv2.waitKey(-1)
    elif key == ord("s"):  # Save snapshot
        cv2.imwrite("snapshot_heatmap.jpg", overlay)
        print("Snapshot with heatmap saved!")

cap.release()
cv2.destroyAllWindows()
