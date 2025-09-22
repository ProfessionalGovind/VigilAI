"""
VigilAI - Smart Video Surveillance System
------------------------------------------
This script detects people in a video stream using YOLOv8 and overlays:
- Green bounding boxes for detected persons
- Real-time person count per frame
- Running total of persons detected
- 10x10 grid-based crowd density heatmap (semi-transparent overlay)
- FPS (frames per second) display for performance monitoring

Controls:
    Q - Quit program
    SPACE - Pause video
    S - Save snapshot of current frame

Author: VigilAI Project
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO


# ----------------------------
# Utility: FPS calculation
# ----------------------------
class FPSCounter:
    def __init__(self):
        self.prev_time = time.time()
        self.fps = 0.0

    def update(self):
        """Update FPS value based on frame processing time"""
        curr_time = time.time()
        elapsed = curr_time - self.prev_time
        self.prev_time = curr_time
        if elapsed > 0:
            self.fps = 1.0 / elapsed
        return self.fps


# ----------------------------
# Detection + Heatmap Overlay
# ----------------------------
def process_frame(frame, model, fps_counter, grid_x=10, grid_y=10):
    """
    Process a single video frame:
    - Detect persons
    - Draw bounding boxes
    - Update heatmap
    - Overlay stats
    """
    height, width = frame.shape[:2]

    # Run YOLOv8 detection
    results = model(frame, verbose=False)
    detections = results[0].boxes

    # Person count and density map
    person_count = 0
    density_map = np.zeros((grid_y, grid_x), dtype=np.int32)

    # Grid cell sizes
    cell_w = width // grid_x
    cell_h = height // grid_y

    for box in detections:
        cls_id = int(box.cls[0])
        if cls_id == 0:  # Person class
            person_count += 1

            # Bounding box coords
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # Draw bounding box (green)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Person center point → grid cell index
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            gx, gy = min(cx // cell_w, grid_x - 1), min(cy // cell_h, grid_y - 1)
            density_map[gy, gx] += 1

    # Create heatmap (scaled to frame size)
    if density_map.max() > 0:
        heatmap = cv2.resize(density_map, (width, height), interpolation=cv2.INTER_NEAREST)
        heatmap = np.uint8(255 * heatmap / heatmap.max())
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    else:
        heatmap_color = np.zeros_like(frame)

    # Overlay heatmap on video frame
    overlay = cv2.addWeighted(heatmap_color, 0.4, frame, 0.6, 0)

    # FPS
    fps = fps_counter.update()

    # Add text overlays
    cv2.putText(overlay, f"Persons in frame: {person_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(overlay, f"FPS: {fps:.2f}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return overlay, person_count


# ----------------------------
# Main function
# ----------------------------
def main():
    # Load YOLOv8 nano (person detection model)
    model = YOLO("yolov8n.pt")

    # Video input
    video_path = "data/videos/test_video_720p_1.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    fps_counter = FPSCounter()
    total_persons_detected = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        processed_frame, person_count = process_frame(frame, model, fps_counter)

        # Update running total
        total_persons_detected += person_count

        # Add running total text
        cv2.putText(processed_frame, f"Running total: {total_persons_detected}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display
        cv2.imshow("VigilAI - Person Detection & Heatmap", processed_frame)

        # Key controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):  # Pause
            cv2.waitKey(-1)
        elif key == ord("s"):  # Save snapshot
            cv2.imwrite("snapshot_heatmap.jpg", processed_frame)
            print("Snapshot saved as snapshot_heatmap.jpg")

    cap.release()
    cv2.destroyAllWindows()


# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    main()
