import sys
import os
import argparse
import cv2

# This line dynamically adds the project's root directory to Python's path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.analytics import Heatmap, FlowAnalyzer
from src.yolov8_detector import YOLOv8Detector
from src.faster_rcnn_detector import FasterRCNNDetector

def main():
    # First, I'm setting up a way to get inputs from the command line.
    parser = argparse.ArgumentParser(description="Run object detection with a specified model.")
    parser.add_argument("--model", type=str, default="yolov8", help="Model to use: 'yolov8' or 'frcnn'")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file")
    args = parser.parse_args()

    # Now, I'm going to choose which detector to use based on the input.
    if args.model == "yolov8":
        detector = YOLOv8Detector(model_path="models/yolov8n.pt")
        print("Using YOLOv8 model for detection.")
    elif args.model == "frcnn":
        detector = FasterRCNNDetector(model_path="models/FasterRCNN/faster_rcnn_model.pth")
        print("Using Faster R-CNN model for detection.")
    else:
        print("Error: Invalid model name. Please choose 'yolov8' or 'frcnn'.")
        return

    # Now I'm setting up the video stream.
    if not os.path.exists(args.video):
        print(f"Error: Video file not found at {args.video}")
        return

    cap = cv2.VideoCapture(args.video)
    ret, initial_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame to get dimensions.")
        return

    frame_size = (initial_frame.shape[0], initial_frame.shape[1])
    heatmap_generator = Heatmap(frame_size)
    flow_analyzer = FlowAnalyzer(line_y_coord=int(frame_size[0] * 0.5), line_direction='up')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video}")
        return

    print("Starting video processing. Press 'q' to exit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # This is where I call the 'detect' method on my chosen detector.
        detections = detector.detect(frame)

        # Update the heatmap with the latest detections
        heatmap_generator.update(detections)

        # Now I'll update my flow analyzer with the new detections
        flow_analyzer.update(detections)
        person_count = flow_analyzer.person_count

        # Draw the heatmap first and get the new frame
        annotated_frame = heatmap_generator.draw(frame)

        # Now I'll draw the bounding boxes, the counting line, and the text.
        for detection in detections:
            box = detection['box']
            label = detection['label']
            score = detection['score']
            
            x1, y1, x2, y2 = [int(coord) for coord in box]
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"{label}: {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw the counting line on the frame
        cv2.line(annotated_frame, (0, flow_analyzer.line_y_coord), (frame_size[1], flow_analyzer.line_y_coord), (255, 0, 0), 2)

        # Display the current person count
        cv2.putText(annotated_frame, f"People Crossed: {person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Display the frame to the screen.
        cv2.imshow("VigilAI - Video Stream", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
