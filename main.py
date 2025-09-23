import argparse
import cv2
import os

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
        # Using my YOLOv8 class
        detector = YOLOv8Detector(model_path="models/yolov8n.pt")
        print("Using YOLOv8 model for detection.")
    elif args.model == "frcnn":
        # Using my Faster R-CNN class
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

    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video}")
        return

    # This is the main loop where all the magic happens.
    print("Starting video processing. Press 'q' to exit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # This is where I call the 'detect' method on my chosen detector.
        # It's the same for both models because of the blueprint we created!
        detections = detector.detect(frame)

        # Now I'll draw the bounding boxes and labels on the frame.
        for detection in detections:
            box = detection['box']
            label = detection['label']
            score = detection['score']
            
            x1, y1, x2, y2 = [int(coord) for coord in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the frame to the screen.
        cv2.imshow("VigilAI - Video Stream", frame)

        # Let the user exit by pressing 'q'.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Once the loop is done, I'll clean everything up.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()