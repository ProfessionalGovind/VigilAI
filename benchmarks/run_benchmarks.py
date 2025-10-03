import time
import cv2
import numpy as np
import os
import sys

# Making sure the script can find modules in the src/ directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.yolov8_detector import YOLOv8Detector
from src.faster_rcnn_detector import FasterRCNNDetector

# CONFIGURATION
NUM_FRAMES_TO_TEST = 100
DUMMY_IMAGE_SIZE = (640, 480, 3) # Standard test resolution (Height, Width, Channels)

def run_benchmark(detector, detector_name):
    """
    Runs the detection loop for a set number of frames and measures the time.
    """
    print(f"\n🚀 Starting benchmark for {detector_name}...")
    
    # Using a dummy image to simulate video frames
    dummy_frame = np.random.randint(0, 256, size=DUMMY_IMAGE_SIZE, dtype=np.uint8)
    
    start_time = time.time()
    
    for i in range(NUM_FRAMES_TO_TEST):
        # Simply calling the existing detect method repeatedly
        _ = detector.detect(dummy_frame)
        
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculating Frames Per Second
    fps = NUM_FRAMES_TO_TEST / total_time
    
    print(f"✅ {detector_name} completed {NUM_FRAMES_TO_TEST} frames.")
    print(f"   Total Time: {total_time:.4f} seconds")
    print(f"   Average FPS: {fps:.2f} FPS")
    
    return total_time, fps

if __name__ == "__main__":
    # MODEL PATHS
    YOLO_PATH = "models/yolov8n.pt"
    FRCNN_PATH = "models/FasterRCNN/faster_rcnn_model.pth"
    
    # Initialize the detectors (this also times the loading process, which is useful)
    yolo_detector = YOLOv8Detector(model_path=YOLO_PATH)
    frcnn_detector = FasterRCNNDetector(model_path=FRCNN_PATH)
    
    results = {}
    
    # 1. Benchmark YOLOv8
    _, yolo_fps = run_benchmark(yolo_detector, "YOLOv8")
    results['YOLOv8'] = yolo_fps
    
    # 2. Benchmark Faster R-CNN
    _, frcnn_fps = run_benchmark(frcnn_detector, "Faster R-CNN")
    results['Faster R-CNN'] = frcnn_fps
    
    print("\n--- FINAL BENCHMARK RESULTS ---")
    print(f"YOLOv8 FPS:       {results['YOLOv8']:.2f}")
    print(f"Faster R-CNN FPS: {results['Faster R-CNN']:.2f}")

    print("\n💡 NOTE: You still need to manually add accuracy (mAP/Precision/Recall) results to your report for a complete comparison.")