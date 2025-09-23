from ultralytics import YOLO
import numpy as np
import torch # Need this to check if a GPU is available
from src.detector import Detector # Our own Detector blueprint

class YOLOv8Detector(Detector):
    """
    My custom YOLOv8 detector. I'm building this class to match
    the Detector blueprint, so I can easily swap it with other models.
    """
    def __init__(self, model_path: str):
        # I'm loading the YOLO model here. I'll make sure to use the GPU if it's available.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path)
        print(f"YOLOv8 model loaded and using device: {device}")

    def detect(self, image: np.ndarray):
        # This is the main method that does the detection.
        # It takes a frame and runs the YOLO model on it.
        results = self.model(image, verbose=False) # verbose=False just to keep the terminal clean
        
        # Get the first result from the list since we're processing one image.
        result = results[0]
        
        # Now I'm going to get the good stuff: bounding boxes, labels, and scores.
        detections = []
        for box in result.boxes:
            # The class ID for 'person' is 0, so I'll filter for that.
            cls_id = int(box.cls[0])
            if cls_id == 0:
                # Getting the box coordinates.
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # Grabbing the confidence score.
                score = float(box.conf[0])

                # I'm storing this info in a dictionary.
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "label": 'person', # We already know it's a person
                    "score": score
                })

        return detections