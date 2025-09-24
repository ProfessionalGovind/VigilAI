from ultralytics import YOLO
import numpy as np
import torch
from src.detector import Detector

class YOLOv8Detector(Detector):
    """
    My custom YOLOv8 detector class. I'm building it to match the
    Detector blueprint so I can easily swap it with other models.
    """
    def __init__(self, model_path: str):
        # The first thing is to load the YOLO model.
        # I'll check if a GPU is available to make sure it's fast.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path).to(device)
        print(f"YOLOv8 model loaded and using device: {device}")

    def detect(self, image: np.ndarray):
        # This is the main method that actually does the detection.
        # It takes a frame (NumPy array) and runs the model.
        results = self.model(image, verbose=False) # verbose=False just to keep the terminal clean
        
        # Get the first result from the list (since we only passed one image).
        result = results[0]
        
        # Now, I'll pull out all the good stuff like bounding boxes, labels, and confidence scores.
        detections = []
        for box in result.boxes:
            # Getting the class ID for 'person' (which is 0) to filter for it.
            cls_id = int(box.cls[0])
            if cls_id == 0:
                # Getting the box coordinates.
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # Grabbing the confidence score.
                score = float(box.conf[0])
            
                # Storing it all in a neat dictionary.
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "label": 'person',
                    "score": score
                })

        return detections