import torch
import torchvision
from src.detector import Detector
import numpy as np

class FasterRCNNDetector(Detector):
    """
    My Faster R-CNN detector. I'm building it to follow the same blueprint as my
    YOLOv8 one, so I can use them interchangeably in my main script.
    """
    def __init__(self, model_path: str):
        # Loading a pre-trained model here and setting it to evaluation mode.
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')
        self.model.eval()

    def detect(self, image: np.ndarray):
        # The main method. It takes a NumPy image and runs detection on it.
        # First, convert the image to a tensor that PyTorch can use.
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Now I'll run the detection without calculating gradients for speed.
        with torch.no_grad():
            predictions = self.model([image_tensor])

        # Get the first prediction and pull out the good stuff.
        prediction = predictions[0]
        detections = []
        for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
            # I'm only keeping detections with a decent confidence score.
            if score > 0.5:
                detections.append({
                    "box": box.tolist(),
                    "label": str(label.item()), 
                    "score": score.item()
                })

        return detections