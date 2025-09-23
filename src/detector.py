from abc import ABC, abstractmethod
import numpy as np

class Detector(ABC):
    """
    An abstract base class for my detectors.
    This is basically a blueprint for how every model should behave,
    so I can swap them out easily later on.
    """
    @abstractmethod
    def __init__(self, model_path: str):
        """
        Initializes the detector. Gotta pass the path to the model file here.
        """
        pass

    @abstractmethod
    def detect(self, image: np.ndarray):
        """
        The core method for running detection on an image.
        
        Args:
            image (np.ndarray): The input image as a NumPy array.
            
        Returns:
            list: A list of dictionaries with results like
                  bounding boxes, labels, and confidence scores.
        """
        pass

class DetectionResult:
    """
    A simple class to hold the detection results. Keeps things clean!
    """
    def __init__(self, box: list, label: str, score: float):
        self.box = box
        self.label = label
        self.score = score