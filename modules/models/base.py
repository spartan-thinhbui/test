from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    def __init__(self, model_path: str, device: str, **kwargs):
        self.device = device
        self.model_path = model_path
        self.model = self._load_model()
    
    @abstractmethod
    def _load_model(self):
        """Load the model from the specified path"""
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray, **kwargs):
        """
        Run inference on an image
        Args:
            image: numpy array of shape (height, width, channels)
        Returns:
            mask: numpy array of shape (height, width)
        """
        pass