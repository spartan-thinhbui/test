import numpy as np
import cv2

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .base import BaseModel

class SAM2Model(BaseModel):
    def __init__(self, model_path, device="cpu",  **kwargs):
        super().__init__(model_path, device)
        
    def _load_model(self):
        if "sam2.1_hiera_large" in self.model_path:
            config = "configs/sam2.1/sam2.1_hiera_l.yaml"
        elif "sam2.1_hiera_base_plus" in self.model_path:
            config = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        else:
            raise ValueError("Model not exists or yet to be supported")

        sam = build_sam2(config, self.model_path, device=self.device)
        self.predictor = SAM2ImagePredictor(sam)
        return sam
    

    @staticmethod
    def erode_masks(masks, kernel_size=5, iterations=3):
        """
        Shrinks the object in a binary mask by applying morphological erosion.

        Parameters:
        - binary_mask: Input binary image (numpy array)
        - kernel_size: Size of the erosion kernel (higher = more shrinking)
        - iterations: Number of erosion steps (higher = more shrinking)

        Returns:
        - Eroded binary mask
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8) # Create erosion kernel

        eroded_masks = []
        for mask in masks:
            eroded_masks.append(cv2.erode(mask, kernel, iterations=iterations))

        return np.array(eroded_masks)

    
    def predict(self, image: np.ndarray, bboxes: np.array, erode=False) -> np.array:
        """
        Args:
            image: numpy array of shape (H, W, channels) in RGB format
            bboxes: list of bounding boxes inside the image with shape (Nx4)
        Returns:
            masks: numpy array of shape (N, 1, H, W) with values 0 or 1 including masks for the input bounding boxes
        """
        try:
            self.predictor.set_image(np.array(image))
            masks, _, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bboxes,
                multimask_output=False,
            )

            # add batch if the model only returns 1 mask
            if len(masks.shape) == 3:
                masks = np.array([masks])
            
            if erode:
                masks = self.erode_masks([mask[0] for mask in masks])
                masks = np.expand_dims(masks, axis=1)
            
            return masks
        except Exception as e:
            print(f"Error processing image: {str(e)}, return empty masks")
            return np.zeros((len(bboxes), 4))