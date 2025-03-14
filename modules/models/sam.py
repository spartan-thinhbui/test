from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import torch

from .base import BaseModel

class SAMModel(BaseModel):
    def __init__(self, model_path, device="cpu", model_type="vit_h",  **kwargs):
        self.model_type = model_type
        super().__init__(model_path, device)
        
    def _load_model(self):
        sam = sam_model_registry[self.model_type](checkpoint=self.model_path)
        sam.to(device=self.device)
        #self.mask_generator = SamAutomaticMaskGenerator(sam)
        self.predictor = SamPredictor(sam)
        return sam
    
    
    def predict(self, image: np.ndarray, bboxes: np.array) -> np.array:
        """
        Args:
            image: numpy array of shape (512, 512, channels) in RGB format
            bboxes: numpy array bounding boxes inside the
        Returns:
            masks: numpy array of shape (N, 1, 512, 512) with values 0 or 1 including masks for the input bounding boxes
        """
        try:
            self.predictor.set_image(np.array(image))

            bboxes_tensor = torch.Tensor(np.array(bboxes)).to(self.device)

            transformed_boxes = self.predictor.transform.apply_boxes_torch(bboxes_tensor, image.shape[:2])
            masks, _, _ = self.predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                multimask_output = False,
            )
            return masks.cpu().numpy()
        except Exception as e:
            print(f"Error processing image: {str(e)}, return empty masks")
            return np.zeros((len(bboxes), 4))