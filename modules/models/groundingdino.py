# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import predict
import torch
from PIL import Image
import numpy as np

from .base import BaseModel

class GroundingDINOModel(BaseModel):
    def __init__(self,
        model_path: str,
        model_config_path: str,
        device: str='cpu',
        **kwargs
    ):
        self.model_config_path = model_config_path
        super().__init__(model_path, device)
        
    def _load_model(self):
        args = SLConfig.fromfile(self.model_config_path)
        args.device = self.device
        model = build_model(args).to(self.device)

        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        return model

    @staticmethod
    def _load_image(image: np.array) -> torch.Tensor:
        """
        Load and transform image as torch Tensor
        Source: https://github.com/IDEA-Research/GroundingDINO/blob/856dde20aee659246248e20734ef9ba5214f5e44/groundingdino/util/inference.py
        """
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_transformed, _ = transform(Image.fromarray(image), None)
        return image_transformed
    

    @staticmethod
    def transform_bboxes(boxes, image_shape) -> np.array:
        """
        Transform GroundingDINO bounding box format to x, y, x, y
        """
        H, W = image_shape[:2]
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        return boxes_xyxy.numpy()
    
    @staticmethod
    def _contruct_object_detection_prompt(target_objects: list[str]) -> str:
        """
            Create object detection prompt from list of objects
            Args:
                target_objects (list[str]): list of objects
            Returns:
                prompt: constructed prompt
        """

        object_detection_prompt = ", ".join(target_objects)
        return object_detection_prompt

    
    def predict(self, image: np.ndarray, objects: list[str], bbox_threshold: float=0.25, text_threshold: float=0.25) -> dict:
        """
        Args:
            image: numpy array of shape (height, width, channels) in RGB format
            objects: list of objects to be detected
            bbox_threshold: detected bounding box similarity threshold
            text_threshold: text similarity threshold
        Returns:
            dict: a dictionary where key is the label and value is a list of detected bounding boxes
        """
        target_bounding_boxes = {object : [] for object in objects}

        try:
            text_prompt = self._contruct_object_detection_prompt(objects)


            image_tensor = self._load_image(image)
            
            boxes, logits, labels = predict(
                model=self.model,
                image=image_tensor,
                caption=text_prompt,
                box_threshold=bbox_threshold,
                text_threshold=text_threshold,
                device=self.device
            )

            boxes = self.transform_bboxes(boxes, image.shape)

            for label, bounding_box in zip(labels, boxes):
                if label in target_bounding_boxes:
                    target_bounding_boxes[label].append(bounding_box)
                else:
                    target_bounding_boxes[label] = [bounding_box]

            return target_bounding_boxes
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return target_bounding_boxes

