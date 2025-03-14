import numpy as np
import torch
from torch import nn
from torchvision.transforms import (
    Compose,
    ConvertImageDtype,
    Lambda,
    Normalize,
    ToTensor,
)

from .base import BaseModel
from .networks.depth_pro import load_model

class DepthProModel(BaseModel):
    def __init__(self, model_path, model_type="dinov2l16_384", device="cpu",  **kwargs):
        self.model_type = model_type

        self.transform = Compose(
            [
                ToTensor(),
                Lambda(lambda x: x.to(self.device)),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ConvertImageDtype(torch.float16),
            ]
        )

        super().__init__(model_path, device)
        
    def _load_model(self):
        model = load_model(self.model_type, self.model_path, self.device)
        return model
    

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        focal_length: float | None,
        interpolation_mode="bilinear",
    ) -> np.ndarray:
        """Estimate depth and fov for a given image or a batch of images

        Note: if the focal length is given, the estimated value is ignored and the provided
        focal length is use to generate the metric depth values.

        Args:
            image (numpy.array): Input image(s) either with shape (HxWxC) or (BxHxWxC)
            focal_length (float): Optional focal length in pixels corresponding to `image`.
            interpolation_mode (str): Interpolation function for downsampling/upsampling. 
        Returns:
            depth (numpy.array): depth [m].
        """

        try:
            x = self.transform(image).to(self.device)
            f_px = torch.Tensor([focal_length]).to(self.device) if focal_length else None

            if len(x.shape) == 3:
                x = x.unsqueeze(0)

            _, _, H, W = x.shape
            resize = H != self.model.img_size or W != self.model.img_size

            if resize:
                x = nn.functional.interpolate(
                    x,
                    size=(self.model.img_size, self.model.img_size),
                    mode=interpolation_mode,
                    align_corners=False,
                )

            canonical_inverse_depth, fov_deg = self.model.forward(x)
            if f_px is None:
                f_px = 0.5 * W / torch.tan(0.5 * torch.deg2rad(fov_deg.to(torch.float)))
            
            inverse_depth = canonical_inverse_depth * (W / f_px)
            f_px = f_px.squeeze()

            if resize:
                inverse_depth = nn.functional.interpolate(
                    inverse_depth, size=(H, W), mode=interpolation_mode, align_corners=False
                )

            depth = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4)
            depth = depth.squeeze()
            return depth.detach().cpu().numpy()
        except Exception as e:
            print(f'Error when processing depth estimation: {e}, return empty depth map.')
            return np.zeros(image.shape)