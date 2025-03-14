from __future__ import annotations

from typing import Optional, Tuple


import torch
from torch import nn

from .decoder import MultiresConvDecoder
from .encoder import DepthProEncoder
from .fov import FOVNetwork
from .vit_factory import create_vit
from .decoder import MultiresConvDecoder
from .encoder import DepthProEncoder
from .fov import FOVNetwork

class DepthPro(nn.Module):
    """DepthPro network."""

    def __init__(
        self,
        encoder: DepthProEncoder,
        decoder: MultiresConvDecoder,
        last_dims: tuple[int, int],
        use_fov_head: bool = True,
        fov_encoder: Optional[nn.Module] = None,
    ):
        """Initialize DepthPro.

        Args:
        ----
            encoder: The DepthProEncoder backbone.
            decoder: The MultiresConvDecoder decoder.
            last_dims: The dimension for the last convolution layers.
            use_fov_head: Whether to use the field-of-view head.
            fov_encoder: A separate encoder for the field of view.

        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
    
        dim_decoder = decoder.dim_decoder
        self.head = nn.Sequential(
            nn.Conv2d(
                dim_decoder, dim_decoder // 2, kernel_size=3, stride=1, padding=1
            ),
            nn.ConvTranspose2d(
                in_channels=dim_decoder // 2,
                out_channels=dim_decoder // 2,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
            ),
            nn.Conv2d(
                dim_decoder // 2,
                last_dims[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(True),
            nn.Conv2d(last_dims[0], last_dims[1], kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )

        # Set the final convolution layer's bias to be 0.
        self.head[4].bias.data.fill_(0)

        # Set the FOV estimation head.
        if use_fov_head:
            self.fov = FOVNetwork(num_features=dim_decoder, fov_encoder=fov_encoder)

    @property
    def img_size(self) -> int:
        """Return the internal image size of the network."""
        return self.encoder.img_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Decode by projection and fusion of multi-resolution encodings.

        Args:
        ----
            x (torch.Tensor): Input image.

        Returns:
        -------
            The canonical inverse depth map [m] and the optional estimated field of view [deg].

        """
        _, _, H, W = x.shape
        assert H == self.img_size and W == self.img_size

        encodings = self.encoder(x)
        features, features_0 = self.decoder(encodings)
        canonical_inverse_depth = self.head(features)

        fov_deg = None
        if hasattr(self, "fov"):
            fov_deg = self.fov.forward(x, features_0.detach())

        return canonical_inverse_depth, fov_deg


def load_model(model: str, model_path: str, device: torch.device = torch.device("cpu")) -> DepthPro:
    patch_encoder = create_vit(preset=model, use_pretrained=False)
    image_encoder = create_vit(preset=model, use_pretrained=False)
    fov_encoder = create_vit(preset=model, use_pretrained=False)

    encoder = DepthProEncoder(
        dims_encoder=[256, 512, 1024, 1024],
        patch_encoder=patch_encoder,
        image_encoder=image_encoder,
        hook_block_ids=[5, 11, 17, 23],
        decoder_features=256,
    )

    decoder = MultiresConvDecoder(
        dims_encoder=[256] + list(encoder.dims_encoder),
        dim_decoder=256,
    )

    model = DepthPro(
        encoder=encoder,
        decoder=decoder,
        last_dims=(32, 1),
        use_fov_head=True,
        fov_encoder=fov_encoder,
    ).to(device)

    model.half()

    state_dict = torch.load(model_path, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict=state_dict, strict=True
    )

    if len(unexpected_keys) != 0:
        print(
            f"Found unexpected keys when loading monodepth: {unexpected_keys}"
        )

    # fc_norm is only for the classification head,
    # which we would not use. We only use the encoding.
    missing_keys = [key for key in missing_keys if "fc_norm" not in key]
    if len(missing_keys) != 0:
        print(f"Keys are missing when loading monodepth: {missing_keys}")

    return model.eval()