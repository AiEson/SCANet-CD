from typing import Optional, Union, List
import torch
from copy import deepcopy

from ...encoders import get_encoder
from ...base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from .decoder import UnetDecoder


class Unet(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        # Create two encoders with shared weights
        self.encoder1 = get_encoder(
            encoder_name,
            in_channels=3,  # Each encoder takes 3-channel input
            depth=encoder_depth,
            weights=encoder_weights,
        )
        self.encoder2 = deepcopy(self.encoder1)
        
        # Adjust decoder input channels since features will be concatenated
        adjusted_encoder_channels = [ch * 2 for ch in self.encoder1.out_channels]
        
        self.decoder = UnetDecoder(
            encoder_channels=[ch * 2 for ch in self.encoder1.out_channels],
            decoder_channels=decoder_channels,
            n_blocks=self.encoder1._depth,
            use_batchnorm=decoder_use_batchnorm,
            center=encoder_name.startswith("vgg"),
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=adjusted_encoder_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    