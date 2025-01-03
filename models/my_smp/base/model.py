import torch
import torch.nn as nn
from . import initialization as init


class SegmentationModel(torch.nn.Module):
    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

        # self.projector = nn.Sequential(nn.Conv2d(2048, 2048, kernel_size=1),
        #                                nn.ReLU(),
        #                                nn.Conv2d(2048, 64, kernel_size=1))

    def check_input_shape(self, x):
        h, w = x.shape[-2:]
        output_stride = self.encoder1.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (
                (h // output_stride + 1) * output_stride
                if h % output_stride != 0
                else h
            )
            new_w = (
                (w // output_stride + 1) * output_stride
                if w % output_stride != 0
                else w
            )
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, img_a, img_b):
        """Process two images through separate encoders and combine features"""
        self.check_input_shape(img_a)
        self.check_input_shape(img_b)
        
        # Get features from both encoders
        features1 = self.encoder1(img_a)
        features2 = self.encoder2(img_b)
        
        # Concatenate features at each level
        features = []
        for f1, f2 in zip(features1, features2):
            features.append(torch.cat([f1, f2], dim=1))
        
        # Pass through decoder
        decoder_output = self.decoder(*features)
        
        # Generate masks
        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels
            
        return masks


    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x
