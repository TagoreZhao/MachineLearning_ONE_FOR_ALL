# models/MobileNetV1.py
import torch
import torch.nn as nn
from .layers import DepthwiseSeparableConv
from .base_model import BaseModel

class MobileNetV1(BaseModel):
    """
    MobileNetV1 Architecture with resolution and width multipliers.

    Description:
    - Uses depthwise separable convolutions to reduce the number of parameters and computations.
    - Incorporates resolution and width multipliers to adjust the computational cost and model size.

    Parameters:
    - num_classes: Number of output classes (default is 10 for CIFAR-10).
    - width_multiplier: Scales the number of channels in each layer.
    - resolution_multiplier: Scales the input resolution of the model.
    """
    def __init__(self, num_classes=10, width_multiplier=1.0, resolution_multiplier=1.0, dropout_rate=0.5):
        super(MobileNetV1, self).__init__()
        self.width_multiplier = width_multiplier
        self.resolution_multiplier = resolution_multiplier
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Adjust the input channels according to the width multiplier
        def adjust_channels(channels):
            return max(1, int(channels * self.width_multiplier))

        # Initial standard convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, adjust_channels(32), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(adjust_channels(32)),
            nn.ReLU(inplace=True)
        )

        # Define the MobileNetV1 blocks using Depthwise Separable Convolutions
        self.layers = nn.Sequential(
            DepthwiseSeparableConv(adjust_channels(32), adjust_channels(64), stride=1),
            DepthwiseSeparableConv(adjust_channels(64), adjust_channels(128), stride=2),
            DepthwiseSeparableConv(adjust_channels(128), adjust_channels(128), stride=1),
            DepthwiseSeparableConv(adjust_channels(128), adjust_channels(256), stride=2),
            DepthwiseSeparableConv(adjust_channels(256), adjust_channels(256), stride=1),
            DepthwiseSeparableConv(adjust_channels(256), adjust_channels(512), stride=2),
            *[DepthwiseSeparableConv(adjust_channels(512), adjust_channels(512), stride=1) for _ in range(5)],
            DepthwiseSeparableConv(adjust_channels(512), adjust_channels(1024), stride=2),
            DepthwiseSeparableConv(adjust_channels(1024), adjust_channels(1024), stride=1)
        )

        # Dropout layer to reduce overfitting
        self.dropout = nn.Dropout(p=self.dropout_rate)

        # Fully connected layer for classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(adjust_channels(1024), num_classes)

    def forward(self, x):
        """
        Forward pass of the MobileNetV1 model.
        """
        # Scale the input resolution if resolution_multiplier is different from 1.0
        if self.resolution_multiplier != 1.0:
            height, width = x.size(2), x.size(3)
            new_height, new_width = int(height * self.resolution_multiplier), int(width * self.resolution_multiplier)
            x = nn.functional.interpolate(x, size=(new_height, new_width), mode='bilinear', align_corners=False)

        x = self.initial_conv(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)  # Apply dropout before the fully connected layer
        x = self.fc(x)
        return x

# Factory function to create MobileNetV1 model
def mobilenet_v1(num_classes=10, width_multiplier=1.0, resolution_multiplier=1.0, dropout_rate=0.5):
    """
    Constructs a MobileNetV1 model with specified multipliers.
    """
    return MobileNetV1(num_classes=num_classes, width_multiplier=width_multiplier, resolution_multiplier=resolution_multiplier, dropout_rate=dropout_rate)