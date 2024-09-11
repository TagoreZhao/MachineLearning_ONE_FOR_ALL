# models/mobilenetv2.py
import torch
import torch.nn as nn
from .layers import InvertedResidualBlock
from .base_model import BaseModel

class MobileNetV2(BaseModel):
    """
    MobileNetV2 Architecture with resolution and width multipliers.

    Description:
    - Uses inverted residual blocks with linear bottlenecks.
    - Includes resolution and width multipliers to adjust model size and computational cost.

    Parameters:
    - num_classes: Number of output classes (default is 10 for CIFAR-10).
    - width_multiplier: Scales the number of channels in each layer.
    - resolution_multiplier: Scales the input resolution of the model.
    """
    def __init__(self, num_classes=10, width_multiplier=1.0, resolution_multiplier=1.0, dropout_rate=0.2):
        super(MobileNetV2, self).__init__()
        self.width_multiplier = width_multiplier
        self.resolution_multiplier = resolution_multiplier
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        def adjust_channels(channels):
            return max(1, int(channels * self.width_multiplier))

        # Initial layer
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, adjust_channels(32), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(adjust_channels(32)),
            nn.ReLU6(inplace=True)
        )

        # Configuration of inverted residual blocks
        inverted_residual_config = [
            # t, c, n, s (expand ratio, output channels, num blocks, stride)
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        # Building inverted residual blocks
        layers = []
        input_channel = adjust_channels(32)
        for t, c, n, s in inverted_residual_config:
            output_channel = adjust_channels(c)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidualBlock(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        self.layers = nn.Sequential(*layers)

        # Last several layers before the classification head
        self.last_conv = nn.Sequential(
            nn.Conv2d(input_channel, adjust_channels(1280), kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(adjust_channels(1280)),
            nn.ReLU6(inplace=True)
        )

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(adjust_channels(1280), num_classes)

    def forward(self, x):
        """
        Forward pass of the MobileNetV2 model.
        """
        if self.resolution_multiplier != 1.0:
            height, width = x.size(2), x.size(3)
            new_height, new_width = int(height * self.resolution_multiplier), int(width * self.resolution_multiplier)
            x = nn.functional.interpolate(x, size=(new_height, new_width), mode='bilinear', align_corners=False)

        x = self.initial_conv(x)
        x = self.layers(x)
        x = self.last_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Factory function to create MobileNetV2 model
def mobilenet_v2(num_classes=10, width_multiplier=1.0, resolution_multiplier=1.0, dropout_rate=0.2):
    """
    Constructs a MobileNetV2 model with specified multipliers.
    """
    return MobileNetV2(num_classes=num_classes, width_multiplier=width_multiplier, resolution_multiplier=resolution_multiplier, dropout_rate=dropout_rate)