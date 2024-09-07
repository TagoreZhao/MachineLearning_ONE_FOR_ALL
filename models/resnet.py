# models/resnet.py
import torch
import torch.nn as nn
from .layers import ResNetBasicBlock, ResNetBottleneckBlock
from .base_model import BaseModel

class ResNet(BaseModel):
    """
    ResNet Architecture for various depths (18, 34, 50, 101, 152).

    Description:
    - Uses a series of convolutional layers followed by residual blocks.
    - Tracks training and validation metrics for visualization.
    - Supports different block types: BasicBlock for shallower networks and BottleneckBlock for deeper networks.
    
    Parameters:
    - block: The block type (either ResNetBasicBlock or ResNetBottleneckBlock).
    - layers: A list defining the number of blocks in each layer.
    - num_classes: Number of output classes (default is 10 for CIFAR-10).
    - shortcut_type: Type of shortcut connection used in the blocks ('A', 'B', or 'C').
    """
    def __init__(self, block, layers, num_classes=10, shortcut_type='B'):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Create ResNet layers using _make_layer function
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, shortcut_type=shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, shortcut_type=shortcut_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, shortcut_type=shortcut_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, shortcut_type=shortcut_type)

        # Fully connected layer for classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1, shortcut_type='B'):
        """
        Create a layer of residual blocks.
        
        Parameters:
        - block: The block type (ResNetBasicBlock or ResNetBottleneckBlock).
        - out_channels: Number of output channels for the blocks.
        - blocks: Number of blocks in this layer.
        - stride: Stride for the first block in this layer.
        - shortcut_type: Type of shortcut used in the blocks ('A', 'B', or 'C').

        Returns:
        - A sequential layer composed of multiple blocks.
        """
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [
            block(
                self.in_channels, out_channels, stride, shortcut_type=shortcut_type
            )
        ]
        self.in_channels = out_channels * block.expansion
        layers.extend(
            block(self.in_channels, out_channels, shortcut_type=shortcut_type)
            for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the ResNet model.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Factory functions to create specific ResNet models

def resnet18(num_classes=10, shortcut_type='B'):
    """
    Constructs a ResNet-18 model using ResNetBasicBlock.
    """
    return ResNet(ResNetBasicBlock, [2, 2, 2, 2], num_classes=num_classes, shortcut_type=shortcut_type)

def resnet34(num_classes=10, shortcut_type='B'):
    """
    Constructs a ResNet-34 model using ResNetBasicBlock.
    """
    return ResNet(ResNetBasicBlock, [3, 4, 6, 3], num_classes=num_classes, shortcut_type=shortcut_type)

def resnet50(num_classes=10, shortcut_type='B'):
    """
    Constructs a ResNet-50 model using ResNetBottleneckBlock.
    """
    return ResNet(ResNetBottleneckBlock, [3, 4, 6, 3], num_classes=num_classes, shortcut_type=shortcut_type)

def resnet101(num_classes=10, shortcut_type='B'):
    """
    Constructs a ResNet-101 model using ResNetBottleneckBlock.
    """
    return ResNet(ResNetBottleneckBlock, [3, 4, 23, 3], num_classes=num_classes, shortcut_type=shortcut_type)

def resnet152(num_classes=10, shortcut_type='B'):
    """
    Constructs a ResNet-152 model using ResNetBottleneckBlock.
    """
    return ResNet(ResNetBottleneckBlock, [3, 8, 36, 3], num_classes=num_classes, shortcut_type=shortcut_type)
