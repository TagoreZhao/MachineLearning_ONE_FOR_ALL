import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import StandardDWBlock, PreBlock, MidBlock, PostBlock, BinaryConv2d
from .base_model import BaseModel

class MoBiNet(BaseModel):
    """
    MoBiNet - A modular binary neural network that uses various types of blocks.

    This class allows you to configure the network with different blocks such as PreBlock,
    MidBlock, StandardDWBlock, and PostBlock. The design ensures the correct flow of input
    and output dimensions between layers.

    Args:
        block_type (nn.Module): The type of block to use (e.g., MidBlock, PreBlock).
        num_blocks (int): Number of blocks to repeat in the network.
        num_classes (int): Number of output classes for the final fully connected layer.
    """
    def __init__(self, block_type, num_blocks, num_classes=10):
        super(MoBiNet, self).__init__()

        # Initial Convolution
        self.init_conv = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.init_bn = nn.BatchNorm2d(32)
        self.init_prelu = nn.PReLU(num_parameters=32)

        # Define the channel dimensions across the blocks
        # Ensure that the dimensions align properly with the block transitions
        channels = [32, 64, 128, 256, 512]  # Example channel configuration that aligns with MidBlock requirements

        # Dynamically create layers with the specified block type
        self.layers = self._make_layers(block_type, channels, num_blocks)

        # Final layers before classification
        self.final_conv = BinaryConv2d(channels[-1], 1024, kernel_size=1, bias=False)
        self.final_bn = nn.BatchNorm2d(1024)
        self.final_prelu = nn.PReLU(num_parameters=1024)

        # Global pooling and final fully connected layer
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(1024, num_classes)

    def _make_layers(self, block_type, channels, num_blocks):
        """
        Create layers of the network by stacking the specified blocks.

        Args:
            block_type (nn.Module): The type of block to use.
            channels (list): List of channel sizes for each block stage.
            num_blocks (int): Number of blocks to use in each stage.

        Returns:
            nn.Sequential: Sequential model containing the stacked blocks.
        """
        layers = []
        in_channels = channels[0]

        for i in range(1, len(channels)):
            out_channels = channels[i]
            for _ in range(num_blocks):
                layers.append(block_type(in_channels, out_channels))
                in_channels = out_channels  # Update in_channels for the next block

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolution and activation
        x = self.init_conv(x)
        x = self.init_bn(x)
        x = self.init_prelu(x)

        # Forward through the stacked blocks
        x = self.layers(x)

        # Final layers before classification
        x = self.final_conv(x.sign())  # Binary activation before final convolution
        x = self.final_bn(x)
        x = self.final_prelu(x)

        # Global pooling and fully connected output
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
        x = self.fc(x)
        return x