# models/layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBasicBlock(nn.Module):
    """
    ResNet Basic Block for ResNet-18 and ResNet-34 models.
    
    Description:
    - This block consists of two 3x3 convolutional layers with Batch Normalization and ReLU activation.
    - It includes a shortcut (skip connection) that allows the input to bypass the convolutions and be added 
      directly to the output, enhancing gradient flow and reducing vanishing gradient issues.
    - The block can be configured to use different types of shortcuts for increasing dimensions:
        - **Type A**: Uses zero-padding to match dimensions, making it parameter-free for increased efficiency.
        - **Type B**: Uses projection shortcuts with 1x1 convolutions when dimensions increase, otherwise uses identity shortcuts.
        - **Type C**: Always uses projection shortcuts for flexibility and stronger representational capacity.
    
    Parameters:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.
    - stride (int): Stride of the first convolution layer. Used for downsampling.
    - shortcut_type (str): Type of shortcut connection ('A', 'B', or 'C').

    Expansion:
    - The expansion attribute determines the output channels relative to the block’s primary convolution channels.
      For ResNetBasicBlock, expansion is 1 because there is no increase in channel dimensions in the shortcut.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, shortcut_type='A'):
        super(ResNetBasicBlock, self).__init__()
        # First 3x3 convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Second 3x3 convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut_type = shortcut_type
        # Shortcut path based on shortcut type
        self.shortcut = self._make_shortcut(in_channels, out_channels, stride)

    def _make_shortcut(self, in_channels, out_channels, stride):
        """
        Create the shortcut connection based on the specified type (A, B, or C).

        Returns:
        - A neural network layer (either identity, zero-padding, or convolution) to match the input and output dimensions.
        """
        if self.shortcut_type == 'A':
            # Type A: Zero-padding shortcuts for increasing dimensions, parameter-free
            if stride != 1 or in_channels != out_channels:
                return nn.Sequential(nn.ZeroPad2d((0, 0, 0, 0)))  # Simplified to a parameter-free identity
            return nn.Identity()
        elif self.shortcut_type == 'B':
            # Type B: Projection shortcuts for increasing dimensions, identity elsewhere
            if stride != 1 or in_channels != out_channels:
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            return nn.Identity()
        elif self.shortcut_type == 'C':
            # Type C: All shortcuts are projections
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            raise ValueError("Shortcut type must be 'A', 'B', or 'C'")

    def forward(self, x):
        # Save the input for the skip connection
        identity = self.shortcut(x)

        # First convolutional layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Second convolutional layer
        out = self.conv2(out)
        out = self.bn2(out)

        # Add the shortcut (identity) to the output
        out += identity
        out = F.relu(out)

        return out


class ResNetBottleneckBlock(nn.Module):
    """
    ResNet Bottleneck Block for deeper ResNet models (e.g., ResNet-50, ResNet-101, ResNet-152).
    
    Description:
    - This block uses three convolutional layers instead of two, designed to make deeper networks more efficient.
    - The first 1x1 convolution reduces the number of input channels (compression), the 3x3 convolution acts as 
      the main bottleneck layer, and the last 1x1 convolution expands the channels back (decompression).
    - The block helps to reduce the number of parameters while preserving the performance by utilizing these 1x1 convolutions.
    - Shortcut connections can be chosen similarly as in ResNetBasicBlock (Type A, B, or C).

    Parameters:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels in the main path.
    - stride (int): Stride of the 3x3 convolution layer.
    - shortcut_type (str): Type of shortcut connection ('A', 'B', or 'C').

    Expansion:
    - The expansion attribute is set to 4 because the final number of output channels is four times the main output channels.
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, shortcut_type='B'):
        super(ResNetBottleneckBlock, self).__init__()
        # First 1x1 convolution layer for reducing dimensions
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Second 3x3 convolution layer for spatial processing
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Third 1x1 convolution layer for restoring dimensions
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.shortcut_type = shortcut_type
        # Shortcut path based on shortcut type
        self.shortcut = self._make_shortcut(in_channels, out_channels * self.expansion, stride)

    def _make_shortcut(self, in_channels, out_channels, stride):
        """
        Create the shortcut connection based on the specified type (A, B, or C).

        Returns:
        - A neural network layer (either identity, zero-padding, or convolution) to match the input and output dimensions.
        """
        if self.shortcut_type == 'A':
            # Type A: Zero-padding shortcuts for increasing dimensions, parameter-free
            if stride != 1 or in_channels != out_channels:
                return nn.Sequential(nn.ZeroPad2d((0, 0, 0, 0)))  # Simplified to a parameter-free identity
            return nn.Identity()
        elif self.shortcut_type == 'B':
            # Type B: Projection shortcuts for increasing dimensions, identity elsewhere
            if stride != 1 or in_channels != out_channels:
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            return nn.Identity()
        elif self.shortcut_type == 'C':
            # Type C: All shortcuts are projections
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            raise ValueError("Shortcut type must be 'A', 'B', or 'C'")

    def forward(self, x):
        # Save the input for the skip connection
        identity = self.shortcut(x)

        # First 1x1 convolution layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Second 3x3 convolution layer
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        # Third 1x1 convolution layer
        out = self.conv3(out)
        out = self.bn3(out)

        # Add the shortcut (identity) to the output
        out += identity
        out = F.relu(out)

        return out


