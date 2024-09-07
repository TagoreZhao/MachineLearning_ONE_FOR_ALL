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

class BinaryConv2d(nn.Conv2d):
    """
    Binary Convolution Layer that uses binarized weights and activations.
    
    This class overrides the standard convolution operation by binarizing 
    the weights and inputs during the forward pass using the sign method. 
    This approach reduces computational complexity and memory usage.
    """
    def forward(self, input):
        binarized_weights = self.weight.sign()  # Binarize weights to -1 and +1
        input = input.sign()  # Binarize input
        return F.conv2d(input, binarized_weights, self.bias, self.stride, self.padding, self.dilation, self.groups)

class PreBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(PreBlock, self).__init__()
        self.binary_conv1 = BinaryConv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.prelu1 = nn.PReLU(num_parameters=out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.binary_depthwise = BinaryConv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=out_channels, bias=False)
        self.prelu2 = nn.PReLU(num_parameters=out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.binary_conv3 = BinaryConv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.prelu3 = nn.PReLU(num_parameters=out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Adjust channels if needed for skip connections
        self.adjust_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False) if in_channels != out_channels or stride != 1 else nn.Identity()

    def forward(self, x):
        # Initial Binary Activation -> 1x1 Conv -> PReLU -> BatchNorm
        residual = self.adjust_channels(x)  # Adjust residual size
        x = x.sign()
        out = self.binary_conv1(x)
        out = self.prelu1(out)
        out = self.bn1(out)

        # Nested Depthwise 3x3 Conv block
        out = out.sign()
        out = self.binary_depthwise(out)
        out = self.prelu2(out)
        out = self.bn2(out)

        # Nested 1x1 Conv block
        out = out.sign()
        out = self.binary_conv3(out)
        out = self.prelu3(out)
        out = self.bn3(out)

        # Add skip connection
        out += residual
        return out

class MidBlock(nn.Module):
    """
    MidBlock - Enhances intermediate feature representations with nested binary operations.

    The block structure is:
    Input -> (Bin Activate -> Bin Depthwise 3x3 Conv -> PReLU -> BatchNorm) 
    -> Bin Activate -> Bin 1x1 Conv -> PReLU -> BatchNorm 
    -> (Bin Activate -> Bin 1x1 Conv -> PReLU -> BatchNorm) -> Output.
    
    Skip connections are included within the nested sequences to preserve information flow.

    Args:
        in_channels (int): Number of input channels (M).
        out_channels (int): Number of output channels (N).
        stride (int, optional): Stride of the convolution. Default is 1.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(MidBlock, self).__init__()
        
        # First 3x3 Depthwise Convolution, keeping channels M -> M
        self.binary_depthwise = BinaryConv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.prelu1 = nn.PReLU(num_parameters=in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)

        # Transition from M channels to N channels using 1x1 Convolution
        self.binary_conv1 = BinaryConv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.prelu2 = nn.PReLU(num_parameters=out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Second 1x1 Convolution with N -> N transition
        self.binary_conv2 = BinaryConv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.prelu3 = nn.PReLU(num_parameters=out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Adjust channels for skip connections if the input and output channels differ
        self.adjust_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False) if in_channels != out_channels or stride != 1 else nn.Identity()

    def forward(self, x):
        # First step: (DxDxM) -> (DxDxM) with depthwise convolution
        residual1 = x  # Adjust residual if needed
        out = x.sign()
        out = self.binary_depthwise(out)
        out = self.prelu1(out)
        out = self.bn1(out)

        # Add the first skip connection
        out += residual1

        # Second step: (DxDxM) -> (DxDxN) using the first 1x1 convolution
        out = out.sign()
        out = self.binary_conv1(out)
        out = self.prelu2(out)
        out = self.bn2(out)

        # Third step: (DxDxN) -> (DxDxN) using the second 1x1 convolution with skip connection
        residual2 = out  # Keep the residual for skip connection
        out = out.sign()
        out = self.binary_conv2(out)
        out = self.prelu3(out)
        out = self.bn3(out)

        # Add the second skip connection
        out += residual2
        return out


class PostBlock(nn.Module):
    """
    PostBlock - Finalizes feature processing with binary operations and skip connections.

    This block structure includes multiple nested binary convolutions with skip connections
    to enhance the network's depth and performance.

    Args:
        in_channels (int): Number of input channels (M).
        out_channels (int): Number of output channels (N).
        stride (int, optional): Stride of the convolution. Default is 1.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(PostBlock, self).__init__()

        # First Depthwise Convolution, keeping channels M -> M
        self.binary_depthwise = BinaryConv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.prelu1 = nn.PReLU(num_parameters=in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)

        # Second Convolution, keeping channels M -> M
        self.binary_conv1 = BinaryConv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.prelu2 = nn.PReLU(num_parameters=in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)

        # Third Convolution, keeping channels M -> M
        self.binary_conv2 = BinaryConv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.prelu3 = nn.PReLU(num_parameters=in_channels)
        self.bn3 = nn.BatchNorm2d(in_channels)

        # Final Convolution, changing channels M -> N
        self.binary_conv3 = BinaryConv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.prelu4 = nn.PReLU(num_parameters=out_channels)
        self.bn4 = nn.BatchNorm2d(out_channels)

        # Adjust channels if needed for skip connections
        self.adjust_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # First sequence with skip connection: (DxDxM) -> (DxDxM)
        residual1 = x  # Keep the residual for skip connection
        out = x.sign()
        out = self.binary_depthwise(out)
        out = self.prelu1(out)
        out = self.bn1(out)

        out += residual1  # Add first skip connection

        # Second sequence with skip connection: (DxDxM) -> (DxDxM)
        residual2 = out  # Keep the residual for skip connection
        out = out.sign()
        out = self.binary_conv1(out)
        out = self.prelu2(out)
        out = self.bn2(out)

        out += residual2  # Add second skip connection

        # Third sequence without skip connection: (DxDxM) -> (DxDxM)
        out = out.sign()
        out = self.binary_conv2(out)
        out = self.prelu3(out)
        out = self.bn3(out)

        # Final sequence, changing channels: (DxDxM) -> (DxDxN)
        out = out.sign()
        out = self.binary_conv3(out)
        out = self.prelu4(out)
        out = self.bn4(out)

        return out


class StandardDWBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(StandardDWBlock, self).__init__()
        self.binary_depthwise = BinaryConv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.prelu1 = nn.PReLU(num_parameters=in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.binary_pointwise = BinaryConv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.prelu2 = nn.PReLU(num_parameters=out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Adjust channels if needed for skip connections
        self.adjust_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False) if in_channels != out_channels or stride != 1 else nn.Identity()

    def forward(self, x):
        # Adjust channels for residual connection
        residual = self.adjust_channels(x)

        # Binary Activation -> Depthwise Convolution -> PReLU -> BatchNorm
        x = x.sign()
        x = self.binary_depthwise(x)
        x = self.prelu1(x)
        x = self.bn1(x)

        # Binary Activation -> Pointwise Convolution -> PReLU -> BatchNorm
        x = x.sign()
        x = self.binary_pointwise(x)
        x = self.prelu2(x)
        x = self.bn2(x)

        # Add skip connection
        x += residual
        return x
