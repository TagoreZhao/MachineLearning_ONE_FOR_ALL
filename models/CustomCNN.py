from torch import nn
from .base_model import BaseModel
import torch.nn.functional as F
from config import (  # Import necessary settings from config.py
    INPUT_CHANNELS, NUM_CLASSES, CONV1_OUT_CHANNELS, CONV2_OUT_CHANNELS,
    CONV3_OUT_CHANNELS, CONV4_OUT_CHANNELS, CONV5_OUT_CHANNELS, CONV6_OUT_CHANNELS,
    KERNEL_SIZE, STRIDE, PADDING, POOL_KERNEL_SIZE, POOL_STRIDE,
    DROPOUT_RATE_1, DROPOUT_RATE_2, DROPOUT_RATE_3, FC1_OUTPUT
)


class CustomCNN(BaseModel):
    def __init__(self):
        super(CustomCNN, self).__init__()

        # A very small model with only one convolutional block and one fully connected layer
        self.conv1 = nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces the spatial size by half

        # A simple fully connected layer that connects to the number of classes
        self.fc1 = nn.Linear(16 * 16 * 16, NUM_CLASSES)  # Assuming CIFAR-10, image size is 32x32

        # Tracking attributes inherited from BaseModel
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_accuracy_history = []
        self.val_accuracy_history = []

    def forward(self, x):
        # Apply first convolution, batch norm, ReLU activation, and pooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)  # Output size: [batch_size, 16, 16, 16]

        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = self.fc1(x)
        return x
