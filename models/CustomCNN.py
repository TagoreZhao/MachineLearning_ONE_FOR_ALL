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

        # Block 1
        self.conv1 = nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=CONV1_OUT_CHANNELS, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING)
        self.bn1 = nn.BatchNorm2d(CONV1_OUT_CHANNELS)
        self.conv2 = nn.Conv2d(in_channels=CONV1_OUT_CHANNELS, out_channels=CONV2_OUT_CHANNELS, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING)
        self.bn2 = nn.BatchNorm2d(CONV2_OUT_CHANNELS)
        self.pool1 = nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE)  # Output: [batch_size, 64, 16, 16]
        self.conv3 = nn.Conv2d(in_channels=CONV2_OUT_CHANNELS, out_channels=CONV3_OUT_CHANNELS, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING)
        self.dropout1 = nn.Dropout(DROPOUT_RATE_1)

        # Block 2
        self.conv4 = nn.Conv2d(in_channels=CONV3_OUT_CHANNELS, out_channels=CONV4_OUT_CHANNELS, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING)
        self.bn3 = nn.BatchNorm2d(CONV4_OUT_CHANNELS)
        self.conv5 = nn.Conv2d(in_channels=CONV4_OUT_CHANNELS, out_channels=CONV5_OUT_CHANNELS, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING)
        self.bn4 = nn.BatchNorm2d(CONV5_OUT_CHANNELS)
        self.pool2 = nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE)  # Output: [batch_size, 256, 8, 8]
        self.conv6 = nn.Conv2d(in_channels=CONV5_OUT_CHANNELS, out_channels=CONV6_OUT_CHANNELS, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING)
        self.dropout2 = nn.Dropout(DROPOUT_RATE_2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(CONV6_OUT_CHANNELS * 8 * 8, FC1_OUTPUT)
        self.dropout3 = nn.Dropout(DROPOUT_RATE_3)
        self.fc2 = nn.Linear(FC1_OUTPUT, NUM_CLASSES)  # 10 classes for CIFAR-10

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.dropout1(x)

        # Block 2
        x = F.relu(self.bn3(self.conv4(x)))
        x = F.relu(self.bn4(self.conv5(x)))
        x = self.pool2(x)
        x = F.relu(self.conv6(x))
        x = self.dropout2(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten the tensor

        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x
