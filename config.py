# config.py

# General Settings
SEED = 42  # Random seed for reproducibility

# Data Parameters
BATCH_SIZE = 64
NUM_WORKERS = 4  # Number of workers for data loading

# Model Parameters
INPUT_CHANNELS = 3
NUM_CLASSES = 10

# Convolutional Layer Settings
CONV1_OUT_CHANNELS = 32
CONV2_OUT_CHANNELS = 64
CONV3_OUT_CHANNELS = 128
CONV4_OUT_CHANNELS = 128
CONV5_OUT_CHANNELS = 256
CONV6_OUT_CHANNELS = 512

KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1

# Pooling Parameters
POOL_KERNEL_SIZE = 2
POOL_STRIDE = 2

# Dropout Rates
DROPOUT_RATE_1 = 0.25
DROPOUT_RATE_2 = 0.25
DROPOUT_RATE_3 = 0.5

# Fully Connected Layers
FC1_OUTPUT = 1024

# Training Parameters
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
