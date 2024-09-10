import torch
import torch.optim as optim
import torch.nn as nn
from models.MobileNetV1 import mobilenet_v1  # Import your MobileNetV1 model
from utils.dataset import get_CIFAR_10  # Function to load CIFAR-10 data
from utils.training import evaluate_and_continue_training  # Import the training function with early stopping
from torch.cuda.amp import GradScaler, autocast  # For mixed precision training

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load CIFAR-10 dataset with data augmentation suitable for MobileNet
train_loader, val_loader = get_CIFAR_10(batch_size=256, num_workers=8, augmentation='mobilenetv1')

# Initialize MobileNetV1 model with width and resolution multipliers
model = mobilenet_v1(num_classes=10, width_multiplier=1.0, resolution_multiplier=1.0).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# Define learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# Use mixed precision training to reduce memory usage
scaler = GradScaler()

# Train MobileNetV1 with early stopping and subset validation criteria
evaluate_and_continue_training(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    device, 
    initial_epochs=1, 
    scheduler=scheduler, 
    max_epochs=1000,
    target_accuracy=0.5,
    save_path='mobilenet_v1_model.pth', 
    clip_value=1.0  # Gradient clipping value, adjust if necessary
)