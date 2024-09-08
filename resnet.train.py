# resnet.train.py
import torch
import torch.optim as optim
import torch.nn as nn
from models.resnet import resnet18  # Import your ResNet models
from utils.dataset import get_CIFAR_10  # Function to load CIFAR-10 data
from utils.training import evaluate_and_continue_training  # Import the training function with early stopping
from torch.cuda.amp import GradScaler, autocast  # For mixed precision training

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load CIFAR-10 dataset with data augmentation suitable for ResNet
train_loader, val_loader = get_CIFAR_10(batch_size=256, num_workers=8, augmentation='resnet')

# Initialize ResNet-34 model (or switch to ResNet-50)
model = resnet18(num_classes=10).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)

# Define One Cycle Learning Rate Scheduler
max_lr = 0.1  # Set maximum learning rate
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_loader), epochs=200)

# Use mixed precision training to reduce memory usage
scaler = GradScaler()

# Train ResNet-34 (or ResNet-50) with early stopping and subset validation criteria
evaluate_and_continue_training(
    model, 
    train_loader, 
    val_loader, 
    criterion, 
    optimizer, 
    device, 
    initial_epochs=1, 
    scheduler=scheduler, 
    save_path='resnet_model.pth'
)
