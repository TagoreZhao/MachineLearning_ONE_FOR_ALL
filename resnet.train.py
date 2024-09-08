# Import necessary modules
import torch
import torch.optim as optim
import torch.nn as nn
from models.resnet import resnet18  # Ensure correct import from your module path
from utils.dataset import get_CIFAR_10
from utils.training import evaluate_and_continue_training  # Your training function
from torch.cuda.amp import GradScaler, autocast

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load dataset
train_loader, val_loader = get_CIFAR_10(batch_size=128, num_workers=4, augmentation='resnet')

# Initialize ResNet-18 model
model = resnet18(num_classes=10).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0005)

# Define learning rate scheduler
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.05, steps_per_epoch=len(train_loader), epochs=1)

# Use mixed precision training
scaler = GradScaler()

# Train the model
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

