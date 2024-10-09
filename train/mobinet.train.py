import torch
import torch.optim as optim
import torch.nn as nn
from models.mobinet import MoBiNet
from models.layers import PreBlock, MidBlock, StandardDWBlock, PostBlock  # Ensure paths are correct based on your structure
from utils.dataset import get_CIFAR_10  # Function to load CIFAR-10 data
from utils.training import evaluate_and_continue_training  # Import the training and evaluation function
from utils.visualization import plot_metrics  # Import the plot_metrics function

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load CIFAR-10 dataset with data augmentation suitable for MoBiNet
train_loader, val_loader = get_CIFAR_10(batch_size=32, num_workers=5, augmentation='resnet')  # Adjust augmentation as needed

# Initialize MoBiNet with a specific block type (choose the block you want to test)
model = MoBiNet(block_type=MidBlock, num_blocks=4, num_classes=10).to(device)  # Replace MidBlock with desired block type

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# Define learning rate scheduler (optional)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# Train and evaluate the MoBiNet model, with training continuation until the target accuracy is achieved
evaluate_and_continue_training(model, train_loader, val_loader, criterion, optimizer, device, initial_epochs=20, scheduler=scheduler, save_path='mobinet_model.pth', clip_value=0.1)