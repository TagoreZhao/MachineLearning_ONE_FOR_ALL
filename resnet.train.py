# train_resnet.py
import torch
import torch.optim as optim
import torch.nn as nn
from models.resnet import resnet34, resnet50  # Import your ResNet models
from utils.training import train_model  # Import the training function
from utils.dataset import get_CIFAR_10  # Function to load CIFAR-10 data
from utils.visualization import plot_metrics  # Import the plot_metrics function
from torch.cuda.amp import GradScaler, autocast  # For mixed precision training

# Set device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load CIFAR-10 dataset with data augmentation suitable for ResNet
train_loader, val_loader = get_CIFAR_10(batch_size=64, num_workers=5, agumentation='resnet')

# Initialize ResNet-34 model
model = resnet34(num_classes=10).to(device)  # Change to resnet50() if you want to train ResNet-50

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)

# Define a learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # Reduces LR at specified epochs

# Use mixed precision training to reduce memory usage
scaler = GradScaler()

# Modified training loop with mixed precision
def train_model_with_amp(model, train_loader, criterion, optimizer, device, num_epochs=200, scheduler=None, save_path='model.pth'):
    """
    Train the model using Automatic Mixed Precision (AMP) to save memory.

    Args:
        model: The neural network model to train.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer for training.
        device: Device to use ('cuda' or 'cpu').
        num_epochs: Number of epochs to train the model.
        scheduler: Learning rate scheduler, if any.
        save_path: File path to save the trained model.

    Returns:
        None
    """
    model.to(device)
    
    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        epoch_train_accuracy = 0.0
        train_batches = 0
        
        # Set the model to training mode
        model.train()
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass with AMP
            with autocast():  # Enable mixed precision
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            scaler.scale(loss).backward()  # Scale the loss for mixed precision
            scaler.step(optimizer)
            scaler.update()

            # Calculate training accuracy
            train_accuracy = (outputs.argmax(dim=1) == targets).float().mean().item()
            
            # Accumulate loss and accuracy
            epoch_train_loss += loss.item()
            epoch_train_accuracy += train_accuracy
            train_batches += 1
            
            # Track metrics per iteration
            model.track_metrics_iteration(train_loss=loss.item(), train_accuracy=train_accuracy)
        
        # Average the accumulated metrics over the training dataset
        epoch_train_loss /= train_batches
        epoch_train_accuracy /= train_batches
        
        # Step the scheduler if provided
        if scheduler is not None:
            scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.4f}")
    
    # Save the model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}.")

# Train ResNet-34 (or ResNet-50) with mixed precision to save memory
train_model_with_amp(model, train_loader, criterion, optimizer, device, num_epochs=200, scheduler=scheduler, save_path='resnet_model.pth')

# Retrieve and save the training metrics plot
metrics = model.get_metrics_per_iteration()
plot_metrics(metrics, save_path='training_metrics.png')
