# utils/evaluation.py
import torch
import torch.nn as nn
from torch.utils.data import Subset
import random

def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate the model on a given dataset.

    Args:
        model: The neural network model to evaluate.
        data_loader: DataLoader for the dataset (validation or test set).
        criterion: Loss function used for evaluation (e.g., CrossEntropyLoss).
        device: Device to use ('cuda' or 'cpu').

    Returns:
        A dictionary containing the average loss and accuracy of the model on the dataset.
    """
    model.to(device)
    model.eval()  # Set model to evaluation mode

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Disable gradient computation to speed up evaluation
    with torch.no_grad():
        for inputs, targets in data_loader:
            # Move inputs and targets to the specified device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Update loss
            total_loss += loss.item() * inputs.size(0)

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)

    # Calculate average loss and accuracy
    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    # Print evaluation results
    print(f"Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Return metrics as a dictionary
    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }

# Function to create a subset of the validation set
def get_val_subset(val_loader, subset_size=1000):
    """
    Get a subset of the validation data for faster evaluation.

    Args:
        val_loader: Original validation DataLoader.
        subset_size: Number of samples in the subset.

    Returns:
        Subset DataLoader.
    """
    val_dataset = val_loader.dataset
    subset_indices = random.sample(range(len(val_dataset)), min(subset_size, len(val_dataset)))
    val_subset = Subset(val_dataset, subset_indices)
    return torch.utils.data.DataLoader(
        val_subset,
        batch_size=val_loader.batch_size,
        shuffle=False,
        num_workers=val_loader.num_workers,
    )

