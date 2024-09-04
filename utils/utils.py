# train.py
import torch
import torch.optim as optim
import torch.nn as nn
from models.CustomCNN import CustomCNN
from config import LEARNING_RATE, NUM_EPOCHS  # Import settings from your config file
from utils import data_loaders as dl

# Define the training function with metrics tracking
def train_model(model):
    """
    Trains the given model on the CIFAR-10 dataset.

    Args:
        model (nn.Module): The PyTorch model to be trained.
    """
    # Set device to GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the selected device
    model = model.to(device)

    # Define the loss function (Cross Entropy Loss for classification)
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer (Adam optimizer in this case)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Load CIFAR-10 training and testing datasets using your custom loader function
    trainloader, testloader = dl.get_CIFAR_10(batch_size=64, num_workers=5)

    # Lists to store loss and accuracy for plotting
    train_losses = []
    train_accuracies = []

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        # Iterate over the training data
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update running loss and accuracy
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Calculate and print average loss and accuracy for the epoch
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100. * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    print('Training complete.')
    return train_losses, train_accuracies