# utils/training.py
import torch

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=1, scheduler=None, save_path='model.pth'):
    """
    Train the model.

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
    # Move the model to the specified device
    model.to(device)

    for epoch in range(num_epochs):
        # Initialize accumulators for loss and accuracy
        epoch_train_loss = 0.0
        epoch_train_accuracy = 0.0
        train_batches = 0

        # Training phase
        model.train()  # Set model to training mode
        for inputs, targets in train_loader:
            # Move inputs and targets to the specified device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate training accuracy
            train_accuracy = (outputs.argmax(dim=1) == targets).float().mean().item()

            # Accumulate loss and accuracy
            epoch_train_loss += loss.item()
            epoch_train_accuracy += train_accuracy
            train_batches += 1

            # Track metrics per iteration (ensure it’s recorded every batch)
            model.track_metrics_iteration(train_loss=loss.item(), train_accuracy=train_accuracy)

        # Average the accumulated metrics over the training dataset
        epoch_train_loss /= train_batches
        epoch_train_accuracy /= train_batches

        # Step the scheduler based on training loss if a scheduler is provided
        if scheduler is not None:
            scheduler.step(epoch_train_loss)
            # Manually log the current learning rate
            current_lr = scheduler.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}: Learning rate adjusted to {current_lr:.6f}")

        # Print results for each epoch
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.4f}")

    # Save the model after training
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}.")
