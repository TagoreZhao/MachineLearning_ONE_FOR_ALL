# utils/training.py
import torch

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=1, save_path='mini_cnn_model.pth'):
    """
    Train and validate the model.

    Args:
        model: The neural network model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer for training.
        device: Device to use ('cuda' or 'cpu').
        num_epochs: Number of epochs to train the model.
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

        # Validation phase
        model.eval()  # Set model to evaluation mode
        epoch_val_loss = 0.0
        epoch_val_accuracy = 0.0
        val_batches = 0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                # Move validation inputs and targets to the specified device
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)

                # Forward pass
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_targets).item()
                val_accuracy = (val_outputs.argmax(dim=1) == val_targets).float().mean().item()

                # Accumulate validation loss and accuracy
                epoch_val_loss += val_loss
                epoch_val_accuracy += val_accuracy
                val_batches += 1

                # Track validation metrics per batch
                model.track_metrics_iteration(val_loss=val_loss, val_accuracy=val_accuracy)

        # Average the accumulated metrics over the validation dataset
        epoch_val_loss /= val_batches
        epoch_val_accuracy /= val_batches

        # Track the averaged metrics for the epoch
        model.track_metrics_iteration(
            train_loss=epoch_train_loss, 
            train_accuracy=epoch_train_accuracy,
            val_loss=epoch_val_loss, 
            val_accuracy=epoch_val_accuracy
        )

        # Print results for each epoch
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.4f}")

    # Save the model after training
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}.")
