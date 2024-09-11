# utils/training.py
import torch
from torch.cuda.amp import GradScaler, autocast  # For mixed precision training
from utils.visualization import plot_metrics  # Import the plot_metrics function
from utils.evaluation import evaluate_model  # Import your evaluate_model function

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

# Define a function that trains and evaluates, and continues training until >90% accuracy is achieved
def save_checkpoint(model, optimizer, epoch, val_accuracy, filepath):
    """
    Saves a checkpoint of the model with its current state.

    Parameters:
    - model: The model to save.
    - optimizer: The optimizer used during training.
    - epoch: The current epoch number.
    - val_accuracy: The validation accuracy of the model.
    - filepath: The path to save the checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_accuracy
    }
    torch.save(checkpoint, filepath)
    print(f"New best model saved with validation accuracy: {val_accuracy:.4f}")


def load_checkpoint(filepath, model, optimizer=None):
    """
    Loads a checkpoint of the model, including model weights, optimizer state, and the epoch number.

    Args:
        filepath: Path to the checkpoint file.
        model: The model to load the state into.
        optimizer: The optimizer to load the state into (optional).

    Returns:
        model: The model with loaded weights.
        optimizer: The optimizer with loaded state (if provided).
        epoch: The epoch number from the checkpoint.
        best_val_accuracy: The best validation accuracy from the checkpoint.
    """
    checkpoint = torch.load(filepath)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if optimizer is provided
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    best_val_accuracy = checkpoint['val_accuracy']
    
    print(f"Checkpoint loaded: Epoch {epoch}, Best Validation Accuracy: {best_val_accuracy:.4f}")
    
    return model, optimizer, epoch, best_val_accuracy

def evaluate_and_continue_training(model, train_loader, val_loader, criterion, optimizer, device, initial_epochs, max_epochs = 1000, target_accuracy = 0.90, scheduler=None, save_path='resnet_model.pth', clip_value=1.0):
    """
    Train the model and evaluate validation accuracy, continue training until >90% accuracy is achieved.

    Args:
        model: The neural network model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer for training.
        device: Device to use ('cuda' or 'cpu').
        initial_epochs: Initial number of epochs to train the model.
        scheduler: Learning rate scheduler, if any.
        save_path: File path to save the trained model.
        clip_value: The maximum allowed value for gradient clipping.

    Returns:
        None
    """
    achieved_accuracy = 0.0
    best_val_accuracy = 0.0  # Track the best validation accuracy
    total_epochs = 0  # Initialize total epochs counter
    increment = 20  # Number of epochs to increment if target accuracy is not reached
    scaler = GradScaler()  # Use mixed precision training to reduce memory usage

    # Continue training until target accuracy is reached or max epochs
    while achieved_accuracy < target_accuracy and total_epochs < max_epochs:
        print(f"Starting training for the next {initial_epochs} epochs.")

        for epoch in range(initial_epochs):
            total_epochs += 1
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
                
                # Gradient clipping
                scaler.unscale_(optimizer)  # Unscale gradients before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)  # Clip gradients
                
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
            
            print(f"Epoch {total_epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.4f}")

        # Retrieve and save the training metrics plot
        metrics = model.get_metrics_per_iteration()
        plot_filename = f'training_metrics_epoch_{total_epochs}.png'
        plot_metrics(metrics, save_path=plot_filename)
        print(f"Saved training metrics plot as {plot_filename}.")

        # Evaluate model on the validation set after each set of epochs
        eval_results = evaluate_model(model, val_loader, criterion, device)
        achieved_accuracy = eval_results['accuracy']
        val_loss = eval_results['loss']  # Capture validation loss for scheduler if needed
        print(f"Validation Accuracy: {achieved_accuracy * 100:.2f}%, Validation Loss: {val_loss:.4f}")

        # Save the model if this is the best accuracy so far
        if achieved_accuracy > best_val_accuracy:
            best_val_accuracy = achieved_accuracy
            save_checkpoint(model, optimizer, epoch, best_val_accuracy, save_path)
            print(f"New best model saved to {save_path} with accuracy {best_val_accuracy:.4f}.")

        # Step the scheduler if provided
        if scheduler is not None:
            # Pass validation loss if using ReduceLROnPlateau
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
                scheduler.get_last_lr()
            else:
                scheduler.step()
        
        # Check if target accuracy is reached, stop if true
        if achieved_accuracy >= target_accuracy:
            print(f"Target accuracy reached at {total_epochs} total epochs!")
            break
        
        # If not reached, increment the epochs and continue training
        print("Target accuracy not reached, increasing epochs and continuing training.")
        initial_epochs = increment

    print(f"Training ended after {total_epochs} epochs with best validation accuracy of {best_val_accuracy:.4f}.")