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
def evaluate_and_continue_training(model, train_loader, val_loader, criterion, optimizer, device, initial_epochs=200, scheduler=None, save_path='resnet_model.pth'):
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

    Returns:
        None
    """
    target_accuracy = 0.90  # Set target accuracy to 90%
    achieved_accuracy = 0.0
    epochs = initial_epochs
    
    # Use mixed precision training to reduce memory usage
    scaler = GradScaler()
    
    while achieved_accuracy < target_accuracy:
        print(f"Starting training for {epochs} epochs.")
        
        for epoch in range(epochs):
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
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.4f}")
        
        # Retrieve and save the training metrics plot
        metrics = model.get_metrics_per_iteration()
        plot_metrics(metrics, save_path='training_metrics.png')

        # Save the model
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}.")
        
        # Evaluate model on the validation set
        eval_results = evaluate_model(model, val_loader, criterion, device)
        achieved_accuracy = eval_results['accuracy']
        print(f"Validation Accuracy: {achieved_accuracy * 100:.2f}%")

        # Check if target accuracy is reached, else continue training with more epochs
        if achieved_accuracy < target_accuracy:
            print("Target accuracy not reached, increasing epochs and continuing training.")
            epochs += 50  # Increment epochs by 50 if target not reached