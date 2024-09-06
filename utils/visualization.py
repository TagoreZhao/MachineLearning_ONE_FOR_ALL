# utils/visualization.py
import matplotlib.pyplot as plt

def plot_metrics(metrics):
    """
    Plot training and validation loss and accuracy against iterations.

    Args:
        metrics (dict): A dictionary containing lists of tracked metrics per iteration.

    Returns:
        None
    """
    train_loss = metrics['train_loss']
    val_loss = metrics['val_loss']
    train_accuracy = metrics['train_accuracy']
    val_accuracy = metrics['val_accuracy']

    # Ensure the lengths match for plotting
    min_length = min(len(train_loss), len(val_loss))

    # Limit to the shortest length
    iterations = range(1, min_length + 1)
    
    # Adjust the data to match the shortest sequence
    train_loss = train_loss[:min_length]
    val_loss = val_loss[:min_length]
    train_accuracy = train_accuracy[:min_length]
    val_accuracy = val_accuracy[:min_length]

    plt.figure(figsize=(14, 5))

    # Plot Training and Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(iterations, train_loss, label='Training Loss')
    plt.plot(iterations, val_loss, label='Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs. Iterations')
    plt.legend()

    # Plot Training and Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(iterations, train_accuracy, label='Training Accuracy')
    plt.plot(iterations, val_accuracy, label='Validation Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy vs. Iterations')
    plt.legend()

    plt.tight_layout()
    plt.show()
