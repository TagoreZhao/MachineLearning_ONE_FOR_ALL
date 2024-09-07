# utils/visualization.py
import matplotlib.pyplot as plt

def plot_metrics(metrics):
    """
    Plot training loss and accuracy against iterations.

    Args:
        metrics (dict): A dictionary containing lists of tracked metrics per iteration.

    Returns:
        None
    """
    train_loss = metrics['train_loss']
    train_accuracy = metrics['train_accuracy']

    # Determine the number of iterations based on the tracked metrics
    iterations = range(1, len(train_loss) + 1)

    plt.figure(figsize=(14, 5))

    # Plot Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(iterations, train_loss, label='Training Loss', color='blue')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Iterations')
    plt.legend()

    # Plot Training Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(iterations, train_accuracy, label='Training Accuracy', color='green')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy vs. Iterations')
    plt.legend()

    plt.tight_layout()
    plt.show()
