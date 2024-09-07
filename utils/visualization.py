# utils/visualization.py
import matplotlib.pyplot as plt

def plot_metrics(metrics, save_path=None):
    """
    Plot and optionally save training and validation loss and accuracy against iterations.

    Args:
        metrics (dict): A dictionary containing lists of tracked metrics per iteration.
        save_path (str, optional): Path to save the plot image. Default is None.

    Returns:
        None
    """
    train_loss = metrics['train_loss']
    train_accuracy = metrics['train_accuracy']

    # Plot Training Loss and Accuracy
    plt.figure(figsize=(12, 5))

    # Plot Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Iterations')
    plt.legend()

    # Plot Training Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy vs. Iterations')
    plt.legend()

    plt.tight_layout()

    # Save plot if a path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}.")

    plt.show()

