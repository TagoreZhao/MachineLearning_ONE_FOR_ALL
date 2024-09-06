# models/base_model.py
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

        # Initialize lists to track loss and accuracy per iteration
        self.train_loss_per_iteration = []
        self.val_loss_per_iteration = []
        self.train_accuracy_per_iteration = []
        self.val_accuracy_per_iteration = []

    def forward(self, x):
        raise NotImplementedError("Forward method not implemented.")

    def track_metrics_iteration(self, train_loss=None, val_loss=None, train_accuracy=None, val_accuracy=None):
        """
        Method to record loss and accuracy after each iteration.
        Can accept None for unused metrics.
        """
        if train_loss is not None:
            print(f"Tracking training loss: {train_loss}")
            self.train_loss_per_iteration.append(train_loss)
        if val_loss is not None:
            print(f"Tracking validation loss: {val_loss}")
            self.val_loss_per_iteration.append(val_loss)
        if train_accuracy is not None:
            print(f"Tracking training accuracy: {train_accuracy}")
            self.train_accuracy_per_iteration.append(train_accuracy)
        if val_accuracy is not None:
            print(f"Tracking validation accuracy: {val_accuracy}")
            self.val_accuracy_per_iteration.append(val_accuracy)

    def get_metrics_per_iteration(self):
        """
        Method to retrieve tracked metrics for visualization per iteration.
        """
        return {
            'train_loss': self.train_loss_per_iteration,
            'val_loss': self.val_loss_per_iteration,
            'train_accuracy': self.train_accuracy_per_iteration,
            'val_accuracy': self.val_accuracy_per_iteration
        }
