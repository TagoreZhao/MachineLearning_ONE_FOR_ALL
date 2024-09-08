# One for All For CIFAR10

## Directory Structure
```
CNN-CIFAR10/
│
├── data/
│   ├── __init__.py            # Initialize the data module
│
├── models/
│   ├── __init__.py            # Initialize the models module
│   ├── base_model.py          # BaseModel class (already implemented)
│   ├── custom_cnn.py          # Custom CNN model (already implemented, this is a small model that can be used to test)
│   ├── resnet.py              # Implementation of ResNet architectures (ResNet18, ResNet34, etc.)
│   ├── mobinet.py             # Implementation of Mobile Binary Net architecture
│   └── layers.py              # Custom layers or building blocks used in ResNet and MobileNet (e.g., residual blocks, depthwise separable convs)
│
├── utils/
│   ├── __init__.py            # Initialize the utils module
│   ├── training.py            # Training functions (move your current training function here)
│   ├── evaluation.py          # Evaluation functions to test accuracy on validation/test sets
│   ├── visualization.py       # Functions for visualizing training results, like loss and accuracy plots
│   ├── dataset.py             # Code to load and preprocess the CIFAR-10 dataset
│   ├── augmentation.py        # Optional: Data augmentation techniques (flip, crop, etc.)
│   └── helper.py              # Helper functions such as saving models, loading checkpoints, etc.
│
├── config.py                  # Configuration file for hyperparameters (learning rate, batch size, etc.)
├── requirements.txt           # List of required Python packages (e.g., PyTorch, torchvision)
├── README.md                  # Documentation for setting up and using the project
├── test.ipynb                 # Jupyter notebook for testing models, visualizing results, etc.
├── resnet.train.py            # script to train resnet to 90% accuracy
└── .gitignore                 # Specify which files and directories to ignore in version control
```


