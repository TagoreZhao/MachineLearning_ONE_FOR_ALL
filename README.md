# One for All For Machine Learning

## Directory Structure
```
CNN-CIFAR10/
│
├── data/                          # Contains CIFAR-10 dataset and related files
│   ├── cifar-10-batches-py/        # Extracted CIFAR-10 dataset batches
│   ├── cifar-10-python.tar.gz      # Compressed CIFAR-10 dataset
│   ├── __init__.py                 # Initialize the data module
│   └── __pycache__/                # Compiled Python cache files
│
├── MITcourse/                      # Jupyter notebooks from MIT course labs
│   ├── Copy_of_Lab1.ipynb          
│   ├── Copy_of_Lab2.ipynb          
│   ├── Copy_of_Lab3.ipynb          
│   ├── Copy_of_Lab4.ipynb          
│   └── requirements.txt            # Python packages required for MIT course labs
│
├── models/                         # Model architectures and custom layers
│   ├── base_model.py               # Base model class for all networks
│   ├── CustomCNN.py                # Custom CNN model implementation
│   ├── __init__.py                 # Initialize the models module
│   ├── layers.py                   # Custom layers and building blocks for models
│   ├── MobileNetV1.py              # MobileNetV1 architecture implementation
│   ├── MobileNetV2.py              # MobileNetV2 architecture implementation
│   ├── mobinet.py                  # MobileNet-based architecture with modifications
│   ├── __pycache__/                # Compiled Python cache files for the models
│   ├── resnet.py                   # ResNet architecture implementation (ResNet18, ResNet34, etc.)
│   └── RNN_np.py                   # Implementation of RNN using numpy
│
├── picture/                        # Visualizations of model performance
│   ├── Enhanced_Mobile_Neural_Network_Outline.png    # Architecture diagram for MobileNet
│   ├── MobileNetV1_epoch_120.png                    # Training metrics for MobileNetV1 (120 epochs)
│   ├── MobileNetV1(ndp)_epoch_380.png               # Metrics for modified MobileNetV1 (380 epochs)
│   ├── mobile_v2_epoch_1000.png                     # MobileNetV2 performance after 1000 epochs
│   ├── resnet50_epoch_801.png                       # ResNet50 training metrics after 801 epochs
│   └── training_metrics_epoch_2040.png              # General training metrics after 2040 epochs
│
├── train/                          # Training scripts for different models
│   ├── mobilenetv1.train.py        # Training script for MobileNetV1
│   ├── mobilenetv2.train.py        # Training script for MobileNetV2
│   ├── mobinet.train.py            # Training script for MobileNet (modified)
│   └── resnet.train.py             # Training script for ResNet (target: 90% accuracy)
│
├── utils/                          # Utility functions for training, evaluation, and dataset handling
│   ├── augmentation.py             # Data augmentation techniques
│   ├── DataReader.py               # Helper function to read and process data
│   ├── dataset.py                  # Code to load and preprocess the CIFAR-10 dataset
│   ├── evaluation.py               # Functions to evaluate model performance
│   ├── helper.py                   # Helper functions (saving/loading models, checkpoints)
│   ├── __init__.py                 # Initialize the utils module
│   ├── training.py                 # Functions for training models
│   └── visualization.py            # Functions to visualize training results (loss, accuracy, etc.)
│
├── __pycache__/                    # Compiled Python cache files for various scripts
│   └── config.cpython-310.pyc
│
├── .gitignore                      # Specifies files/directories to ignore in version control
├── config.py                       # Configuration file for hyperparameters (learning rate, batch size, etc.)
├── environment.yml                 # Conda environment configuration for dependency management
├── input.txt                       # Input data for training/testing
├── mini_cnn_model.pth              # Saved model weights for a small custom CNN model
├── presentation.ipynb              # Notebook for recording model results and experiments
├── README.md                       # Project documentation and description
├── requirements.txt                # List of required Python packages (e.g., PyTorch, torchvision)
└── test.ipynb                      # Notebook for testing models and visualizing results

```
# T5 Model Implementation Workflow

This section provides a detailed workflow and checkpoints for implementing a **T5 (Text-to-Text Transfer Transformer)** model from scratch using PyTorch.

## 1. Folder Structure Setup for T5
You can expand the `models` and `utils` directories to implement the T5 architecture:
- `models/t5_model.py`: The full T5 model.
- `models/encoder.py`: Encoder implementation for T5.
- `models/decoder.py`: Decoder implementation for T5.
- `models/attention.py`: Multi-head attention mechanism.
- `utils/training_t5.py`: Training loop for T5.
- `utils/evaluation_t5.py`: Evaluation functions specific to the T5 model.

## 2. Checklist of Components to Implement

### **Data Preprocessing**
- **Checkpoint**: Set up a tokenizer for T5. T5 uses a text-to-text format, so handle tokenizing both inputs (source text) and outputs (target text).
  - Add a `tokenizer.py` or `data_processing.py` in the `data/` folder for text preprocessing.

### **T5 Encoder**
- **Checkpoint**: Create an encoder following the transformer architecture. The encoder consists of embedding, positional encoding, self-attention, feed-forward layers, and normalization.
  - File: `models/encoder.py`
  - **To-Do**:
    - Input Embedding Layer.
    - Positional Encoding.
    - Multi-head Self-Attention Layer.
    - Layer Normalization.
    - Feed-Forward Network.
    - Dropout.
    - Stack multiple encoder layers.



### **T5 Decoder**
- **Checkpoint**: Implement the decoder with causal (masked) self-attention, cross-attention (linking to encoder output), feed-forward network, and normalization.
  - File: `models/decoder.py`
  - **To-Do**:
    - Input Embedding Layer.
    - Positional Encoding.
    - Causal Self-Attention (masked).
    - Cross-Attention with encoder output.
    - Feed-Forward Network.
    - Stack multiple decoder layers.

### **Attention Mechanism**
- **Checkpoint**: Implement multi-head attention with scaled dot-product attention.
  - File: `models/attention.py`
  - **To-Do**:
    - Scaled Dot-Product Attention.
    - Multi-Head Attention Layer.

### **Model Assembly**
- **Checkpoint**: Assemble the T5 model by combining the encoder, decoder, and final linear layers.
  - File: `models/t5_model.py`
  - **To-Do**:
    - Build encoder-decoder architecture.
    - Include masking for padding and future tokens.
    - Output logits for the target vocabulary.

## 3. Training Setup
- **Checkpoint**: Prepare a training loop for T5, similar to your existing `training.py` for CNN models, but adapted for sequence-to-sequence tasks.
  - File: `utils/training_t5.py`
  - **To-Do**:
    - Define training loop with CrossEntropyLoss.
    - Use teacher forcing for faster convergence.
    - Implement learning rate scheduler and AdamW optimizer.
    - Incorporate checkpoint saving.

## 4. Evaluation and Testing
- **Checkpoint**: Implement evaluation functions to check the model's performance on sequence generation tasks.
  - File: `utils/evaluation_t5.py`
  - **To-Do**:
    - Compute accuracy or BLEU score for tasks like translation/summarization.
    - Handle greedy decoding, beam search, or top-k sampling for generating sequences.

## 5. Hyperparameter Configuration
- **Checkpoint**: Add hyperparameters to the `config.py` file. This helps manage training parameters, model size, learning rates, batch size, etc.
  - File: `config.py`
  - **To-Do**:
    - Define model dimensions (embedding size, number of heads, number of layers).
    - Set optimizer settings (learning rate, weight decay).
    - Specify number of epochs, batch size, etc.

## 6. Visualization & Logging
- **Checkpoint**: Visualize training performance (loss, accuracy) over time.
  - File: `utils/visualization.py`
  - **To-Do**:
    - Plot loss and accuracy per epoch.
    - Track gradients or attention weights (optional).
    - Use TensorBoard or Matplotlib for visualizations.

## 7. Testing and Debugging
- **Checkpoint**: Test the model on simple tasks (e.g., text summarization, translation) to ensure everything works.
  - File: `test.ipynb`
  - **To-Do**:
    - Ensure model compiles without errors.
    - Perform a few inference steps to see if the model generates reasonable text.

## 8. Training Pipeline
Once your model architecture is ready:
- **Checkpoint**: Train the T5 model on a small dataset (e.g., a subset of machine translation or summarization data).
  - Start with a small model (e.g., T5-small) for debugging.
  - Save model checkpoints regularly to monitor progress.

---

## Final Checklist Summary:
1. **Data Preprocessing**: Tokenizer and text preprocessing functions.
2. **T5 Encoder**: Positional encoding, multi-head attention, feed-forward network.
3. **T5 Decoder**: Masked self-attention, cross-attention, feed-forward network.
4. **Attention Mechanism**: Scaled dot-product and multi-head attention.
5. **Model Assembly**: Combine encoder-decoder into the full T5 model.
6. **Training Setup**: Loss function, optimizer, scheduler, checkpoint saving.
7. **Evaluation and Testing**: Performance metrics, decoding strategies.
8. **Hyperparameter Tuning**: Config file for managing model hyperparameters.
9. **Visualization & Logging**: Track loss, accuracy, and visualize results.
10. **Final Testing**: Run on simple text generation tasks.



