# One for All For Machine Learning

## Directory Structure
```
MachineLearning_OFA/
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
# Workflow for Implementing LLaMA3 Model from Scratch in PyTorch

## Overview
This workflow provides a step-by-step guide to implement a LLaMA3 model from scratch in PyTorch. It includes tasks for architecture setup, model implementation, data preparation, training, and evaluation. Each task has estimated time based on the assumption of using a GPU 3080ti, which offers good performance but requires scaling the project appropriately.

---

## 1. **Environment Setup**
   - **Task:** Set up the Python environment with necessary packages and dependencies.
   - **Details:**
     - Install PyTorch and relevant libraries like `torchvision`, `datasets`, `transformers`, etc.
     - Ensure GPU compatibility with CUDA.
     - Create a project structure (you can follow your existing project file structure).
   - **Estimated Time:** 1-2 hours

---

## 2. **Understanding LLaMA3 Architecture**
   - **Task:** Review the LLaMA3 architecture to understand transformer layers, attention mechanisms, and tokenization strategies.
   - **Details:**
     - Read relevant papers and PyTorch documentation for transformer architecture.
     - Focus on layer normalization, attention heads, and feedforward networks.
     - Study multi-head self-attention and position embeddings.
   - **Estimated Time:** 1 day for reading and concept mapping

---

## 3. **Model Definition**
   - **Task:** Implement the LLaMA3 model architecture.
   - **Details:**
     - Create custom layers for self-attention, feedforward, and layer normalization.
     - Define residual connections and attention heads.
     - Implement transformer block and stack multiple blocks for the model.
     - Ensure to handle scalability by defining the model size and attention head carefully (starting with a smaller version).
   - **Estimated Time:** 2-3 days
     - **File:** `models/LLaMA3.py`
       - **Description:** Contains the full architecture of the LLaMA3 model, including transformer blocks, attention mechanisms, and embedding layers.
     - **File:** `models/transformers.py`
       - **Description:** Defines custom transformer layers used in the LLaMA3 model, such as self-attention and feed-forward networks.
     - **Checkpoint:** Basic model implementation without training

---

## 4. **Data Preprocessing**
   - **Task:** Prepare the dataset and tokenize it for input to the model.
   - **Details:**
     - Use a smaller dataset first to speed up iteration (e.g., WikiText or similar).
     - Tokenize the data using a suitable tokenizer like Byte-Pair Encoding (BPE) or SentencePiece.
     - Create PyTorch datasets and dataloaders for efficient batching.
   - **Estimated Time:** 1-2 days
     - **File:** `utils/llama_tokenizer.py`
       - **Description:** Tokenizes input text data and prepares it for training by creating PyTorch dataloaders.
     - **Checkpoint:** Data pipelines ready and tokenization implemented

---

## 5. **Model Training Loop**
   - **Task:** Write the training loop, including loss functions, optimizers, and gradient updates.
   - **Details:**
     - Define the loss function (CrossEntropyLoss for language modeling).
     - Implement an optimizer (AdamW) and learning rate scheduler (Cosine Annealing recommended for transformer models).
     - Write the training and validation loop, track loss and accuracy.
   - **Estimated Time:** 2 days
     - **File:** `utils/training_llama3.py`
       - **Description:** Contains helper functions for training the LLaMA3 model, including gradient updates, learning rate scheduling, and mixed precision support.
     - **Checkpoint:** Training loop complete and tested on a small dataset

---

## 6. **Training on Smaller Dataset**
   - **Task:** Train the LLaMA3 model on a small dataset to ensure functionality.
   - **Details:**
     - Train using a batch size that fits your GPU memory (~8-16 depending on your token size).
     - Monitor GPU usage with `nvidia-smi`.
     - Save checkpoints periodically.
   - **Estimated Time:** 3-5 days for smaller dataset training
     - **File:** `train/llama3.train.py`
       - **Description:** Training script for the LLaMA3 model. Includes initialization, loading datasets, and training the model while saving checkpoints.
     - **Checkpoint:** Model successfully trained on a small dataset (basic performance)

---

## 7. **Evaluation & Debugging**
   - **Task:** Test the model's output on validation data and debug any issues.
   - **Details:**
     - Generate text based on prompts and evaluate the output quality.
     - Check for overfitting or underfitting, and adjust hyperparameters accordingly.
     - Visualize attention weights and loss curves.
   - **Estimated Time:** 2 days
     - **File:** `utils/evaluation_llama3.py`
       - **Description:** Evaluation functions for the LLaMA3 model, including text generation, validation, and quality assessment.
     - **Checkpoint:** Model generates coherent text and validation loss is reasonable

---

## 8. **Scaling Up for Full Training**
   - **Task:** Gradually scale the model and dataset size while monitoring GPU memory.
   - **Details:**
     - Train with a larger dataset, but keep the batch size manageable to avoid memory overflow.
     - Track training speed and loss to ensure stability.
     - Use mixed precision training (`torch.cuda.amp`) to optimize memory and performance.
   - **Estimated Time:** 5-7 days depending on dataset size
     - **Checkpoint:** Larger dataset training without memory overflow

---

## 9. **Checkpoints & Model Saving**
   - **Task:** Implement a robust checkpointing system to save model weights and training states.
   - **Details:**
     - Save model weights periodically during training (every few epochs).
     - Implement a mechanism to resume training from the latest checkpoint.
   - **Estimated Time:** 1 hour
     - **Included in:** `train/llama3.train.py` and `utils/training_llama3.py`
     - **Checkpoint:** Checkpointing and saving system works as expected

---

## 10. **Fine-Tuning & Hyperparameter Tuning**
   - **Task:** Fine-tune the model and explore different hyperparameters to improve performance.
   - **Details:**
     - Experiment with learning rates, batch sizes, and number of layers/attention heads.
     - Apply techniques like dropout to regularize the model.
   - **Estimated Time:** 3-5 days for iterative tuning
     - **Checkpoint:** Optimal model configuration with improved performance

---

## 11. **Applying Pruning Techniques**
   - **Task:** After achieving satisfactory performance, apply pruning techniques to compress the model.
   - **Details:**
     - Test structured and unstructured pruning methods.
     - Monitor the trade-off between model size and performance.
   - **Estimated Time:** 5-7 days (can be longer depending on experimentation)
     - **Checkpoint:** Pruned model achieves satisfactory performance with reduced size

---

## 12. **Final Testing & Documentation**
   - **Task:** Test the final model on different datasets and document the results.
   - **Details:**
     - Evaluate generalization on unseen datasets.
     - Write a detailed report on model performance, pruning effectiveness, and lessons learned.
   - **Estimated Time:** 2-3 days
     - **Checkpoint:** Final model performance documented

---

# File Overview
- **`models/LLaMA3.py`**: Full implementation of the LLaMA3 model.
- **`models/transformers.py`**: Custom transformer layers used in LLaMA3, including self-attention, feed-forward networks, etc.
- **`train/llama3.train.py`**: Script to train the LLaMA3 model, initialize training configurations, load dataset, and save checkpoints.
- **`utils/llama_tokenizer.py`**: Tokenize and prepare the text dataset for training, with efficient dataloaders.
- **`utils/training_llama3.py`**: Helper functions for the LLaMA3 training loop, including optimizers, learning rate schedulers, and gradient updates.
- **`utils/evaluation_llama3.py`**: Functions for evaluating LLaMA3 model performance, including generating text based on prompt and computing validation metrics.
