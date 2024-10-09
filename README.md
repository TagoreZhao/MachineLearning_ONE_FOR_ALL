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
|   ├── MobileNetV1            # Implementation of Mobile Net Architecture
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
├── presentation.ipynb         # Keep recording my result model.
├── requirements.txt           # List of required Python packages (e.g., PyTorch, torchvision)
├── README.md                  # Documentation for setting up and using the project
├── test.ipynb                 # Jupyter notebook for testing models, visualizing results, etc.
├── resnet.train.py            # script to train resnet to 90% accuracy
└── .gitignore                 # Specify which files and directories to ignore in version control
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



