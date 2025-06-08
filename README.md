
# MNIST Digit Classifier — PyTorch Implementation

This project implements a fully connected neural network using PyTorch to classify handwritten digits from the MNIST dataset. The model achieves over 95% accuracy on the test set through careful architecture design and the use of adaptive optimization techniques.

## Overview

- Model architecture: 784 → 64 → 64 → 10
- Activation: ReLU in hidden layers
- Final layer uses raw logits with `CrossEntropyLoss`, which internally applies softmax
- Optimizer: Adam with learning rate 0.0005 and L2 regularization (`weight_decay=1e-5`)
- Dataset: MNIST, CSV format with 10,000 examples (normalized to [0, 1])
- Regularization: Dropout with 0.3 probability between layers
- Achieved Test Accuracy: **95.15%**
- Training loss tracked and plotted using Matplotlib

## Why Adam Optimizer?

Adam (Adaptive Moment Estimation) was chosen over standard SGD due to its:

- Adaptive learning rate for each parameter
- Built-in momentum (uses first and second moment estimates)
- Faster and more stable convergence, especially with default parameters
- Reduced need for manual tuning, making it ideal for compact networks like this one

## Model Architecture

```
Input Layer      : 784 (flattened 28×28 grayscale image)
Hidden Layer 1   : 64 neurons → ReLU → Dropout(0.3)
Hidden Layer 2   : 64 neurons → ReLU → Dropout(0.3)
Output Layer     : 10 neurons (one for each digit class)
```

## Files

- `mnist_train.py` — PyTorch training script
- `mnist.csv` — Dataset file (first column: label, rest 784 columns: pixels)
- `loss.png` — Line plot of loss values across epochs
- `requirements.txt` — List of dependencies

## Training Loss Curve

Include `loss.png` here in your GitHub repo to show the learning trajectory.

## How to Run

1. Install required packages:
   ```bash
   pip install torch pandas matplotlib scikit-learn
   ```

2. Place `mnist.csv` in the same directory as the script.

3. Run the training script:
   ```bash
   python mnist_train.py
   ```

## Key Learnings

- Implementation of neural networks using PyTorch modules
- Understanding and applying dropout and L2 regularization
- Use of Adam optimizer to automate and stabilize training
- Working with DataLoader, TensorDataset, and model evaluation flow
- Visualizing training behavior using loss curves

## Potential Extensions

- Add validation loss and accuracy tracking
- Try alternate architectures like CNNs
- Experiment with learning rate schedulers
- Save and load model checkpoints using `torch.save`

## Author

This project was completed as part of a self-driven deep learning curriculum to transition from mathematical foundations to practical ML engineering.
