# Project 2: Multi-Class Classifier

> **Build a neural network that classifies data into multiple categories**

**Difficulty**: ⭐⭐ Intermediate  
**Time**: 2-3 hours  
**Prerequisites**: Steps 0-5 (All foundational steps, including hidden layers)

---

## 📚 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem: Handwritten Digit Recognition](#problem-handwritten-digit-recognition)
3. [Key Concepts](#key-concepts)
4. [Step-by-Step Guide](#step-by-step-guide)
5. [Code Structure](#code-structure)
6. [Expected Results](#expected-results)
7. [Extension Ideas](#extension-ideas)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

This project teaches you to build a **multi-class classifier** - a neural network that can distinguish between **more than 2 classes**. Unlike binary classification (spam/not spam), multi-class classification handles multiple categories (digits 0-9, animal types, etc.).

### Why Multi-Class Classification?

- **Real-world applications**: Most problems have multiple categories
- **Foundation for advanced models**: Understanding this prepares you for complex tasks
- **Neural network skills**: Learn to build multi-layer networks with softmax

---

## 📋 Problem: Handwritten Digit Recognition

### Task

Classify handwritten digits (0-9) based on pixel values. This is a **10-class classification problem**.

### Learning Objectives

- Build multi-layer neural networks
- Understand softmax activation
- Implement categorical cross-entropy loss
- Evaluate multi-class performance
- Interpret confusion matrices

### Dataset Description

The project uses synthetic 8×8 pixel digit images:

| Feature | Description | Range |
|---------|-------------|-------|
| **Pixels** | 64 pixel values (8×8 image) | 0.0 - 1.0 (normalized) |
| **Label** | Digit class | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |

**Pattern Examples:**
- **Digit 0**: Circular pattern (ring shape)
- **Digit 1**: Vertical line
- **Digit 2**: S-like pattern
- **Digit 3**: Curved pattern
- And so on...

### Architecture Overview

```
Input Layer:  64 neurons (one per pixel)
Hidden Layer: 16 neurons (learns patterns)
Output Layer: 10 neurons (one per digit class)
```

---

## 🧠 Key Concepts

### 1. Multi-Class vs Binary Classification

**Binary Classification** (Project 1):
- 2 classes: Spam (1) or Not Spam (0)
- Output: Single probability
- Activation: Sigmoid

**Multi-Class Classification** (This Project):
- 10 classes: Digits 0-9
- Output: 10 probabilities (one per class)
- Activation: Softmax

### 2. Softmax Activation

**Purpose**: Convert raw scores to probabilities that sum to 1.0

**Formula**:
```
softmax(z_i) = exp(z_i) / sum(exp(z_j)) for all j
```

**Example**:
```python
# Raw scores for digit 3
scores = [0.1, 0.2, 0.1, 5.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
# After softmax
probabilities = [0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
# Sum = 1.0, highest probability at index 3 (digit 3)
```

**Code**:
```python
def softmax(z):
    """Softmax activation function"""
    # Subtract max for numerical stability
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
```

### 3. Categorical Cross-Entropy Loss

**Purpose**: Measure how wrong predictions are for multi-class problems

**Formula**:
```
Loss = -sum(y_true * log(y_pred)) / N
```

**One-Hot Encoding**:
- Convert class labels to vectors
- Example: Digit 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

**Code**:
```python
def categorical_cross_entropy(y_true, y_pred):
    """Categorical cross-entropy loss"""
    epsilon = 1e-9  # Avoid log(0)
    return -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))
```

### 4. One-Hot Encoding

**Purpose**: Convert class labels to vectors for multi-class problems

**Example**:
```python
# Labels: [0, 1, 2, 3]
# One-hot encoded:
# [1, 0, 0, 0]  # Class 0
# [0, 1, 0, 0]  # Class 1
# [0, 0, 1, 0]  # Class 2
# [0, 0, 0, 1]  # Class 3
```

**Code**:
```python
def one_hot_encode(labels, num_classes):
    """Convert labels to one-hot encoding"""
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot
```

---

## 🚀 Step-by-Step Guide

### Step 1: Create Digit Dataset

```python
import numpy as np

def create_digit_pattern(digit, size=8):
    """Create a simple pattern representing a digit"""
    img = np.zeros((size, size))
    
    if digit == 0:
        # Circle pattern (ring)
        center = size // 2
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                if 2 <= dist <= 3:
                    img[i, j] = 1.0
    elif digit == 1:
        # Vertical line
        img[:, size//2] = 1.0
    # ... patterns for other digits
    
    return img.flatten()  # Flatten to 64 pixels

# Generate dataset
num_samples_per_class = 50
X = []
y = []

for digit in range(10):  # Digits 0-9
    for _ in range(num_samples_per_class):
        pattern = create_digit_pattern(digit)
        X.append(pattern)
        y.append(digit)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)
```

**Code Explanation:**
- `create_digit_pattern()`: Creates visual pattern for each digit
- `flatten()`: Converts 8×8 image to 64-element vector
- Each digit has 50 samples (500 total samples)

### Step 2: One-Hot Encode Labels

```python
def one_hot_encode(labels, num_classes=10):
    """Convert labels to one-hot encoding"""
    # Create zero matrix: (samples, num_classes)
    one_hot = np.zeros((len(labels), num_classes))
    
    # Set 1 at position of true class
    # one_hot[i, labels[i]] = 1 sets the correct class to 1
    one_hot[np.arange(len(labels)), labels] = 1
    
    return one_hot

y_one_hot = one_hot_encode(y, num_classes=10)
```

**Example**:
- Label: `3`
- One-hot: `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`

### Step 3: Build Neural Network

```python
def sigmoid(z):
    """Sigmoid activation for hidden layer"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def softmax(z):
    """Softmax activation for output layer"""
    # Subtract max for numerical stability
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Initialize weights
np.random.seed(42)
W1 = np.random.randn(64, 16) * 0.1  # Input → Hidden
b1 = np.zeros((1, 16))
W2 = np.random.randn(16, 10) * 0.1   # Hidden → Output
b2 = np.zeros((1, 10))
```

**Architecture**:
- **Input**: 64 features (pixels)
- **Hidden**: 16 neurons (learns patterns)
- **Output**: 10 neurons (one per digit)

### Step 4: Forward Pass

```python
def forward(X, W1, b1, W2, b2):
    """Forward pass through network"""
    # Layer 1: Input → Hidden
    Z1 = X @ W1 + b1        # (samples, 64) @ (64, 16) = (samples, 16)
    A1 = sigmoid(Z1)        # Hidden layer activations
    
    # Layer 2: Hidden → Output
    Z2 = A1 @ W2 + b2       # (samples, 16) @ (16, 10) = (samples, 10)
    A2 = softmax(Z2)        # Output probabilities (sum to 1 per sample)
    
    return A1, A2
```

**Output Shape**:
- `A2`: (samples, 10) - Each row sums to 1.0 (probabilities)

### Step 5: Calculate Loss

```python
def categorical_cross_entropy(y_true, y_pred):
    """Categorical cross-entropy loss"""
    epsilon = 1e-9  # Small value to avoid log(0)
    # y_true: (samples, 10) - one-hot encoded
    # y_pred: (samples, 10) - probabilities
    # np.sum(..., axis=1): Sum over classes for each sample
    # np.mean(): Average over all samples
    return -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))

# Calculate loss
_, y_pred = forward(X, W1, b1, W2, b2)
loss = categorical_cross_entropy(y_one_hot, y_pred)
print(f"Initial loss: {loss:.4f}")
```

### Step 6: Backpropagation

```python
def backward(X, y_one_hot, A1, A2, W2):
    """Calculate gradients using backpropagation"""
    samples = X.shape[0]
    
    # Output layer gradients
    dZ2 = A2 - y_one_hot  # Error at output layer
    dW2 = (A1.T @ dZ2) / samples  # Gradient for W2
    db2 = np.mean(dZ2, axis=0, keepdims=True)  # Gradient for b2
    
    # Hidden layer gradients
    dA1 = dZ2 @ W2.T  # Error flowing back
    dZ1 = dA1 * A1 * (1 - A1)  # Sigmoid derivative
    dW1 = (X.T @ dZ1) / samples  # Gradient for W1
    db1 = np.mean(dZ1, axis=0, keepdims=True)  # Gradient for b1
    
    return dW1, db1, dW2, db2
```

**Gradient Flow**:
1. Calculate output error: `dZ2 = predictions - actual`
2. Backpropagate to hidden: `dZ1 = dA1 * sigmoid_derivative`
3. Calculate weight gradients: `dW = input.T @ error`

### Step 7: Training Loop

```python
lr = 0.1
epochs = 1000
losses = []

for epoch in range(epochs):
    # Forward pass
    A1, A2 = forward(X, W1, b1, W2, b2)
    
    # Calculate loss
    loss = categorical_cross_entropy(y_one_hot, A2)
    losses.append(loss)
    
    # Backward pass
    dW1, db1, dW2, db2 = backward(X, y_one_hot, A1, A2, W2)
    
    # Update weights
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
```

### Step 8: Make Predictions

```python
# Forward pass with trained weights
_, probabilities = forward(X, W1, b1, W2, b2)

# Get predicted class (highest probability)
predictions = np.argmax(probabilities, axis=1)

# Calculate accuracy
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.2%}")
```

**Prediction Process**:
1. Get probabilities: `[0.01, 0.02, 0.01, 0.95, 0.01, ...]`
2. Find index with highest probability: `np.argmax()` → 3
3. Predicted class: 3

### Step 9: Evaluate Performance

```python
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Create confusion matrix
cm = confusion_matrix(y, predictions)

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Calculate per-class accuracy
for digit in range(10):
    mask = y == digit
    class_accuracy = np.mean(predictions[mask] == digit)
    print(f"Digit {digit}: {class_accuracy:.2%} accuracy")
```

---

## 📊 Expected Results

### Training Output

```
Step 1: Creating Digit Dataset
Generated 500 samples (50 per digit)
Image shape: (500, 64)

Step 2: Training Neural Network
Epoch 100/1000, Loss: 0.8234
Epoch 200/1000, Loss: 0.4521
Epoch 300/1000, Loss: 0.2345
...
Epoch 1000/1000, Loss: 0.0123

Step 3: Evaluation
Overall Accuracy: 94.2%

Per-Class Accuracy:
Digit 0: 96.0% accuracy
Digit 1: 98.0% accuracy
Digit 2: 92.0% accuracy
...
```

### Confusion Matrix

The confusion matrix shows:
- **Diagonal**: Correct predictions (should be high)
- **Off-diagonal**: Misclassifications (should be low)

**Example**:
```
        Predicted
        0  1  2  3  4  5  6  7  8  9
Actual 0 [48  0  1  0  0  0  1  0  0  0]
       1 [ 0 49  0  0  0  0  0  1  0  0]
       2 [ 1  0 46  1  0  1  0  0  1  0]
       ...
```

---

## 💡 Extension Ideas

### Beginner Extensions

1. **Try Different Architectures**
   - More hidden neurons (8, 16, 32)
   - More hidden layers (2, 3 layers)
   - Compare performance

2. **Experiment with Learning Rates**
   - Try: 0.01, 0.1, 0.5, 1.0
   - Observe training speed and stability

3. **Visualize Learned Features**
   - Plot hidden layer weights
   - See what patterns the network learned

### Intermediate Extensions

4. **Add Regularization**
   - Implement dropout
   - Add L2 regularization
   - Compare with/without regularization

5. **Train/Test Split**
   - Split data: 80% train, 20% test
   - Evaluate on test set
   - Check for overfitting

6. **Different Activation Functions**
   - Try ReLU in hidden layer
   - Compare sigmoid vs ReLU
   - Observe training differences

### Advanced Extensions

7. **Use Real Dataset**
   - Load MNIST dataset (real handwritten digits)
   - Compare with synthetic data
   - Handle larger images (28×28)

8. **Implement Batch Training**
   - Train on batches instead of full dataset
   - Compare batch vs full batch training
   - Experiment with batch sizes

9. **Hyperparameter Tuning**
   - Grid search for best learning rate
   - Find optimal number of hidden neurons
   - Optimize architecture

---

## 🔧 Troubleshooting

### Common Issues

**Issue 1: All predictions are the same class**
- **Solution**: Check if weights are updating
- **Solution**: Verify softmax is working correctly
- **Solution**: Ensure one-hot encoding is correct

**Issue 2: Loss doesn't decrease**
- **Solution**: Try different learning rate
- **Solution**: Check gradient calculations
- **Solution**: Verify data is normalized

**Issue 3: Numerical instability (NaN values)**
- **Solution**: Add epsilon to log calculations
- **Solution**: Clip values in softmax
- **Solution**: Use numerical stability trick in softmax

**Issue 4: Low accuracy (<50%)**
- **Solution**: Check if dataset patterns are distinguishable
- **Solution**: Increase number of hidden neurons
- **Solution**: Train for more epochs

### Debugging Tips

1. **Print intermediate values**
   ```python
   print(f"Probabilities shape: {probabilities.shape}")
   print(f"Probabilities sum: {probabilities.sum(axis=1)}")  # Should be ~1.0
   print(f"Predictions: {predictions[:10]}")
   ```

2. **Visualize data**
   ```python
   # Plot first digit
   plt.imshow(X[0].reshape(8, 8), cmap='gray')
   plt.title(f'Digit: {y[0]}')
   plt.show()
   ```

3. **Check gradients**
   ```python
   print(f"Gradient magnitudes: W1={np.abs(dW1).mean():.6f}, W2={np.abs(dW2).mean():.6f}")
   ```

---

## 📚 Key Concepts Summary

### Multi-Class Classification
- **Problem**: Classify into multiple categories (not just 2)
- **Output**: Probabilities for each class (sum to 1.0)
- **Loss**: Categorical Cross-Entropy
- **Activation**: Softmax (output layer)

### Neural Network Architecture
- **Input Layer**: Number of features
- **Hidden Layer(s)**: Learn intermediate patterns
- **Output Layer**: Number of classes
- **Activation**: Sigmoid/ReLU (hidden), Softmax (output)

### Evaluation Metrics
- **Accuracy**: Overall correctness
- **Confusion Matrix**: Per-class performance
- **Per-Class Accuracy**: Performance for each class

---

## ✅ Success Criteria

- ✅ Model achieves >85% accuracy
- ✅ All classes are reasonably predicted
- ✅ Training converges smoothly
- ✅ Confusion matrix shows good performance
- ✅ Code is well-organized and documented

---

## 🎓 Learning Outcomes

By completing this project, you will:

- ✅ Understand multi-class classification
- ✅ Implement softmax activation
- ✅ Use categorical cross-entropy loss
- ✅ Build multi-layer neural networks
- ✅ Evaluate multi-class performance
- ✅ Interpret confusion matrices
- ✅ Debug neural network training

---

## 📖 Additional Resources

- **Step 4 Documentation**: `docs/Step_4_Multiple_Neurons.md`
- **Step 5 Documentation**: `docs/Step_5_XOR_and_Hidden_Layers.md`
- **Neural Network Basics**: Review Steps 4-5

---

**Ready to classify multiple categories? Let's build it!** 🚀

**Next Steps**: After completing this project, move on to **Project 3: Text Analyzer** to learn about RNNs and text processing.
