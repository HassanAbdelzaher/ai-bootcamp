"""
Project 2: Handwritten Digit Classifier
Multi-class classification using neural networks

This project applies concepts from Steps 4-5:
- Multi-layer neural networks
- Multi-class classification
- Softmax activation
- Categorical cross-entropy loss
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from plotting import plot_learning_curve

print("=" * 70)
print("Project 2: Handwritten Digit Classifier")
print("=" * 70)
print()

# ============================================================================
# Step 1: Create Synthetic Digit Dataset
# ============================================================================
print("Step 1: Creating Digit Dataset")
print("-" * 70)

def create_digit_pattern(digit, size=8):
    """Create a simple pattern representing a digit"""
    img = np.zeros((size, size))
    
    if digit == 0:
        # Circle pattern
        center = size // 2
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                if 2 <= dist <= 3:
                    img[i, j] = 1.0
    elif digit == 1:
        # Vertical line
        img[:, size//2] = 1.0
    elif digit == 2:
        # S-like pattern
        img[0, :] = 1.0
        img[size//2, :] = 1.0
        img[-1, :] = 1.0
        img[:size//2, -1] = 1.0
        img[size//2:, 0] = 1.0
    else:
        # Random pattern for other digits
        np.random.seed(digit)
        img = np.random.rand(size, size) > 0.7
        img = img.astype(float)
    
    return img

# Generate dataset
num_samples_per_digit = 20
num_classes = 10
img_size = 8

X = []
y = []

for digit in range(num_classes):
    for _ in range(num_samples_per_digit):
        # Create digit pattern
        img = create_digit_pattern(digit, img_size)
        
        # Add some noise
        noise = np.random.normal(0, 0.1, (img_size, img_size))
        img = np.clip(img + noise, 0, 1)
        
        # Flatten to 1D
        X.append(img.flatten())
        y.append(digit)

X = np.array(X)
y = np.array(y)

print(f"Created {len(X)} digit images")
print(f"Image size: {img_size}×{img_size} = {img_size*img_size} pixels")
print(f"Number of classes: {num_classes} (digits 0-9)")
print(f"Shape: X={X.shape}, y={y.shape}")
print()

# ============================================================================
# Step 2: One-Hot Encode Labels
# ============================================================================
print("Step 2: Encoding Labels")
print("-" * 70)

def one_hot_encode(labels, num_classes):
    """Convert class labels to one-hot encoding"""
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

y_one_hot = one_hot_encode(y, num_classes)

print("Example one-hot encoding:")
print(f"  Digit 0: {y_one_hot[0]}")
print(f"  Digit 5: {y_one_hot[num_samples_per_digit * 5]}")
print()

# ============================================================================
# Step 3: Build Neural Network
# ============================================================================
print("Step 3: Building Neural Network")
print("-" * 70)

def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def softmax(z):
    """Softmax activation for multi-class output"""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def categorical_cross_entropy(y_true, y_pred):
    """Categorical cross-entropy loss"""
    epsilon = 1e-9
    return -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))

# Network architecture
input_size = img_size * img_size  # 64
hidden_size = 16
output_size = num_classes  # 10

# Initialize weights
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.1
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.1
b2 = np.zeros((1, output_size))

print("Network Architecture:")
print(f"  Input layer: {input_size} neurons (pixels)")
print(f"  Hidden layer: {hidden_size} neurons")
print(f"  Output layer: {output_size} neurons (classes)")
print(f"  Total parameters: {(input_size * hidden_size + hidden_size) + (hidden_size * output_size + output_size):,}")
print()

# ============================================================================
# Step 4: Training
# ============================================================================
print("Step 4: Training Network")
print("-" * 70)

learning_rate = 0.1
epochs = 2000
losses = []

print("Training...")
for epoch in range(epochs):
    # Forward pass
    Z1 = X @ W1 + b1
    A1 = sigmoid(Z1)
    
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)  # Output probabilities
    
    # Calculate loss
    loss = categorical_cross_entropy(y_one_hot, A2)
    losses.append(loss)
    
    # Backward pass (simplified backpropagation)
    dZ2 = A2 - y_one_hot
    dW2 = A1.T @ dZ2
    db2 = np.mean(dZ2, axis=0, keepdims=True)
    
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * A1 * (1 - A1)
    dW1 = X.T @ dZ1
    db1 = np.mean(dZ1, axis=0, keepdims=True)
    
    # Update weights
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    
    if (epoch + 1) % 400 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

print("Training complete!")
print()

# ============================================================================
# Step 5: Evaluate Model
# ============================================================================
print("Step 5: Model Evaluation")
print("-" * 70)

# Make predictions
Z1 = X @ W1 + b1
A1 = sigmoid(Z1)
Z2 = A1 @ W2 + b2
predictions_probs = softmax(Z2)
predictions = np.argmax(predictions_probs, axis=1)

# Calculate accuracy
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy*100:.1f}%")
print()

# Per-class accuracy
print("Per-Class Accuracy:")
print("-" * 30)
for digit in range(num_classes):
    mask = y == digit
    if np.sum(mask) > 0:
        class_accuracy = np.mean(predictions[mask] == y[mask])
        print(f"  Digit {digit}: {class_accuracy*100:.1f}%")
print()

# ============================================================================
# Step 6: Visualize Results
# ============================================================================
print("Step 6: Visualizing Results")
print("-" * 70)

# Learning curve
plot_learning_curve(losses,
                   title="Digit Classifier - Learning Curve",
                   ylabel="Loss (Categorical Cross-Entropy)")

# Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y, predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=range(num_classes),
           yticklabels=range(num_classes))
plt.xlabel("Predicted Digit", fontsize=12, fontweight='bold')
plt.ylabel("Actual Digit", fontsize=12, fontweight='bold')
plt.title("Confusion Matrix - Digit Classification", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Show sample predictions
print("Sample Predictions:")
print("-" * 70)
num_samples = 10
for i in range(num_samples):
    idx = i * num_samples_per_digit
    actual = y[idx]
    pred = predictions[idx]
    confidence = predictions_probs[idx, pred]
    match = "✓" if actual == pred else "✗"
    
    print(f"Image {i+1}: Actual={actual}, Predicted={pred}, "
          f"Confidence={confidence:.1%} {match}")

print()
print("=" * 70)
print("Project 2 Complete!")
print("You've built a multi-class digit classifier!")
print("=" * 70)
