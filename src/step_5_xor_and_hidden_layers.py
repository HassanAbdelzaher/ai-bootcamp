"""
Step 5 — Hidden Layers & XOR (Why Deep Learning Exists)
Goal: Prove why a single layer is not enough and how hidden layers solve complex problems.
Tools: Python + NumPy + Matplotlib
"""

import numpy as np
from plotting import plot_xor_data, plot_learning_curves_comparison, plot_decision_regions

# 5.2 The XOR Problem
print("=== 5.2 The XOR Problem ===")
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([[0], [1], [1], [0]])

print("XOR Truth Table:")
print("x1 | x2 | XOR")
print("----|----|----")
for i in range(len(X)):
    print(f"{int(X[i,0])}  | {int(X[i,1])}  | {int(y[i,0])}")
print()

# 5.3 Visualizing XOR (Why One Line Fails)
print("=== 5.3 Visualizing XOR ===")
plot_xor_data(X, y)

# 5.4 Try a Single-Layer Model (It Will Fail)
print("=== 5.4 Single-Layer Model (Will Fail) ===")
def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))

np.random.seed(42)
W = np.random.randn(2, 1)
b = np.zeros((1,))

lr = 0.1
losses_single = []

for epoch in range(2000):
    z = X @ W + b
    y_pred = sigmoid(z)
    
    loss = -np.mean(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))
    losses_single.append(loss)
    
    dW = X.T @ (y_pred - y)
    db = np.mean(y_pred - y)
    
    W -= lr * dW
    b -= lr * db

single_predictions = (y_pred >= 0.5).astype(int)
print("Single-layer predictions:", single_predictions.flatten())
print("Actual values:", y.flatten())
print("Accuracy:", np.mean(single_predictions == y))
print("❌ The model cannot solve XOR!")
print()

# 5.6 Initialize the Network (With Hidden Layer)
print("=== 5.6 Initialize Network with Hidden Layer ===")
np.random.seed(42)
W1 = np.random.randn(2, 4)  # input → hidden (4 neurons)
b1 = np.zeros((1, 4))

W2 = np.random.randn(4, 1)  # hidden → output
b2 = np.zeros((1, 1))

lr = 0.1
losses = []

print("Network architecture:")
print("  Input: 2 features")
print("  Hidden: 4 neurons")
print("  Output: 1 neuron")
print()

# 5.7 Forward Pass
print("=== 5.7 Forward Pass Function ===")
def forward(X):
    """Forward pass through the network"""
    Z1 = X @ W1 + b1
    A1 = sigmoid(Z1)
    
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)
    
    return A1, A2

# 5.8 Training with Backpropagation
print("=== 5.8 Training with Backpropagation ===")
for epoch in range(5000):
    A1, y_pred = forward(X)
    
    loss = -np.mean(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))
    losses.append(loss)
    
    # Backpropagation
    dZ2 = y_pred - y
    dW2 = A1.T @ dZ2
    db2 = np.mean(dZ2, axis=0, keepdims=True)
    
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * A1 * (1 - A1)  # sigmoid derivative
    dW1 = X.T @ dZ1
    db1 = np.mean(dZ1, axis=0, keepdims=True)
    
    # Update weights and biases
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

print("Training complete!")
print("Final loss:", losses[-1])
print()

# 5.9 Learning Curve
print("=== 5.9 Learning Curve ===")
plot_learning_curves_comparison(losses_single, losses)

# 5.10 Final Predictions (SUCCESS)
print("=== 5.10 Final Predictions (SUCCESS) ===")
_, final_probs = forward(X)
final_preds = (final_probs >= 0.5).astype(int)

print("Final probabilities:\n", final_probs)
print("Final predictions:\n", final_preds)
print("Actual values:\n", y)
print("Accuracy:", np.mean(final_preds == y))
print("✅ The network solves XOR!")
print()

# 5.11 What Just Happened?
print("=== 5.11 What Just Happened? ===")
print("- Hidden neurons learned intermediate patterns")
print("- Network combined them to solve XOR")
print("- This is the foundation of deep learning")
print()

# 5.12 Visualizing Decision Regions (Optional)
print("=== 5.12 Visualizing Decision Regions ===")
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

_, probs = forward(grid)
Z = probs.reshape(xx.shape)
plot_decision_regions(xx, yy, Z, X, y)

# Why Step 5 Is Critical
print("=== 5.13 Why Step 5 Is Critical ===")
print("✅ Explains why deep learning exists")
print("✅ Shows limits of shallow models")
print("✅ Makes hidden layers intuitive")
print("✅ Creates a lasting 'Aha!' moment")
print()

# Exercises
print("=== Exercises ===")
print("Exercise 1: Try different numbers of hidden neurons (2, 8)")
print("Exercise 2: Replace sigmoid with ReLU in hidden layer")
print("Exercise 3: Why does adding depth help but adding width sometimes doesn't?")
