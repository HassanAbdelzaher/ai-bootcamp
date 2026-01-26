"""
Step 4 — Multiple Neurons & Neural Network Layer (Matrix Thinking)
Goal: Move from a single neuron to many neurons working together using matrices.
Tools: Python + NumPy + Matplotlib
"""

import numpy as np
from plotting import plot_neuron_outputs, plot_learning_curve

# 3.3 Dataset Example (Student Features)
print("=== 4.3 Dataset Example ===")
X = np.array([
    [80, 70],
    [60, 65],
    [90, 95],
    [50, 45]
], dtype=float)

y = np.array([[1], [0], [1], [0]])

print("Features (Math, Science):")
print(X)
print("Targets (Pass=1, Fail=0):")
print(y)
print()

# 4.4 Understanding Shapes
print("=== 4.4 Understanding Shapes ===")
print("X shape:", X.shape, "(samples, features)")
print("y shape:", y.shape, "(samples, outputs)")
print()

# 4.5 Weight Matrix (Many Neurons)
print("=== 4.5 Weight Matrix (3 Neurons) ===")
np.random.seed(42)  # for reproducibility
W = np.random.randn(2, 3)  # 2 features → 3 neurons
b = np.zeros((1, 3))

print("W shape:", W.shape, "(features, neurons)")
print("b shape:", b.shape, "(1, neurons)")
print("W:\n", W)
print("b:", b)
print()

# 4.6 Forward Pass (Matrix Multiplication)
print("=== 4.6 Forward Pass ===")
def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))

Z = X @ W + b
A = sigmoid(Z)

print("Z shape:", Z.shape, "(samples, neurons)")
print("A (outputs) shape:", A.shape)
print("A (outputs):\n", A)
print()

# 4.7 Visualizing Neuron Outputs
print("=== 4.7 Visualizing Neuron Outputs ===")
plot_neuron_outputs(A)

# 4.8 Single Output Neuron (Combining the Layer)
print("=== 4.8 Single Output Neuron ===")
W_out = np.random.randn(3, 1)
b_out = np.zeros((1, 1))

Z_out = A @ W_out + b_out
y_pred = sigmoid(Z_out)

print("W_out shape:", W_out.shape)
print("Final output probabilities:\n", y_pred)
print()

# 4.9 Loss Function
print("=== 4.9 Loss Function ===")
def binary_cross_entropy(y, y_pred):
    """Binary Cross-Entropy Loss"""
    epsilon = 1e-9
    return -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))

loss = binary_cross_entropy(y, y_pred)
print("Initial loss:", loss)
print()

# 4.10 Training the Network (One Hidden Layer)
print("=== 4.10 Training the Network ===")
lr = 0.1
losses = []

# Reinitialize for training
np.random.seed(42)
W = np.random.randn(2, 3)
b = np.zeros((1, 3))
W_out = np.random.randn(3, 1)
b_out = np.zeros((1, 1))

for epoch in range(1000):
    # Forward pass
    Z = X @ W + b
    A = sigmoid(Z)
    
    Z_out = A @ W_out + b_out
    y_pred = sigmoid(Z_out)
    
    # Loss
    loss = binary_cross_entropy(y, y_pred)
    losses.append(loss)
    
    # Backpropagation (simplified)
    dZ_out = y_pred - y
    dW_out = A.T @ dZ_out
    db_out = np.mean(dZ_out, axis=0, keepdims=True)
    
    dA = dZ_out @ W_out.T
    dZ = dA * A * (1 - A)  # sigmoid derivative
    dW = X.T @ dZ
    db = np.mean(dZ, axis=0, keepdims=True)
    
    # Update weights and biases
    W_out -= lr * dW_out
    b_out -= lr * db_out
    W -= lr * dW
    b -= lr * db

print("Training complete!")
print("Final loss:", losses[-1])
print()

# 4.11 Learning Curve
print("=== 4.11 Learning Curve ===")
plot_learning_curve(losses, title="Loss During Training", ylabel="Loss")

# 4.12 Final Predictions
print("=== 4.12 Final Predictions ===")
# Forward pass with trained weights
Z = X @ W + b
A = sigmoid(Z)
Z_out = A @ W_out + b_out
final_probs = sigmoid(Z_out)
predictions = (final_probs >= 0.5).astype(int)

print("Final probabilities:\n", final_probs)
print("Predictions:\n", predictions)
print("Actual values:\n", y)
print("Accuracy:", np.mean(predictions == y))
print()

# Why This Matters
print("=== 4.13 Why This Matters ===")
print("✅ First real neural network")
print("✅ Multiple neurons learn different patterns")
print("✅ Matrix math = speed + power")
print("❌ Still limited to simple patterns")
print()

# Exercises
print("=== Exercises ===")
print("Exercise 1: Try different numbers of neurons (2, 5)")
print("Exercise 2: Change learning rate and observe training")
print("Exercise 3: Why is matrix multiplication faster than loops?")
