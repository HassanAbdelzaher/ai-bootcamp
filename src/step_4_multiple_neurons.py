"""
Step 4 — Multiple Neurons & Neural Network Layer (Matrix Thinking)
Goal: Move from a single neuron to many neurons working together using matrices.
Tools: Python + NumPy + Matplotlib
"""

import numpy as np
from plotting import plot_neuron_outputs, plot_learning_curve, plot_confusion_matrix_style

# 4.3 Dataset Example (Student Features)
# We'll classify students as Pass/Fail based on Math and Science scores
print("=== 4.3 Dataset Example ===")

# X: Input features matrix
# Each row is one student, each column is one feature
# Shape: (4 students, 2 features)
# Features: [Math score, Science score]
X = np.array([
    [80, 70],  # Student 1: Math=80, Science=70
    [60, 65],  # Student 2: Math=60, Science=65
    [90, 95],  # Student 3: Math=90, Science=95
    [50, 45]   # Student 4: Math=50, Science=45
], dtype=float)  # dtype=float ensures decimal calculations work

# y: Target labels (what we want to predict)
# Each row is one student's label
# Shape: (4 students, 1 output)
# 1 = Pass, 0 = Fail
y = np.array([[1], [0], [1], [0]])  # Student 1 and 3 pass, 2 and 4 fail

print("Features (Math, Science):")
print(X)
print("Targets (Pass=1, Fail=0):")
print(y)
print()

# 4.4 Understanding Shapes
# Understanding matrix shapes is crucial for neural networks!
print("=== 4.4 Understanding Shapes ===")
# X.shape returns (number of rows, number of columns)
# (4, 2) means 4 samples and 2 features
print("X shape:", X.shape, "(samples, features)")
# y.shape returns (number of rows, number of columns)
# (4, 1) means 4 samples and 1 output
print("y shape:", y.shape, "(samples, outputs)")
print()

# 4.5 Weight Matrix (Many Neurons)
# Instead of one neuron, we'll use multiple neurons (3 in this case)
# Each neuron learns a different pattern!
print("=== 4.5 Weight Matrix (3 Neurons) ===")

# Set random seed for reproducibility (same random numbers each run)
np.random.seed(42)

# W: Weight matrix connecting inputs to hidden layer
# Shape: (2 features, 3 neurons)
# Each column represents weights for one neuron
# W[0, :] = weights from feature 0 (Math) to all 3 neurons
# W[1, :] = weights from feature 1 (Science) to all 3 neurons
# np.random.randn() creates random values from standard normal distribution
W = np.random.randn(2, 3)  # 2 features → 3 neurons

# b: Bias vector for hidden layer
# Shape: (1, 3) - one bias value per neuron
# np.zeros() creates array filled with zeros
b = np.zeros((1, 3))

print("W shape:", W.shape, "(features, neurons)")
print("b shape:", b.shape, "(1, neurons)")
print("W:\n", W)
print("b:", b)
print()

# 4.6 Forward Pass (Matrix Multiplication)
# This is where the magic happens - matrix multiplication processes all neurons at once!
print("=== 4.6 Forward Pass ===")

def sigmoid(z):
    """Sigmoid activation function - converts any number to 0-1 range"""
    # 1 / (1 + e^(-z)) - S-shaped curve
    # Returns values between 0 and 1 (probabilities)
    return 1 / (1 + np.exp(-z))

# Matrix multiplication: Z = X @ W + b
# X shape: (4, 2) - 4 samples, 2 features
# W shape: (2, 3) - 2 features, 3 neurons
# Result Z shape: (4, 3) - 4 samples, 3 neuron outputs
# 
# How it works:
# - For each sample (row in X), calculate dot product with each neuron (column in W)
# - This gives us 3 scores (one per neuron) for each of 4 samples
# - Then add bias (broadcasted to all samples)
Z = X @ W + b  # @ is matrix multiplication operator

# Apply sigmoid activation to get probabilities
# A shape: (4, 3) - 4 samples, 3 neuron outputs (each between 0 and 1)
A = sigmoid(Z)

print("Z shape:", Z.shape, "(samples, neurons)")
print("A (outputs) shape:", A.shape)
print("A (outputs):\n", A)
print("  Each column is one neuron's output for all samples")
print("  Each row is all neurons' outputs for one sample")
print()

# 4.7 Visualizing Neuron Outputs
# See how different neurons respond to different inputs
print("=== 4.7 Visualizing Neuron Outputs ===")
# plot_neuron_outputs creates a bar chart showing each neuron's output
plot_neuron_outputs(A)
print("  Different neurons learn to detect different patterns!")
print()

# 4.8 Single Output Neuron (Combining the Layer)
# Now we combine the 3 neuron outputs into a single prediction
print("=== 4.8 Single Output Neuron ===")

# W_out: Weight matrix from hidden layer to output
# Shape: (3 neurons, 1 output)
# Each row is the weight from one hidden neuron to the output
W_out = np.random.randn(3, 1)

# b_out: Bias for output neuron
# Shape: (1, 1) - single bias value
b_out = np.zeros((1, 1))

# Combine hidden layer outputs into single output
# A shape: (4, 3) - hidden layer activations
# W_out shape: (3, 1) - weights to output
# Result Z_out shape: (4, 1) - output scores for 4 samples
Z_out = A @ W_out + b_out

# Apply sigmoid to get final probability
# y_pred shape: (4, 1) - probabilities for each sample
y_pred = sigmoid(Z_out)

print("W_out shape:", W_out.shape)
print("Final output probabilities:\n", y_pred)
print("  Each value is the probability of Pass (1) for that student")
print()

# 4.9 Loss Function
# We need to measure how wrong our predictions are
print("=== 4.9 Loss Function ===")

def binary_cross_entropy(y, y_pred):
    """Binary Cross-Entropy Loss - measures prediction quality"""
    # epsilon: small value to avoid log(0) which would be -infinity
    epsilon = 1e-9
    
    # Loss formula: -mean(y * log(y_pred) + (1-y) * log(1-y_pred))
    # - If y=1: loss = -log(y_pred) - want y_pred close to 1
    # - If y=0: loss = -log(1-y_pred) - want y_pred close to 0
    # Lower loss = better predictions
    return -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))

# Calculate initial loss (before training)
loss = binary_cross_entropy(y, y_pred)
print("Initial loss:", loss)
print("  High loss = bad predictions, Low loss = good predictions")
print()

# 4.10 Training the Network (One Hidden Layer)
# Now we'll train the network to learn the correct weights
print("=== 4.10 Training the Network ===")

# Learning rate: controls how big steps we take when updating weights
lr = 0.1
# List to track loss over time (for visualization)
losses = []

# Reinitialize for training (start fresh)
np.random.seed(42)
W = np.random.randn(2, 3)      # Input → Hidden weights
b = np.zeros((1, 3))           # Hidden layer bias
W_out = np.random.randn(3, 1)  # Hidden → Output weights
b_out = np.zeros((1, 1))       # Output bias

# Training loop: repeat many times (epochs)
for epoch in range(1000):
    # ===== FORWARD PASS =====
    # Calculate hidden layer activations
    Z = X @ W + b        # Hidden layer scores
    A = sigmoid(Z)       # Hidden layer activations (probabilities)
    
    # Calculate output
    Z_out = A @ W_out + b_out  # Output scores
    y_pred = sigmoid(Z_out)     # Output probabilities
    
    # ===== CALCULATE LOSS =====
    loss = binary_cross_entropy(y, y_pred)
    losses.append(loss)  # Store for visualization
    
    # ===== BACKPROPAGATION =====
    # Calculate gradients (how to adjust weights to reduce loss)
    
    # Output layer gradients
    # dZ_out: Error at output layer (prediction - actual)
    dZ_out = y_pred - y
    # dW_out: Gradient for output weights
    # A.T @ dZ_out: Transpose of hidden activations times error
    dW_out = A.T @ dZ_out
    # db_out: Gradient for output bias (mean of errors)
    db_out = np.mean(dZ_out, axis=0, keepdims=True)
    
    # Hidden layer gradients (backpropagate error)
    # dA: Error flowing back through output weights
    dA = dZ_out @ W_out.T
    # dZ: Error at hidden layer (before activation)
    # A * (1 - A) is the derivative of sigmoid
    dZ = dA * A * (1 - A)  # sigmoid derivative
    # dW: Gradient for hidden weights
    # X.T @ dZ: Transpose of inputs times hidden error
    dW = X.T @ dZ
    # db: Gradient for hidden bias (mean of errors)
    db = np.mean(dZ, axis=0, keepdims=True)
    
    # ===== UPDATE WEIGHTS =====
    # Move weights in opposite direction of gradient (to reduce loss)
    W_out -= lr * dW_out  # Update output weights
    b_out -= lr * db_out  # Update output bias
    W -= lr * dW          # Update hidden weights
    b -= lr * db          # Update hidden bias

print("Training complete!")
print("Final loss:", losses[-1])
print()

# 4.11 Learning Curve
print("=== 4.11 Learning Curve ===")
plot_learning_curve(losses, title="Loss During Training", ylabel="Loss")

# 4.12 Final Predictions
print("=== 4.12 Final Predictions ===")

# Forward pass with trained weights (make predictions)
# Use the same forward pass as training, but with learned weights
Z = X @ W + b              # Hidden layer scores
A = sigmoid(Z)             # Hidden layer activations
Z_out = A @ W_out + b_out  # Output scores
final_probs = sigmoid(Z_out)  # Final probabilities (0 to 1)

# Convert probabilities to binary predictions
# (final_probs >= 0.5) creates boolean array: True if prob >= 0.5
# .astype(int) converts True/False to 1/0
predictions = (final_probs >= 0.5).astype(int)

print("Final probabilities:\n", final_probs)
print("  Values close to 1 = confident Pass, close to 0 = confident Fail")
print("Predictions:\n", predictions)
print("Actual values:\n", y)
# Calculate accuracy: mean of (predictions == y)
# (predictions == y) creates boolean array: True where predictions match
# np.mean() calculates percentage of correct predictions
print("Accuracy:", np.mean(predictions == y))
print()

# Visualize predictions
print("=== Predictions Visualization ===")
# .flatten() converts 2D arrays to 1D for the plotting function
# plot_confusion_matrix_style shows correct/incorrect predictions visually
plot_confusion_matrix_style(y.flatten(), predictions.flatten(), 
                          class_names=['Fail (0)', 'Pass (1)'])
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
