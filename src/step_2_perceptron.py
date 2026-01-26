"""
Step 2 — Perceptron (First Decision-Making AI)
Goal: Teach how AI makes YES / NO decisions using a simple artificial neuron.
Tools: Python + NumPy + Matplotlib
"""

import numpy as np
from plotting import plot_perceptron_boundary, plot_weight_evolution, plot_confusion_matrix_style

# 2.3 Dataset Example (Pass / Fail)
print("=== 2.3 Dataset Example ===")
X = np.array([1, 2, 3, 4])
y = np.array([0, 0, 1, 1])
print("Study Hours:", X)
print("Pass (0=Fail, 1=Pass):", y)
print()

# 2.4 Step Function (Decision Maker)
print("=== 2.4 Step Function ===")
def step_function(z):
    """Turns a number into a decision (0 or 1)"""
    return 1 if z >= 0 else 0

# Test step function
test_values = [-5, -1, 0, 1, 5]
print("Step function test:")
for val in test_values:
    print(f"  step({val}) = {step_function(val)}")
print()

# 2.5 A Single Perceptron (No Learning Yet)
print("=== 2.5 Single Perceptron (Initial) ===")
w = 1.0
b = -2.5

predictions = []
for x in X:
    z = w * x + b
    predictions.append(step_function(z))

print("Initial predictions:", predictions)
print("Actual values:", y)
print()

# 2.6 Visualizing the Decision Boundary
print("=== 2.6 Decision Boundary Visualization ===")
boundary = (-b / w) if w != 0 else 0
plot_perceptron_boundary(X, y, boundary, title="Perceptron Decision Boundary", 
                        color="red", label="Decision Boundary")

# 2.8 Training Loop
print("=== 2.8 Training the Perceptron ===")
w = 0.0
b = 0.0
lr = 0.1  # learning rate

weights_history = []
biases_history = []

print("Training progress:")
for epoch in range(10):
    for i in range(len(X)):
        z = w * X[i] + b
        y_pred = step_function(z)
        
        error = y[i] - y_pred
        
        w += lr * error * X[i]
        b += lr * error
    
    weights_history.append(w)
    biases_history.append(b)
    print(f"Epoch {epoch}: w={w:.2f}, b={b:.2f}")
print()

# Visualize weight evolution
print("=== Weight Evolution ===")
plot_weight_evolution(weights_history, biases_history)

# 2.9 Final Predictions
print("=== 2.9 Final Predictions ===")
final_preds = []
for x in X:
    z = w * x + b
    final_preds.append(step_function(z))

print("Final predictions:", final_preds)
print("Actual values:", y)
print("Accuracy:", np.mean(np.array(final_preds) == y))
print()

# Visualize predictions vs actual
print("=== Predictions Visualization ===")
plot_confusion_matrix_style(y, np.array(final_preds), class_names=['Fail (0)', 'Pass (1)'])
print()

# 2.10 Visualize Final Decision Boundary
print("=== 2.10 Final Decision Boundary ===")
boundary = (-b / w) if w != 0 else 0
plot_perceptron_boundary(X, y, boundary, title="Learned Perceptron Boundary",
                        color="green", label="Learned Boundary")

# Limitations
print("=== 2.11 Limitations ===")
print("❌ Can only draw straight lines")
print("❌ Cannot solve problems like XOR")
print("❌ Hard YES/NO decisions only")
print()

# Exercises
print("=== Exercises ===")
print("Exercise 1: Try different learning rates (0.01, 1.0)")
print("Exercise 2: Change the bias start value")
print("Exercise 3: Why can't one straight line solve XOR?")
