"""
Step 2 — Perceptron (First Decision-Making AI)
Goal: Teach how AI makes YES / NO decisions using a simple artificial neuron.
Tools: Python + NumPy + Matplotlib
"""

import numpy as np
from plotting import plot_perceptron_boundary, plot_weight_evolution, plot_confusion_matrix_style

# ============================================================================
# 2.3 Dataset Example (Pass / Fail)
# ============================================================================
# We'll classify students as Pass (1) or Fail (0) based on study hours.
# This is a binary classification problem.
print("=== 2.3 Dataset Example ===")

# X: Input features (study hours)
# Each number represents how many hours a student studied
X = np.array([1, 2, 3, 4])

# y: Target labels (0 = Fail, 1 = Pass)
# Binary classification: only two possible outcomes
y = np.array([0, 0, 1, 1])

print("Study Hours:", X)
print("Pass (0=Fail, 1=Pass):", y)
print()
print("Pattern: Students with 3+ hours pass, others fail")
print("  This is linearly separable (can be separated by a straight line)")
print()

# ============================================================================
# 2.4 Step Function (Decision Maker)
# ============================================================================
# The step function converts a continuous score into a binary decision.
# This is what makes perceptron different from linear regression!
print("=== 2.4 Step Function ===")

def step_function(z):
    """
    Converts a continuous score into a binary decision.
    
    Parameters:
    z: calculated score (can be any number)
    
    Returns:
    1 if z >= 0 (YES/PASS)
    0 if z < 0  (NO/FAIL)
    """
    # Simple threshold: if score is non-negative, return 1, else 0
    return 1 if z >= 0 else 0

# Test the step function with various values
test_values = [-5, -1, 0, 1, 5]
print("Step function test:")
print("  Input → Output")
for val in test_values:
    result = step_function(val)
    decision = "YES/PASS" if result == 1 else "NO/FAIL"
    print(f"  {val:3d} → {result} ({decision})")
print()
print("  Key property: Hard threshold at z = 0")
print("  No middle ground - only 0 or 1 (unlike probabilities)")
print()

# ============================================================================
# 2.5 A Single Perceptron (No Learning Yet)
# ============================================================================
# Before training, we start with initial weight and bias values.
# These are usually random or zero, but we'll use specific values for demo.
print("=== 2.5 Single Perceptron (Initial) ===")

# Initial weight and bias (before learning)
# These values are chosen to demonstrate the concept
w = 1.0   # Weight: how important study hours are
b = -2.5  # Bias: threshold adjustment

print(f"Initial weight: {w}")
print(f"Initial bias: {b}")
print()

# Make predictions with initial values
predictions = []
for x in X:
    # Calculate score: z = w * x + b
    # Same math as linear regression!
    z = w * x + b
    
    # Apply step function to convert score to decision
    pred = step_function(z)
    predictions.append(pred)
    
    # Print details for understanding
    decision = "PASS" if pred == 1 else "FAIL"
    actual = "PASS" if y[x-1] == 1 else "FAIL"  # X is 1-indexed
    match = "✓" if pred == y[x-1] else "✗"
    print(f"  Hours: {x}, z = {w}×{x} + ({b}) = {z:.1f}, "
          f"Predicted: {decision}, Actual: {actual} {match}")

print("\nInitial predictions:", predictions)
print("Actual values:", y)
print()
print("  These initial values happen to work, but usually they don't!")
print("  We need to train the perceptron to learn the correct values")
print()

# ============================================================================
# 2.6 Visualizing the Decision Boundary
# ============================================================================
# The decision boundary is where the perceptron switches from one class to another.
# It's the point where z = 0, which means: w·x + b = 0
print("=== 2.6 Decision Boundary Visualization ===")

# Calculate decision boundary
# Solve: w·x + b = 0  →  x = -b / w
# This is where the perceptron switches from Fail to Pass
boundary = (-b / w) if w != 0 else 0

print(f"Decision boundary: x = {boundary:.2f}")
print(f"  Students with x < {boundary:.2f} → FAIL")
print(f"  Students with x ≥ {boundary:.2f} → PASS")
print()

# Visualize the decision boundary
# The graph shows data points and the boundary line
plot_perceptron_boundary(X, y, boundary, title="Perceptron Decision Boundary", 
                        color="red", label="Decision Boundary")
print()

# ============================================================================
# 2.8 Training Loop
# ============================================================================
# Now we'll train the perceptron using the perceptron learning rule.
# The perceptron updates weights only when it makes a mistake!
print("=== 2.8 Training the Perceptron ===")

# Reset to zero (start fresh)
w = 0.0
b = 0.0

# Learning rate: How much to adjust weights when wrong
lr = 0.1  # learning rate

# Store history for visualization
weights_history = []
biases_history = []

print("Training progress:")
print("  Epoch | Weight | Bias | Explanation")
print("  " + "-" * 50)

# Training loop: Multiple passes through the data
for epoch in range(10):
    # Process each data point
    for i in range(len(X)):
        # ===== FORWARD PASS =====
        # Calculate score using current weights
        z = w * X[i] + b
        
        # Make prediction using step function
        y_pred = step_function(z)
        
        # ===== CALCULATE ERROR =====
        # Error = actual - predicted
        #   If actual=1, predicted=0: error = +1 (should increase score)
        #   If actual=0, predicted=1: error = -1 (should decrease score)
        #   If correct: error = 0 (no update needed)
        error = y[i] - y_pred
        
        # ===== UPDATE WEIGHTS (PERCEPTRON LEARNING RULE) =====
        # Only update if prediction is wrong (error != 0)
        # 
        # Weight update: w = w + lr × error × input
        #   - If error > 0: increase weight (make it easier to predict 1)
        #   - If error < 0: decrease weight (make it harder to predict 1)
        #   - Input (X[i]) scales the update by feature value
        #
        # Bias update: b = b + lr × error
        #   - Similar to weight, but doesn't depend on input
        w += lr * error * X[i]
        b += lr * error
    
    # Store values after each epoch
    weights_history.append(w)
    biases_history.append(b)
    
    # Calculate accuracy for this epoch
    correct = sum(1 for j in range(len(X)) 
                  if step_function(w * X[j] + b) == y[j])
    accuracy = correct / len(X) * 100
    
    print(f"  {epoch:5d} | {w:6.2f} | {b:5.2f} | Accuracy: {accuracy:.0f}%")

print()
print("  Key observation: Perceptron stops updating when all predictions are correct!")
print("  This is called 'convergence'")
print()

# Visualize how weights change during training
print("=== Weight Evolution ===")
plot_weight_evolution(weights_history, biases_history)
print("  Weight and bias adjust until they find the correct decision boundary")
print()

# ============================================================================
# 2.9 Final Predictions
# ============================================================================
# After training, let's see how well the perceptron learned!
print("=== 2.9 Final Predictions ===")

# Make predictions with trained weights
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
print("  Green checkmarks = correct predictions")
print("  Red X marks = incorrect predictions")
print()

# ============================================================================
# 2.10 Visualize Final Decision Boundary
# ============================================================================
# Show the learned decision boundary
print("=== 2.10 Final Decision Boundary ===")

# Calculate final boundary
boundary = (-b / w) if w != 0 else 0

print(f"Learned decision boundary: x = {boundary:.2f}")
print(f"  This is where the perceptron learned to separate Pass from Fail")
print()

# Visualize the learned boundary (should be green to show it's learned)
plot_perceptron_boundary(X, y, boundary, title="Learned Perceptron Boundary",
                        color="green", label="Learned Boundary")
print()

# ============================================================================
# 2.11 Limitations
# ============================================================================
# The perceptron has important limitations!
print("=== 2.11 Limitations ===")
print("❌ Can only draw straight lines")
print("   - Decision boundary is always linear")
print("   - Cannot handle curved boundaries")
print()
print("❌ Cannot solve problems like XOR")
print("   - XOR requires a non-linear boundary")
print("   - No single line can separate XOR classes")
print()
print("❌ Hard YES/NO decisions only")
print("   - No probability or confidence")
print("   - Can't express uncertainty")
print()
print("  These limitations led to the development of:")
print("  - Multi-layer perceptrons (hidden layers)")
print("  - Neural networks (multiple layers)")
print("  - Logistic regression (probabilistic decisions)")
print()

# ============================================================================
# Key Concepts Learned
# ============================================================================
print("=== Key Concepts Learned ===")
print("✓ Perceptron: Makes binary decisions (0 or 1)")
print("✓ Step Function: Converts scores to decisions")
print("✓ Decision Boundary: Line that separates classes")
print("✓ Perceptron Learning Rule: Updates weights only when wrong")
print("✓ Convergence: Stops when all predictions are correct")
print("✓ Limitations: Only linear boundaries, no probabilities")
print()

# ============================================================================
# Exercises
# ============================================================================
print("=== Exercises ===")
print("Exercise 1: Try different learning rates (0.01, 1.0)")
print("  - Small lr: Slow but stable learning")
print("  - Large lr: Fast but might overshoot")
print()
print("Exercise 2: Change the bias start value")
print("  - Try b = -5, b = 0, b = 5")
print("  - Observe how it affects training")
print()
print("Exercise 3: Why can't one straight line solve XOR?")
print("  - Try to draw XOR data points")
print("  - Try to separate them with one line")
print("  - You'll see it's impossible!")