"""
Step 1 — Linear Regression (Learning to Predict)
Goal: Teach how AI learns from mistakes by adjusting weights automatically.
Tools: Python + NumPy + Matplotlib
"""

import numpy as np
from plotting import plot_data_scatter, plot_prediction_line, plot_learning_curve, plot_weight_evolution, plot_error_distribution, plot_train_test_split

# ============================================================================
# 1.3 Dataset Example
# ============================================================================
# We'll use a simple dataset: study hours vs exam scores
# This is a perfect linear relationship for demonstration
print("=== 1.3 Dataset Example ===")

# X: Input features (study hours) - what we use to make predictions
# dtype=float ensures we can do decimal calculations
X = np.array([1, 2, 3, 4], dtype=float)

# y: Target values (exam scores) - what we want to predict
# These are the "correct answers" we want our AI to learn
y = np.array([50, 60, 70, 80], dtype=float)

print("Study Hours:", X)
print("Exam Scores:", y)
print()
print("Observation: For every additional hour, score increases by 10 points")
print("  This is a perfect linear relationship: y = 10x + 40")
print()

# ============================================================================
# 1.4 Visualize the Data
# ============================================================================
# Always visualize your data first! This helps you understand the pattern.
print("=== 1.4 Visualize the Data ===")

# Create a scatter plot to see the relationship
# This shows us that more study hours = higher scores (positive correlation)
plot_data_scatter(X, y, xlabel="Study Hours", ylabel="Exam Score", 
                 title="Study Hours vs Exam Score")
print("  The graph shows a clear upward trend (positive correlation)")
print("  This tells us linear regression is appropriate for this data")
print()

# ============================================================================
# 1.5 First Bad Guess (No Learning Yet)
# ============================================================================
# Before training, the AI starts with random/zero values.
# This shows us how bad the predictions are initially.
print("=== 1.5 First Bad Guess ===")

# Initialize weights and bias to zero
# This means: y = 0*x + 0 = 0 (predicts zero for everything!)
w = 0.0  # Weight (slope of the line) - starts at zero
b = 0.0  # Bias (y-intercept) - starts at zero

# Make predictions with these bad values
# For each study hour, calculate: predicted_score = w * hours + b
# Since w=0 and b=0, all predictions are 0
y_pred = w * X + b  # This is vectorized: applies to all X values at once

print("Bad predictions:", y_pred)
print("  All predictions are 0 because w=0 and b=0")
print("  This is obviously wrong - we need to learn the correct values!")
print()

# Visualize how bad these predictions are
# The red line will be flat at y=0, far from the actual data points
plot_prediction_line(X, y, y_pred, xlabel="Study Hours", ylabel="Exam Score",
                    title="Initial Bad Prediction", label_pred="Bad Prediction", color="red")
print()

# ============================================================================
# 1.6 Error (How Wrong Are We?)
# ============================================================================
# We need to measure how wrong our predictions are.
# Mean Squared Error (MSE) is a common way to measure prediction error.
print("=== 1.6 Error Calculation ===")

# Calculate Mean Squared Error (MSE)
# MSE = average((prediction - actual)²)
# 
# Why squared?
# 1. Always positive (no cancellation of positive/negative errors)
# 2. Penalizes large errors more (2x error = 4x cost)
# 3. Smooth function (easier to optimize)

error = np.mean((y_pred - y) ** 2)

print("Mean Squared Error:", error)
print("  This is very high because all predictions are wrong")
print("  Our goal: Make this error as small as possible")
print()

# ============================================================================
# 1.8 Training the Model
# ============================================================================
# Now we'll train the model using gradient descent.
# The AI will learn from its mistakes and improve over time.
print("=== 1.8 Training the Model ===")

# Reset weights and bias to zero (start fresh)
w = 0.0
b = 0.0

# Learning rate: How big steps to take when updating weights
# Too high: Might overshoot the optimal value (unstable)
# Too low: Takes too long to learn (slow convergence)
# 0.01 is a good starting value for this problem
lr = 0.01  # learning rate

# Lists to store values during training (for visualization)
errors = []           # Track error over time
weights_history = []  # Track how weight changes
biases_history = []   # Track how bias changes

# Training loop: Repeat many times (epochs)
# Each epoch: Make predictions, calculate error, update weights
for epoch in range(1000):
    # ===== FORWARD PASS =====
    # Make predictions using current weights
    # y_pred = w * X + b (applied to all X values)
    y_pred = w * X + b
    
    # ===== CALCULATE GRADIENTS =====
    # Gradient tells us which direction to move to reduce error
    # 
    # Gradient for weight (w):
    #   dw = average((prediction - actual) × input)
    #   This tells us: if we increase w, how much will error change?
    dw = np.mean((y_pred - y) * X)
    
    # Gradient for bias (b):
    #   db = average(prediction - actual)
    #   This tells us: if we increase b, how much will error change?
    db = np.mean(y_pred - y)
    
    # ===== UPDATE WEIGHTS (GRADIENT DESCENT) =====
    # Move in the OPPOSITE direction of the gradient (to reduce error)
    # 
    # If gradient is positive → error increases when w increases
    #   → We should DECREASE w (subtract)
    # If gradient is negative → error decreases when w increases
    #   → We should INCREASE w (subtract negative = add)
    #
    # Learning rate controls step size:
    #   w = w - lr * dw
    #   Small lr = small steps (slow but stable)
    #   Large lr = large steps (fast but might overshoot)
    w -= lr * dw  # Update weight
    b -= lr * db  # Update bias
    
    # ===== STORE VALUES FOR VISUALIZATION =====
    weights_history.append(w)  # Save current weight
    biases_history.append(b)   # Save current bias
    
    # ===== CALCULATE AND STORE ERROR =====
    # Track error to see if we're improving
    error = np.mean((y_pred - y) ** 2)
    errors.append(error)

# After training, print final values
print("Final w (weight):", w)
print("Final b (bias):", b)
print("Final error:", errors[-1])
print()
print("  Weight (w) ≈ 10: Each hour adds ~10 points to score")
print("  Bias (b) ≈ 40: Starting score (when hours = 0)")
print("  Error ≈ 0: Perfect fit! (because data is perfectly linear)")
print()

# ============================================================================
# 1.9 Learning Curve
# ============================================================================
# Visualize how the error decreases during training.
# This shows us that the AI is learning!
print("=== 1.9 Learning Curve ===")

# Plot error over time (epochs)
# You should see error start high and decrease to near zero
plot_learning_curve(errors, title="Learning Curve", ylabel="Error (MSE)")
print("  The curve shows error decreasing over time")
print("  Steep drop at first, then gradual improvement")
print()

# Visualize how weights and bias change during training
print("=== Weight and Bias Evolution ===")
plot_weight_evolution(weights_history, biases_history, 
                     title="How Weights and Bias Change During Training")
print("  Weight increases from 0 to ~10")
print("  Bias increases from 0 to ~40")
print("  Both converge to optimal values")
print()

# ============================================================================
# 1.10 Final Prediction Line
# ============================================================================
# Now let's see how good our learned model is!
print("=== 1.10 Final Prediction Line ===")

# Make predictions with the learned weights
y_pred = w * X + b

# Visualize: Green line should pass through (or very close to) all data points
plot_prediction_line(X, y, y_pred, xlabel="Study Hours", ylabel="Exam Score",
                    title="Final Learned Model", label_pred="Learned Line", color="green")
print("  The green line should fit the data points perfectly!")
print()

# Analyze prediction errors
print("=== Error Analysis ===")
plot_error_distribution(y, y_pred, title="Prediction Error Distribution")
print("  Error histogram shows how predictions differ from actual values")
print("  Points close to diagonal line = good predictions")
print()

# ============================================================================
# 1.11 Train/Test Split (Important Practice)
# ============================================================================
# In real projects, we split data into training and testing sets.
# This helps us evaluate how well our model generalizes to new data.
print("=== 1.11 Train/Test Split Visualization ===")
print("Why split data?")
print("  - Training set: Used to learn the model")
print("  - Test set: Used to evaluate performance on unseen data")
print("  - Prevents overfitting (memorizing training data)")
print()

# Create a larger dataset for demonstration
X_full = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
y_full = np.array([50, 60, 70, 80, 90, 100, 110, 120], dtype=float)

# Split: 75% train, 25% test
split_idx = int(len(X_full) * 0.75)
X_train = X_full[:split_idx]
y_train = y_full[:split_idx]
X_test = X_full[split_idx:]
y_test = y_full[split_idx:]

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print()

# Train model on training set only
w_train = 0.0
b_train = 0.0
lr = 0.01

for epoch in range(1000):
    y_pred_train = w_train * X_train + b_train
    dw = np.mean((y_pred_train - y_train) * X_train)
    db = np.mean(y_pred_train - y_train)
    w_train -= lr * dw
    b_train -= lr * db

# Make predictions on both sets
y_pred_train_final = w_train * X_train + b_train
y_pred_test_final = w_train * X_test + b_test

# Calculate errors
train_error = np.mean((y_pred_train_final - y_train) ** 2)
test_error = np.mean((y_pred_test_final - y_test) ** 2)

print(f"Training MSE: {train_error:.2f}")
print(f"Test MSE: {test_error:.2f}")
print("  Good models have similar train and test errors")
print()

# Visualize train/test split
plot_train_test_split(X_train, y_train, X_test, y_test, 
                     y_pred_train_final, y_pred_test_final,
                     xlabel="Study Hours", ylabel="Exam Score",
                     title="Train/Test Split Visualization")
print("  Blue circles = Training data (used for learning)")
print("  Red squares = Test data (used for evaluation)")
print("  Dashed lines = Model predictions")
print()

# ============================================================================
# 1.12 Make Predictions
# ============================================================================
# Now we can use our trained model to predict scores for new students!
print("=== 1.12 Make Predictions ===")

# Predict for a new student who studied 5 hours
# Use the learned equation: score = w * hours + b
study_hours = 5
predicted_score = w * study_hours + b
print(f"Predicted score for {study_hours} hours: {predicted_score:.2f}")
print("  This is a prediction for data the model hasn't seen before!")
print()

# Test multiple predictions
test_hours = [6, 8]
print("Predictions for more students:")
for hours in test_hours:
    score = w * hours + b
    print(f"  Study {hours:2d} hours → Predicted score: {score:.2f}")
print()

# ============================================================================
# Key Concepts Learned
# ============================================================================
print("=== Key Concepts Learned ===")
print("✓ Linear Model: y = w·x + b")
print("✓ Mean Squared Error: Measures prediction quality")
print("✓ Gradient: Direction to move to reduce error")
print("✓ Gradient Descent: Learning algorithm that minimizes error")
print("✓ Learning Rate: Controls step size during learning")
print("✓ Training Loop: Iterative process of improving predictions")
print("✓ Making Predictions: Using learned model for new data")
print()

# ============================================================================
# Exercises
# ============================================================================
print("=== Exercises ===")
print("Exercise 1: Try different learning rates (0.1, 0.001)")
print("  - Higher lr: Faster learning but might overshoot")
print("  - Lower lr: Slower but more stable")
print()
print("Exercise 2: Add more data points")
print("  - Add: [5, 90] and [6, 100] to the dataset")
print("  - Retrain and see if predictions improve")
print()
print("Exercise 3: Predict scores for 6 and 8 hours")
print("  - Use the learned model: score = w * hours + b")
print("  - Compare with the pattern in the data")