"""
Step 1 — Linear Regression (Learning to Predict)
Goal: Teach how AI learns from mistakes by adjusting weights automatically.
Tools: Python + NumPy + Matplotlib
"""

import numpy as np
from plotting import plot_data_scatter, plot_prediction_line, plot_learning_curve, plot_weight_evolution, plot_error_distribution

# 1.3 Dataset Example
print("=== 1.3 Dataset Example ===")
X = np.array([1, 2, 3, 4], dtype=float)
y = np.array([50, 60, 70, 80], dtype=float)
print("Study Hours:", X)
print("Exam Scores:", y)
print()

# 1.4 Visualize the Data
print("=== 1.4 Visualize the Data ===")
plot_data_scatter(X, y, xlabel="Study Hours", ylabel="Exam Score", 
                 title="Study Hours vs Exam Score")

# 1.5 First Bad Guess (No Learning Yet)
print("=== 1.5 First Bad Guess ===")
w = 0.0
b = 0.0

y_pred = w * X + b
print("Bad predictions:", y_pred)

plot_prediction_line(X, y, y_pred, xlabel="Study Hours", ylabel="Exam Score",
                    title="Initial Bad Prediction", label_pred="Bad Prediction", color="red")
print()

# 1.6 Error (How Wrong Are We?)
print("=== 1.6 Error Calculation ===")
error = np.mean((y_pred - y) ** 2)
print("Mean Squared Error:", error)
print()

# 1.8 Training the Model
print("=== 1.8 Training the Model ===")
w = 0.0
b = 0.0
lr = 0.01  # learning rate

errors = []
weights_history = []
biases_history = []

for epoch in range(1000):
    y_pred = w * X + b
    
    # Calculate gradients
    dw = np.mean((y_pred - y) * X)
    db = np.mean(y_pred - y)
    
    # Update weights and bias
    w -= lr * dw
    b -= lr * db
    
    # Store for visualization
    weights_history.append(w)
    biases_history.append(b)
    
    # Calculate and store error
    error = np.mean((y_pred - y) ** 2)
    errors.append(error)

print("Final w (weight):", w)
print("Final b (bias):", b)
print("Final error:", errors[-1])
print()

# 1.9 Learning Curve
print("=== 1.9 Learning Curve ===")
plot_learning_curve(errors, title="Learning Curve", ylabel="Error (MSE)")

# Visualize weight and bias evolution
print("=== Weight and Bias Evolution ===")
plot_weight_evolution(weights_history, biases_history, 
                     title="How Weights and Bias Change During Training")

# 1.10 Final Prediction Line
print("=== 1.10 Final Prediction Line ===")
y_pred = w * X + b

plot_prediction_line(X, y, y_pred, xlabel="Study Hours", ylabel="Exam Score",
                    title="Final Learned Model", label_pred="Learned Line", color="green")

# Error distribution
print("=== Error Analysis ===")
plot_error_distribution(y, y_pred, title="Prediction Error Distribution")
print()

# 1.11 Make Predictions
print("=== 1.11 Make Predictions ===")
study_hours = 5
predicted_score = w * study_hours + b
print(f"Predicted score for {study_hours} hours: {predicted_score:.2f}")

# Test multiple predictions
test_hours = [6, 8]
for hours in test_hours:
    score = w * hours + b
    print(f"Predicted score for {hours} hours: {score:.2f}")
print()

# Exercises
print("=== Exercises ===")
print("Exercise 1: Try different learning rates (0.1, 0.001)")
print("Exercise 2: Add more data points")
print("Exercise 3: Predict scores for 6 and 8 hours")
