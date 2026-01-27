"""
Step 3 — Logistic Regression (Smart Decisions with Probability)
Goal: Upgrade the perceptron from hard YES/NO decisions to probabilistic decisions.
Tools: Python + NumPy + Matplotlib
"""

import numpy as np
from plotting import plot_sigmoid_function, plot_learning_curve, plot_probability_curve, plot_roc_curve

# 3.3 Sigmoid Function (Probability Maker)
print("=== 3.3 Sigmoid Function ===")
def sigmoid(z):
    """Converts any number into a value between 0 and 1"""
    return 1 / (1 + np.exp(-z))

# Visualize Sigmoid
z_vals = np.linspace(-10, 10, 200)
plot_sigmoid_function(z_vals)

# Test sigmoid
print("Sigmoid test values:")
test_z = [-5, -2, 0, 2, 5]
for z in test_z:
    print(f"  sigmoid({z}) = {sigmoid(z):.4f}")
print()

# 3.4 Dataset Example (Pass Probability)
print("=== 3.4 Dataset Example ===")
X = np.array([1, 2, 3, 4], dtype=float)
y = np.array([0, 0, 1, 1], dtype=float)
print("Study Hours:", X)
print("Pass (0/1):", y)
print()

# 3.5 Forward Pass (Prediction)
print("=== 3.5 Forward Pass (Initial) ===")
w = 0.0
b = 0.0

z = w * X + b
y_pred = sigmoid(z)
print("Initial probabilities:", y_pred)
print()

# 3.6 Loss Function (Binary Cross-Entropy)
print("=== 3.6 Loss Function ===")
def binary_cross_entropy(y, y_pred):
    """Binary Cross-Entropy Loss"""
    epsilon = 1e-9  # small value to avoid log(0)
    return -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))

loss = binary_cross_entropy(y, y_pred)
print("Initial loss:", loss)
print()

# 3.7 Gradient Descent (Learning Probabilities)
print("=== 3.7 Training Logistic Regression ===")
lr = 0.1
w = 0.0
b = 0.0
losses = []

for epoch in range(1000):
    z = w * X + b
    y_pred = sigmoid(z)
    
    # Calculate gradients
    dw = np.mean((y_pred - y) * X)
    db = np.mean(y_pred - y)
    
    # Update weights and bias
    w -= lr * dw
    b -= lr * db
    
    # Calculate and store loss
    loss = binary_cross_entropy(y, y_pred)
    losses.append(loss)

print("Final weights:", w)
print("Final bias:", b)
print("Final loss:", losses[-1])
print()

# 3.8 Learning Curve
print("=== 3.8 Learning Curve ===")
plot_learning_curve(losses, title="Loss Decreasing Over Time", 
                   ylabel="Binary Cross-Entropy Loss")

# 3.9 Final Probabilities
print("=== 3.9 Final Probabilities ===")
final_probs = sigmoid(w * X + b)
print("Final probabilities:", final_probs)
print()

# 3.10 From Probability to Decision
print("=== 3.10 From Probability to Decision ===")
threshold = 0.5
decisions = (final_probs >= threshold).astype(int)
print("Decisions (threshold=0.5):", decisions)
print("Actual values:", y)
print("Accuracy:", np.mean(decisions == y))
print()

# 3.11 Visualizing the Probability Curve
print("=== 3.11 Probability Curve Visualization ===")
x_vals = np.linspace(0, 5, 200)
probs = sigmoid(w * x_vals + b)
plot_probability_curve(x_vals, probs, X, y, threshold=0.5)

# 3.12 ROC Curve (Receiver Operating Characteristic)
print("=== 3.12 ROC Curve Analysis ===")
print("ROC curve shows model performance across different thresholds")
print("  - X-axis: False Positive Rate (1 - Specificity)")
print("  - Y-axis: True Positive Rate (Recall/Sensitivity)")
print("  - AUC (Area Under Curve): Higher is better (max = 1.0)")
print()

# Calculate ROC curve
# ROC (Receiver Operating Characteristic) curve shows model performance at different thresholds
def calculate_roc_curve(y_true, y_scores):
    """Calculate ROC curve points (FPR, TPR) and AUC score"""
    # Sort by scores (descending) - highest probabilities first
    # np.argsort(y_scores) returns indices that would sort the array in ascending order
    # [::-1] reverses the array, so we get descending order (highest first)
    sorted_indices = np.argsort(y_scores)[::-1]
    
    # Reorder both true labels and scores according to sorted indices
    # This allows us to process data from highest to lowest probability
    y_true_sorted = y_true[sorted_indices]      # Actual labels, sorted by score
    y_scores_sorted = y_scores[sorted_indices]  # Predicted probabilities, sorted
    
    # Calculate TPR and FPR for each threshold
    # np.unique() gets all unique probability values (potential thresholds)
    thresholds = np.unique(y_scores_sorted)
    # Add endpoints: 1.0 (all predictions positive) and 0.0 (all predictions negative)
    # This ensures we cover the full range of possible thresholds
    thresholds = np.append(thresholds, [1.0, 0.0])  # Add endpoints
    # Sort thresholds in descending order (from 1.0 to 0.0)
    thresholds = np.sort(thresholds)[::-1]
    
    # Lists to store True Positive Rate and False Positive Rate for each threshold
    tpr = []  # True Positive Rate (Recall/Sensitivity)
    fpr = []  # False Positive Rate (1 - Specificity)
    
    # For each threshold, calculate TPR and FPR
    for threshold in thresholds:
        # Convert probabilities to binary predictions
        # (y_scores_sorted >= threshold) creates boolean array: True if score >= threshold
        # .astype(int) converts True/False to 1/0
        y_pred = (y_scores_sorted >= threshold).astype(int)
        
        # Calculate confusion matrix components
        # True Positive (TP): Predicted positive AND actually positive
        # (y_pred == 1) & (y_true_sorted == 1) creates boolean array where both are 1
        # np.sum() counts how many True values
        tp = np.sum((y_pred == 1) & (y_true_sorted == 1))
        
        # False Positive (FP): Predicted positive BUT actually negative
        # (y_pred == 1) & (y_true_sorted == 0) finds false alarms
        fp = np.sum((y_pred == 1) & (y_true_sorted == 0))
        
        # True Negative (TN): Predicted negative AND actually negative
        # (y_pred == 0) & (y_true_sorted == 0) finds correct rejections
        tn = np.sum((y_pred == 0) & (y_true_sorted == 0))
        
        # False Negative (FN): Predicted negative BUT actually positive
        # (y_pred == 0) & (y_true_sorted == 1) finds missed positives
        fn = np.sum((y_pred == 0) & (y_true_sorted == 1))
        
        # Calculate True Positive Rate (TPR) = Recall = Sensitivity
        # TPR = TP / (TP + FN) = "Of all actual positives, how many did we catch?"
        # If denominator is 0 (no actual positives), set TPR to 0 to avoid division by zero
        tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Calculate False Positive Rate (FPR) = 1 - Specificity
        # FPR = FP / (FP + TN) = "Of all actual negatives, how many did we incorrectly flag?"
        # If denominator is 0 (no actual negatives), set FPR to 0 to avoid division by zero
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Store TPR and FPR for this threshold
        tpr.append(tpr_val)
        fpr.append(fpr_val)
    
    # Calculate AUC (Area Under Curve) using trapezoidal rule
    # np.trapz() approximates the integral (area under curve) using trapezoids
    # Higher AUC = better classifier (max = 1.0, random = 0.5)
    auc = np.trapz(tpr, fpr)
    
    # Return FPR array, TPR array, and AUC score
    return np.array(fpr), np.array(tpr), auc

# Get probabilities for all data points
# Use trained model to get predicted probabilities for all inputs
# sigmoid(w * X + b) converts raw scores to probabilities (0 to 1)
final_probs_all = sigmoid(w * X + b)

# Calculate ROC curve: returns FPR, TPR arrays, and AUC score
# y: actual labels (0 or 1)
# final_probs_all: predicted probabilities (0 to 1)
fpr, tpr, auc_score = calculate_roc_curve(y, final_probs_all)

# Print AUC score with 3 decimal places
# AUC (Area Under Curve) is a single number summarizing classifier performance
print(f"AUC Score: {auc_score:.3f}")  # {auc_score:.3f} formats to 3 decimal places
print("  AUC = 1.0: Perfect classifier")                    # Best possible
print("  AUC = 0.5: Random classifier (no better than guessing)")  # Worst (random)
print("  AUC > 0.7: Good classifier")                       # Acceptable performance
print()

# Plot ROC curve
# plot_roc_curve creates a visualization showing:
# - ROC curve (TPR vs FPR)
# - Diagonal line (random classifier baseline)
# - AUC score and shaded area
plot_roc_curve(fpr, tpr, auc_score, title="ROC Curve for Logistic Regression Model")

# Why Logistic Regression is Better
print("=== 3.13 Why Logistic Regression is Better ===")
print("✅ Smooth learning")
print("✅ Probability output")
print("✅ Stable training")
print("❌ Still only straight-line separation")
print()

# Exercises
print("=== Exercises ===")
print("Exercise 1: Try different thresholds (0.7, 0.3) and observe ROC curve changes")
print("Exercise 2: Add more data points and retrain, compare AUC scores")
print("Exercise 3: Why is sigmoid better than step function for learning?")
print("Exercise 4: What does an AUC of 0.8 mean in practical terms?")