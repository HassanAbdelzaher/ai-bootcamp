"""
Step 3c — Advanced Evaluation Metrics
Goal: Learn comprehensive metrics for evaluating model performance
Tools: Python + NumPy + Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from plotting import plot_learning_curve

print("=" * 70)
print("Step 3c: Advanced Evaluation Metrics")
print("=" * 70)
print()
print("Goal: Learn how to properly evaluate model performance")
print()

# ============================================================================
# 3c.1 Why Evaluation Metrics Matter
# ============================================================================
print("=== 3c.1 Why Evaluation Metrics Matter ===")
print()
print("Accuracy alone doesn't tell the full story!")
print("Different metrics reveal different aspects of model performance:")
print("  • Accuracy: Overall correctness")
print("  • Precision: How reliable are positive predictions?")
print("  • Recall: How many positives did we catch?")
print("  • F1-Score: Balanced measure of precision and recall")
print("  • ROC Curve: Trade-off between true and false positives")
print("  • AUC: Overall model quality")
print()

# ============================================================================
# 3c.2 Create a Classification Problem
# ============================================================================
print("=== 3c.2 Creating Classification Problem ===")

np.random.seed(42)

# Generate synthetic binary classification data
# num_samples: Number of data points to create
num_samples = 200
# X: Input features - random 2D points from standard normal distribution
# np.random.randn(num_samples, 2) creates array of shape (200, 2)
# Each row is one sample with 2 features
X = np.random.randn(num_samples, 2)

# Create labels (circular pattern - non-linear boundary)
# X[:, 0]**2 + X[:, 1]**2 calculates distance squared from origin (x² + y²)
# > 1.5 creates boolean array: True if point is outside circle of radius sqrt(1.5)
# .astype(int) converts True/False to 1/0
# This creates a circular decision boundary (harder than linear)
y = ((X[:, 0]**2 + X[:, 1]**2) > 1.5).astype(int)

# Add some noise to make problem more realistic
# np.random.random(num_samples) creates random values between 0 and 1
# < 0.1 means 10% of samples will be flipped (noise)
# np.where(condition, value_if_true, value_if_false)
# If random < 0.1: flip label (1 - y), else keep original label (y)
y = np.where(np.random.random(num_samples) < 0.1, 1 - y, y)

print(f"Dataset: {num_samples} samples")
print(f"Class 0: {np.sum(y == 0)} samples ({np.mean(y == 0)*100:.1f}%)")
print(f"Class 1: {np.sum(y == 1)} samples ({np.mean(y == 1)*100:.1f}%)")
print()

# Train a simple logistic regression model
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def train_logistic_regression(X, y, epochs=2000, lr=0.1):
    """Train a simple logistic regression model"""
    # Initialize weights randomly
    # X.shape[1] gets number of features (2 in this case)
    # np.random.randn(X.shape[1]) creates random values from standard normal distribution
    # * 0.1 scales down initial weights (smaller values help training)
    w = np.random.randn(X.shape[1]) * 0.1
    # Initialize bias to zero
    b = 0.0
    
    # Training loop: repeat many times
    for epoch in range(epochs):
        # ===== FORWARD PASS =====
        # Calculate scores: z = X @ w + b
        # X shape: (200, 2), w shape: (2,), result z shape: (200,)
        z = X @ w + b
        # Convert scores to probabilities using sigmoid
        y_pred = sigmoid(z)  # Shape: (200,) - probabilities between 0 and 1
        
        # ===== CALCULATE GRADIENTS =====
        # Gradient for weights: average((prediction - actual) × input)
        # (y_pred - y) is error for each sample, shape: (200,)
        # .reshape(-1, 1) converts to column vector: (200, 1)
        # * X broadcasts: (200, 1) * (200, 2) → (200, 2)
        # np.mean(..., axis=0) averages across samples, gives gradient per feature
        dw = np.mean((y_pred - y).reshape(-1, 1) * X, axis=0)
        # Gradient for bias: average(prediction - actual)
        db = np.mean(y_pred - y)
        
        # ===== UPDATE WEIGHTS =====
        # Move weights in opposite direction of gradient (to reduce loss)
        w -= lr * dw  # Update weights: w = w - learning_rate * gradient
        b -= lr * db  # Update bias: b = b - learning_rate * gradient
    
    # Return learned weights and bias
    return w, b

print("Training logistic regression model...")
w, b = train_logistic_regression(X, y)
print("Training complete!")
print()

# Get predictions and probabilities using trained model
# Calculate scores for all samples
z = X @ w + b  # Shape: (200,) - raw scores
# Convert scores to probabilities (0 to 1)
probabilities = sigmoid(z)  # Shape: (200,) - probabilities
# Convert probabilities to binary predictions
# (probabilities >= 0.5) creates boolean array: True if prob >= 0.5
# .astype(int) converts True/False to 1/0
predictions = (probabilities >= 0.5).astype(int)  # Shape: (200,) - binary predictions

print(f"Model predictions:")
print(f"  Probabilities range: {probabilities.min():.3f} to {probabilities.max():.3f}")
print(f"  Predictions: {np.sum(predictions == 0)} class 0, {np.sum(predictions == 1)} class 1")
print()

# ============================================================================
# 3c.3 Confusion Matrix (Deep Dive)
# ============================================================================
print("=== 3c.3 Confusion Matrix (Deep Dive) ===")
print()

def confusion_matrix(y_true, y_pred):
    """Calculate confusion matrix components"""
    # True Positives (TP): Predicted positive AND actually positive
    # (y_pred == 1) creates boolean array: True where prediction is 1
    # (y_true == 1) creates boolean array: True where actual is 1
    # & is element-wise AND: True only where both are True
    # np.sum() counts how many True values
    tp = np.sum((y_pred == 1) & (y_true == 1))  # True Positives
    
    # True Negatives (TN): Predicted negative AND actually negative
    # Both predicted and actual are 0
    tn = np.sum((y_pred == 0) & (y_true == 0))  # True Negatives
    
    # False Positives (FP): Predicted positive BUT actually negative
    # Model said "positive" but it was actually "negative" (Type I error)
    fp = np.sum((y_pred == 1) & (y_true == 0))  # False Positives
    
    # False Negatives (FN): Predicted negative BUT actually positive
    # Model said "negative" but it was actually "positive" (Type II error)
    fn = np.sum((y_pred == 0) & (y_true == 1))  # False Negatives
    
    return tp, tn, fp, fn

tp, tn, fp, fn = confusion_matrix(y, predictions)

print("Confusion Matrix:")
print("                 Predicted")
print("              Class 0  Class 1")
print(f"Actual Class 0    {tn:3d}      {fp:3d}")
print(f"         Class 1    {fn:3d}      {tp:3d}")
print()

# Visualize confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
cm = np.array([[tn, fp], [fn, tp]])
im = ax.imshow(cm, interpolation='nearest', cmap='Blues', aspect='auto')
ax.figure.colorbar(im, ax=ax)

# Add text annotations
thresh = cm.max() / 2.
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i, j]}',
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=16, fontweight='bold')

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Class 0', 'Class 1'], fontsize=12, fontweight='bold')
ax.set_yticklabels(['Class 0', 'Class 1'], fontsize=12, fontweight='bold')
ax.set_xlabel('Predicted', fontsize=13, fontweight='bold')
ax.set_ylabel('Actual', fontsize=13, fontweight='bold')
ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=15)

# Add labels
ax.text(0, -0.3, f'TN={tn}', ha='center', fontsize=10, color='green', fontweight='bold')
ax.text(1, -0.3, f'FP={fp}', ha='center', fontsize=10, color='red', fontweight='bold')
ax.text(-0.3, 0, f'FN={fn}', ha='center', va='center', fontsize=10, color='red', fontweight='bold', rotation=90)
ax.text(-0.3, 1, f'TP={tp}', ha='center', va='center', fontsize=10, color='green', fontweight='bold', rotation=90)

plt.tight_layout()
plt.show()

print("Understanding the confusion matrix:")
print(f"  TP (True Positives): {tp} - Correctly predicted as class 1")
print(f"  TN (True Negatives): {tn} - Correctly predicted as class 0")
print(f"  FP (False Positives): {fp} - Incorrectly predicted as class 1 (Type I error)")
print(f"  FN (False Negatives): {fn} - Incorrectly predicted as class 0 (Type II error)")
print()

# ============================================================================
# 3c.4 Basic Metrics from Confusion Matrix
# ============================================================================
print("=== 3c.4 Basic Metrics ===")
print()

# Calculate metrics from confusion matrix components
# Accuracy: Overall correctness - (correct predictions) / (total predictions)
accuracy = (tp + tn) / (tp + tn + fp + fn)

# Precision: How reliable are positive predictions?
# = (True Positives) / (All predicted positives)
# High precision = few false positives (when model says "positive", it's usually right)
# if (tp + fp) > 0: avoid division by zero (if no positive predictions)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0

# Recall (Sensitivity): How many positives did we catch?
# = (True Positives) / (All actual positives)
# High recall = few false negatives (we catch most of the actual positives)
# if (tp + fn) > 0: avoid division by zero (if no actual positives)
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

# Specificity: How many negatives did we catch?
# = (True Negatives) / (All actual negatives)
# High specificity = few false positives (we correctly reject most negatives)
# if (tn + fp) > 0: avoid division by zero (if no actual negatives)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

# F1-Score: Harmonic mean of precision and recall
# Balances precision and recall (useful when you care about both)
# Formula: 2 * (precision * recall) / (precision + recall)
# if (precision + recall) > 0: avoid division by zero
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("Metrics from Confusion Matrix:")
print(f"  Accuracy:  {accuracy:.3f} = (TP + TN) / Total")
print(f"  Precision: {precision:.3f} = TP / (TP + FP) - How reliable are positive predictions?")
print(f"  Recall:    {recall:.3f} = TP / (TP + FN) - How many positives did we catch?")
print(f"  Specificity: {specificity:.3f} = TN / (TN + FP) - How many negatives did we catch?")
print(f"  F1-Score:  {f1_score:.3f} = 2 * (Precision * Recall) / (Precision + Recall)")
print()

# Visualize metrics
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart of metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
values = [accuracy, precision, recall, specificity, f1_score]
colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightyellow', 'plum']

bars = axes[0].bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
axes[0].set_ylabel('Score', fontsize=12, fontweight='bold')
axes[0].set_title('Classification Metrics', fontsize=14, fontweight='bold')
axes[0].set_ylim([0, 1.1])
axes[0].grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, values):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

# Pie chart showing correct vs incorrect
correct = tp + tn
incorrect = fp + fn
sizes = [correct, incorrect]
labels = ['Correct', 'Incorrect']
colors_pie = ['lightgreen', 'lightcoral']
explode = (0.05, 0.05)

axes[1].pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_pie,
           explode=explode, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[1].set_title('Overall Performance', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================================
# 3c.5 ROC Curve and AUC
# ============================================================================
print("=== 3c.5 ROC Curve and AUC ===")
print()
print("ROC (Receiver Operating Characteristic) Curve:")
print("  • Shows trade-off between True Positive Rate (Recall) and")
print("    False Positive Rate (1 - Specificity)")
print("  • X-axis: False Positive Rate")
print("  • Y-axis: True Positive Rate (Recall)")
print("  • Better models have curves closer to top-left corner")
print()

def calculate_roc_curve(y_true, y_scores):
    """Calculate ROC curve"""
    # Sort by scores (descending)
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]
    
    # Calculate TPR and FPR for different thresholds
    thresholds = np.unique(y_scores_sorted)
    thresholds = np.append(thresholds, [1.0, 0.0])
    thresholds = np.sort(thresholds)[::-1]
    
    tpr = []
    fpr = []
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr.append(tpr_val)
        fpr.append(fpr_val)
    
    return np.array(fpr), np.array(tpr), thresholds

def calculate_auc(fpr, tpr):
    """Calculate Area Under Curve (AUC)"""
    # Sort by FPR
    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]
    
    # Calculate AUC using trapezoidal rule
    auc = np.trapz(tpr_sorted, fpr_sorted)
    return auc

fpr, tpr, thresholds = calculate_roc_curve(y, probabilities)
auc = calculate_auc(fpr, tpr)

print(f"AUC (Area Under Curve): {auc:.3f}")
print("  • AUC = 1.0: Perfect classifier")
print("  • AUC = 0.5: Random classifier (no better than guessing)")
print("  • AUC > 0.7: Good classifier")
print("  • AUC > 0.9: Excellent classifier")
print()

# Plot ROC curve
fig, ax = plt.subplots(figsize=(9, 7))
ax.plot(fpr, tpr, linewidth=3, color='steelblue', label=f'ROC Curve (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.5)', alpha=0.5)
ax.fill_between(fpr, 0, tpr, alpha=0.3, color='steelblue', label='AUC Area')
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
ax.set_title('ROC Curve', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

# Add optimal threshold point
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=12, 
       label=f'Optimal Threshold ({optimal_threshold:.2f})')
ax.legend(fontsize=11, loc='lower right')

plt.tight_layout()
plt.show()

# ============================================================================
# 3c.6 Precision-Recall Curve
# ============================================================================
print("=== 3c.6 Precision-Recall Curve ===")
print()
print("Precision-Recall Curve:")
print("  • Better for imbalanced datasets")
print("  • Shows trade-off between Precision and Recall")
print("  • X-axis: Recall")
print("  • Y-axis: Precision")
print()

def calculate_pr_curve(y_true, y_scores):
    """Calculate Precision-Recall curve"""
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]
    
    thresholds = np.unique(y_scores_sorted)
    thresholds = np.append(thresholds, [1.0, 0.0])
    thresholds = np.sort(thresholds)[::-1]
    
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    return np.array(recalls), np.array(precisions), thresholds

recalls_pr, precisions_pr, thresholds_pr = calculate_pr_curve(y, probabilities)
pr_auc = np.trapz(precisions_pr, recalls_pr)

print(f"PR-AUC (Area Under PR Curve): {pr_auc:.3f}")
print("  • Higher is better")
print("  • Shows performance across all thresholds")
print()

# Plot PR curve
fig, ax = plt.subplots(figsize=(9, 7))
ax.plot(recalls_pr, precisions_pr, linewidth=3, color='coral', 
       label=f'PR Curve (AUC = {pr_auc:.3f})')
ax.fill_between(recalls_pr, 0, precisions_pr, alpha=0.3, color='coral', label='PR-AUC Area')
ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc='lower left')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.show()

# ============================================================================
# 3c.7 F-Beta Scores
# ============================================================================
print("=== 3c.7 F-Beta Scores ===")
print()
print("F-Beta Score: Weighted harmonic mean of Precision and Recall")
print("  Fβ = (1 + β²) * (Precision * Recall) / (β² * Precision + Recall)")
print()
print("  • β = 1: F1-Score (equal weight to Precision and Recall)")
print("  • β < 1: Favors Precision (fewer false positives)")
print("  • β > 1: Favors Recall (fewer false negatives)")
print()

def f_beta_score(precision, recall, beta=1):
    """Calculate F-Beta score"""
    if precision + recall == 0:
        return 0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

f1 = f_beta_score(precision, recall, beta=1)
f2 = f_beta_score(precision, recall, beta=2)  # Favors recall
f05 = f_beta_score(precision, recall, beta=0.5)  # Favors precision

print("F-Beta Scores:")
print(f"  F0.5 (favors Precision): {f05:.3f}")
print(f"  F1 (balanced):           {f1:.3f}")
print(f"  F2 (favors Recall):      {f2:.3f}")
print()

# Visualize F-beta scores
fig, ax = plt.subplots(figsize=(8, 5))
betas = [0.5, 1.0, 2.0]
f_scores = [f05, f1, f2]
colors_f = ['lightblue', 'steelblue', 'darkblue']

bars = ax.bar([f'F{β}' for β in betas], f_scores, color=colors_f, 
              alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('F-Beta Score', fontsize=12, fontweight='bold')
ax.set_title('F-Beta Scores Comparison', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1.1])
ax.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, f_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================================
# 3c.8 Regression Metrics
# ============================================================================
print("=== 3c.8 Regression Metrics ===")
print()
print("For regression problems (predicting numbers), different metrics:")
print()

# Create synthetic regression data
np.random.seed(42)
X_reg = np.random.randn(100, 1) * 2
y_reg_true = 2 * X_reg.flatten() + 1 + np.random.randn(100) * 0.5
y_reg_pred = 1.9 * X_reg.flatten() + 1.1 + np.random.randn(100) * 0.3

# Calculate regression metrics
mse = np.mean((y_reg_true - y_reg_pred) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_reg_true - y_reg_pred))
mape = np.mean(np.abs((y_reg_true - y_reg_pred) / (y_reg_true + 1e-9))) * 100

# R-squared
ss_res = np.sum((y_reg_true - y_reg_pred) ** 2)
ss_tot = np.sum((y_reg_true - np.mean(y_reg_true)) ** 2)
r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

print("Regression Metrics:")
print(f"  MSE (Mean Squared Error):  {mse:.3f} - Penalizes large errors")
print(f"  RMSE (Root MSE):            {rmse:.3f} - In same units as target")
print(f"  MAE (Mean Absolute Error):  {mae:.3f} - Average error magnitude")
print(f"  MAPE (Mean Absolute % Error): {mape:.2f}% - Percentage error")
print(f"  R² (R-squared):            {r_squared:.3f} - Proportion of variance explained")
print()
print("Interpretation:")
print(f"  • R² = {r_squared:.3f} means model explains {r_squared*100:.1f}% of variance")
print(f"  • Lower MSE/RMSE/MAE is better")
print(f"  • R² closer to 1.0 is better")
print()

# Visualize regression metrics
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot with predictions
axes[0].scatter(y_reg_true, y_reg_pred, alpha=0.6, s=80, edgecolors='black', linewidth=1)
axes[0].plot([y_reg_true.min(), y_reg_true.max()], 
            [y_reg_true.min(), y_reg_true.max()], 
            'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('True Values', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
axes[0].set_title('True vs Predicted Values', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Metrics bar chart
metrics_reg = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R²']
values_reg = [mse, rmse, mae, mape/10, r_squared]  # Scale MAPE for visualization
colors_reg = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow', 'plum']

bars = axes[1].bar(metrics_reg, values_reg, color=colors_reg, alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
axes[1].set_ylabel('Score', fontsize=12, fontweight='bold')
axes[1].set_title('Regression Metrics', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

for bar, val, metric in zip(bars, values_reg, metrics_reg):
    height = bar.get_height()
    if metric == 'MAPE':
        label = f'{mape:.1f}%'
    elif metric == 'R²':
        label = f'{r_squared:.3f}'
    else:
        label = f'{val:.3f}'
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================================
# 3c.9 Summary and Best Practices
# ============================================================================
print("=== 3c.9 Summary and Best Practices ===")
print()

print("✅ Classification Metrics:")
print("  • Use Accuracy for balanced datasets")
print("  • Use Precision/Recall/F1 for imbalanced datasets")
print("  • Use ROC-AUC for overall model quality")
print("  • Use PR-AUC for imbalanced datasets")
print()

print("✅ Regression Metrics:")
print("  • Use RMSE for same-scale interpretation")
print("  • Use MAE for robust error measure")
print("  • Use R² for variance explanation")
print("  • Use MAPE for percentage errors")
print()

print("✅ Best Practices:")
print("  1. Don't rely on a single metric")
print("  2. Choose metrics based on your problem")
print("  3. Consider business context (cost of false positives vs negatives)")
print("  4. Visualize metrics (ROC, PR curves)")
print("  5. Compare multiple models using same metrics")
print()

print("=" * 70)
print("Step 3c Complete! You understand evaluation metrics!")
print("=" * 70)
