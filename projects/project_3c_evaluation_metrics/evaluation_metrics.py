"""
Project 3c: Advanced Evaluation Metrics
Comprehensive model evaluation for classification and regression
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    mean_squared_error, mean_absolute_error, r2_score
)
import sys
import os

# Add parent directory to path to import plotting utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from plotting import plot_roc_curve, plot_pr_curve, plot_confusion_matrix_style

print("=" * 70)
print("Project 3c: Advanced Evaluation Metrics")
print("=" * 70)
print()

# ============================================================================
# Part 1: Classification Metrics
# ============================================================================
print("=" * 70)
print("Part 1: Classification Metrics")
print("=" * 70)
print()

# Create classification dataset
# make_classification: Generate synthetic classification dataset
# n_samples=1000: Total number of samples to generate
# n_features=10: Number of input features (dimensions)
# n_informative=5: Number of features that actually help with classification
# n_redundant=2: Number of redundant features (linear combinations of informative)
# n_classes=2: Binary classification (2 classes: 0 and 1)
# random_state=42: Seed for reproducibility (same dataset every time)
print("Creating classification dataset...")
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_classes=2,
    random_state=42
)

# Split dataset into training and test sets
# train_test_split: Randomly splits data into train/test
# test_size=0.2: 20% of data goes to test set, 80% to training
# random_state=42: Ensures same split every time (reproducibility)
# Returns: X_train, X_test (features), y_train, y_test (labels)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print()

# Train multiple models
# We'll train 3 different models to compare their performance
# This demonstrates that different models can have different strengths
print("Training multiple models...")
models = {
    # Logistic Regression: Linear classifier, fast and interpretable
    # max_iter=1000: Maximum iterations for convergence
    'Logistic Regression': LogisticRegression(max_iter=1000),
    
    # Random Forest: Ensemble of decision trees, robust and powerful
    # n_estimators=100: Number of trees in the forest
    # random_state=42: For reproducibility
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    
    # SVM (Support Vector Machine): Finds optimal decision boundary
    # probability=True: Enable probability predictions (needed for ROC curve)
    # random_state=42: For reproducibility
    'SVM': SVC(probability=True, random_state=42)
}

# Dictionaries to store predictions and probabilities for each model
# predictions: Binary predictions (0 or 1) for each model
predictions = {}
# probabilities: Probability scores (0.0 to 1.0) for positive class
# Needed for ROC and PR curves
probabilities = {}

# Train each model and make predictions
for name, model in models.items():
    print(f"  Training {name}...")
    # model.fit(): Train the model on training data
    # This learns the patterns from X_train and y_train
    model.fit(X_train, y_train)
    
    # model.predict(): Make binary predictions (0 or 1) on test set
    # Returns array of predicted classes
    predictions[name] = model.predict(X_test)
    
    # model.predict_proba(): Get probability scores for each class
    # Returns array of shape (n_samples, n_classes) with probabilities
    # [:, 1]: Select only probabilities for positive class (class 1)
    # This gives us probability that each sample belongs to positive class
    probabilities[name] = model.predict_proba(X_test)[:, 1]

print()

# Calculate metrics for each model
print("=" * 70)
print("Classification Metrics Results")
print("=" * 70)
print()

results = {}

for name in models.keys():
    # ===== BASIC METRICS =====
    # Accuracy: Overall correctness (correct predictions / total predictions)
    # Formula: (TP + TN) / (TP + TN + FP + FN)
    # Range: 0.0 to 1.0, higher is better
    accuracy = accuracy_score(y_test, predictions[name])
    
    # Precision: Of predicted positives, how many are actually positive?
    # Formula: TP / (TP + FP)
    # Measures: "When I predict positive, how often am I right?"
    # Range: 0.0 to 1.0, higher is better
    precision = precision_score(y_test, predictions[name])
    
    # Recall (Sensitivity): Of actual positives, how many did I catch?
    # Formula: TP / (TP + FN)
    # Measures: "Of all positives, how many did I find?"
    # Range: 0.0 to 1.0, higher is better
    recall = recall_score(y_test, predictions[name])
    
    # F1-Score: Harmonic mean of precision and recall
    # Formula: 2 × (Precision × Recall) / (Precision + Recall)
    # Balances precision and recall (good when both matter)
    # Range: 0.0 to 1.0, higher is better
    f1 = f1_score(y_test, predictions[name])
    
    # ===== CONFUSION MATRIX =====
    # Shows breakdown of predictions vs actual labels
    # Format: [[TN, FP], [FN, TP]]
    #   TN (True Negative): Correctly predicted negative
    #   FP (False Positive): Incorrectly predicted positive (Type I error)
    #   FN (False Negative): Incorrectly predicted negative (Type II error)
    #   TP (True Positive): Correctly predicted positive
    cm = confusion_matrix(y_test, predictions[name])
    
    # ===== ROC CURVE =====
    # ROC (Receiver Operating Characteristic) curve
    # Shows model performance across different classification thresholds
    # roc_curve(): Calculates FPR and TPR for different thresholds
    #   y_test: True labels
    #   probabilities[name]: Predicted probabilities for positive class
    # Returns: fpr (False Positive Rate), tpr (True Positive Rate), thresholds
    # We use _ to ignore thresholds (don't need them)
    fpr, tpr, _ = roc_curve(y_test, probabilities[name])
    
    # AUC (Area Under Curve): Area under ROC curve
    # auc(): Calculates area using trapezoidal rule
    # Range: 0.0 to 1.0
    #   0.5 = Random classifier (no better than guessing)
    #   1.0 = Perfect classifier
    #   >0.7 = Good classifier
    roc_auc = auc(fpr, tpr)
    
    # ===== PRECISION-RECALL CURVE =====
    # Shows precision vs recall across different thresholds
    # Useful when classes are imbalanced (unlike ROC)
    # precision_recall_curve(): Calculates precision and recall for different thresholds
    # Returns: precision, recall, thresholds
    # We use _ to ignore thresholds
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, probabilities[name])
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'roc_curve': (fpr, tpr),
        'roc_auc': roc_auc,
        'pr_curve': (precision_curve, recall_curve)
    }
    
    print(f"{name}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print()

# Visualize ROC curves
print("Plotting ROC curves...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for name, result in results.items():
    fpr, tpr = result['roc_curve']
    plt.plot(fpr, tpr, label=f"{name} (AUC={result['roc_auc']:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Visualize Precision-Recall curves
plt.subplot(1, 2, 2)
for name, result in results.items():
    precision_curve, recall_curve = result['pr_curve']
    plt.plot(recall_curve, precision_curve, label=name, linewidth=2)
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('roc_pr_curves.png', dpi=150, bbox_inches='tight')
print("Saved: roc_pr_curves.png")
print()

# Confusion matrices
print("Plotting confusion matrices...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (name, result) in enumerate(results.items()):
    cm = result['confusion_matrix']
    im = axes[idx].imshow(cm, cmap='Blues', interpolation='nearest')
    axes[idx].set_title(f'{name}\nConfusion Matrix', fontweight='bold')
    axes[idx].set_ylabel('Actual')
    axes[idx].set_xlabel('Predicted')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[idx].text(j, i, format(cm[i, j], 'd'),
                          ha="center", va="center",
                          color="white" if cm[i, j] > thresh else "black",
                          fontweight='bold')
    
    plt.colorbar(im, ax=axes[idx])

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
print("Saved: confusion_matrices.png")
print()

# ============================================================================
# Part 2: Regression Metrics
# ============================================================================
print("=" * 70)
print("Part 2: Regression Metrics")
print("=" * 70)
print()

# Create regression dataset
print("Creating regression dataset...")
X_reg, y_reg = make_regression(
    n_samples=1000,
    n_features=10,
    noise=20,
    random_state=42
)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train_reg)}")
print(f"Test samples: {len(X_test_reg)}")
print()

# Train regression model
# LinearRegression: Simple linear regression model
# Fits a line to the data: y = w1*x1 + w2*x2 + ... + b
print("Training regression model...")
reg_model = LinearRegression()

# Train model on training data
# reg_model.fit(): Learns weights (w) and bias (b) from training data
reg_model.fit(X_train_reg, y_train_reg)

# Make predictions on test set
# reg_model.predict(): Uses learned weights to predict target values
# Returns array of predicted values
y_pred_reg = reg_model.predict(X_test_reg)

print()

# Calculate regression metrics
print("=" * 70)
print("Regression Metrics Results")
print("=" * 70)
print()

# MSE (Mean Squared Error): Average of squared differences
# Formula: mean((y_true - y_pred)²)
# Penalizes large errors more than small errors
# Range: 0 to infinity, lower is better
# Units: Same as target squared (e.g., dollars²)
mse = mean_squared_error(y_test_reg, y_pred_reg)

# RMSE (Root Mean Squared Error): Square root of MSE
# Formula: sqrt(MSE)
# Same units as target (e.g., dollars)
# More interpretable than MSE
# Range: 0 to infinity, lower is better
rmse = np.sqrt(mse)

# MAE (Mean Absolute Error): Average of absolute differences
# Formula: mean(|y_true - y_pred|)
# Treats all errors equally (unlike MSE)
# Range: 0 to infinity, lower is better
# Units: Same as target
mae = mean_absolute_error(y_test_reg, y_pred_reg)

# R² (Coefficient of Determination): Proportion of variance explained
# Formula: 1 - (SS_res / SS_tot)
#   SS_res: Sum of squared residuals (errors)
#   SS_tot: Total sum of squares (variance in data)
# Range: -infinity to 1.0
#   1.0 = Perfect predictions (explains all variance)
#   0.0 = No better than predicting the mean
#   <0.0 = Worse than predicting the mean
r2 = r2_score(y_test_reg, y_pred_reg)

# MAPE (Mean Absolute Percentage Error): Average percentage error
# Formula: mean(|(y_true - y_pred) / y_true|) × 100
# (y_test_reg - y_pred_reg) / y_test_reg: Percentage error for each sample
# np.abs(...): Absolute value (ignore sign)
# + 1e-9: Small value to avoid division by zero
# × 100: Convert to percentage
# Range: 0% to infinity, lower is better
# Useful for understanding error relative to actual values
mape = np.mean(np.abs((y_test_reg - y_pred_reg) / (y_test_reg + 1e-9))) * 100

print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")
print(f"MAPE: {mape:.2f}%")
print()

# Visualize regression results
print("Plotting regression results...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Actual vs Predicted
axes[0].scatter(y_test_reg, y_pred_reg, alpha=0.6)
axes[0].plot([y_test_reg.min(), y_test_reg.max()], 
             [y_test_reg.min(), y_test_reg.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Values', fontsize=12)
axes[0].set_ylabel('Predicted Values', fontsize=12)
axes[0].set_title(f'Actual vs Predicted (R² = {r2:.3f})', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Residuals
residuals = y_test_reg - y_pred_reg
axes[1].scatter(y_pred_reg, residuals, alpha=0.6)
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Values', fontsize=12)
axes[1].set_ylabel('Residuals', fontsize=12)
axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regression_results.png', dpi=150, bbox_inches='tight')
print("Saved: regression_results.png")
print()

print("=" * 70)
print("Project 3c Complete!")
print("=" * 70)
print()
print("Summary:")
print("  ✅ Calculated classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC)")
print("  ✅ Calculated regression metrics (MSE, RMSE, MAE, R², MAPE)")
print("  ✅ Visualized ROC and Precision-Recall curves")
print("  ✅ Visualized confusion matrices")
print("  ✅ Visualized regression predictions and residuals")
print()
