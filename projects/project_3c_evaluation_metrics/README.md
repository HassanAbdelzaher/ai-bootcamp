# Project 3c: Advanced Evaluation Metrics

> **Master model evaluation with comprehensive metrics for classification and regression**

**Difficulty**: ⭐⭐ Intermediate  
**Time**: 2-3 hours  
**Prerequisites**: Steps 0-3 (Especially Step 3: Logistic Regression)

---

## 📚 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem: Comprehensive Model Evaluation](#problem-comprehensive-model-evaluation)
3. [Key Concepts](#key-concepts)
4. [Step-by-Step Guide](#step-by-step-guide)
5. [Expected Results](#expected-results)
6. [Extension Ideas](#extension-ideas)

---

## 🎯 Project Overview

This project teaches you to evaluate models using advanced metrics beyond simple accuracy. You'll learn to:

- Calculate classification metrics (Precision, Recall, F1-Score, ROC, AUC)
- Calculate regression metrics (R², MAE, MAPE, RMSE)
- Visualize model performance
- Compare different models
- Understand when to use each metric

### Why Advanced Metrics?

- **Accuracy is misleading**: Doesn't show per-class performance
- **Real-world needs**: Different metrics for different problems
- **Model comparison**: Need standardized metrics
- **Business decisions**: Metrics guide important choices

---

## 📋 Problem: Comprehensive Model Evaluation

### Task

Build a comprehensive evaluation system that calculates and visualizes:
1. **Classification Metrics**: For binary and multi-class problems
2. **Regression Metrics**: For continuous value predictions
3. **Visualizations**: ROC curves, PR curves, confusion matrices

### Learning Objectives

- Understand precision, recall, F1-score
- Calculate ROC curves and AUC
- Use regression metrics effectively
- Visualize model performance
- Compare multiple models

---

## 🧠 Key Concepts

### 1. Classification Metrics

**Confusion Matrix**:
```
              Predicted
            Negative  Positive
Actual Negative   TN      FP
       Positive   FN      TP
```

**Metrics**:
- **Precision**: TP / (TP + FP) - Of predicted positives, how many are correct?
- **Recall**: TP / (TP + FN) - Of actual positives, how many did we catch?
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall) - Balanced metric
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN) - Overall correctness

### 2. ROC Curve

**Purpose**: Shows model performance across different thresholds

**Axes**:
- X-axis: False Positive Rate (1 - Specificity)
- Y-axis: True Positive Rate (Recall/Sensitivity)

**AUC**: Area Under Curve (0.5 = random, 1.0 = perfect)

### 3. Regression Metrics

- **MSE**: Mean Squared Error (penalizes large errors)
- **RMSE**: Root Mean Squared Error (in same units as target)
- **MAE**: Mean Absolute Error (average error)
- **MAPE**: Mean Absolute Percentage Error (percentage error)
- **R²**: Coefficient of Determination (proportion of variance explained)

---

## 🚀 Step-by-Step Guide

### Step 1: Create Classification Dataset

```python
import numpy as np
from sklearn.datasets import make_classification

# Create binary classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_classes=2,
    random_state=42
)

# Split into train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Step 2: Train Multiple Models

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Train multiple models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'SVM': SVC(probability=True)
}

predictions = {}
probabilities = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions[name] = model.predict(X_test)
    probabilities[name] = model.predict_proba(X_test)[:, 1]
```

### Step 3: Calculate Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

def calculate_classification_metrics(y_true, y_pred, y_proba):
    """Calculate comprehensive classification metrics"""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['f1_score'] = f1_score(y_true, y_pred)
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    metrics['roc_auc'] = auc(fpr, tpr)
    metrics['roc_curve'] = (fpr, tpr)
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    metrics['pr_curve'] = (precision, recall)
    
    return metrics

# Calculate for each model
results = {}
for name in models.keys():
    results[name] = calculate_classification_metrics(
        y_test, predictions[name], probabilities[name]
    )
```

### Step 4: Visualize Results

```python
import matplotlib.pyplot as plt

# Plot ROC curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for name, result in results.items():
    fpr, tpr = result['roc_curve']
    plt.plot(fpr, tpr, label=f"{name} (AUC={result['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(True)

# Plot Precision-Recall curves
plt.subplot(1, 2, 2)
for name, result in results.items():
    precision, recall = result['pr_curve']
    plt.plot(recall, precision, label=name)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

### Step 5: Regression Metrics

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create regression dataset
from sklearn.datasets import make_regression
X_reg, y_reg = make_regression(
    n_samples=1000, n_features=10, noise=20, random_state=42
)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Train model
reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)
y_pred_reg = reg_model.predict(X_test_reg)

# Calculate regression metrics
def calculate_regression_metrics(y_true, y_pred):
    """Calculate comprehensive regression metrics"""
    metrics = {}
    
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return metrics

reg_metrics = calculate_regression_metrics(y_test_reg, y_pred_reg)
print("Regression Metrics:")
for metric, value in reg_metrics.items():
    print(f"  {metric.upper()}: {value:.4f}")
```

---

## 📊 Expected Results

### Classification Results

```
Model Comparison:
Logistic Regression:
  Accuracy: 0.875
  Precision: 0.857
  Recall: 0.800
  F1-Score: 0.828
  ROC-AUC: 0.923

Random Forest:
  Accuracy: 0.915
  Precision: 0.900
  Recall: 0.875
  F1-Score: 0.887
  ROC-AUC: 0.967

SVM:
  Accuracy: 0.890
  Precision: 0.875
  Recall: 0.825
  F1-Score: 0.849
  ROC-AUC: 0.945
```

### Regression Results

```
Regression Metrics:
MSE: 425.67
RMSE: 20.63
MAE: 16.45
R²: 0.892
MAPE: 8.23%
```

---

## 💡 Extension Ideas

1. **Multi-Class Metrics**
   - Extend to 3+ classes
   - Per-class precision/recall
   - Macro vs micro averaging

2. **Custom Metrics**
   - Business-specific metrics
   - Cost-sensitive metrics
   - Weighted metrics

3. **Model Comparison Dashboard**
   - Interactive visualizations
   - Side-by-side comparisons
   - Export reports

---

## ✅ Success Criteria

- ✅ Calculate all classification metrics
- ✅ Calculate all regression metrics
- ✅ Visualize ROC and PR curves
- ✅ Compare multiple models
- ✅ Understand when to use each metric

---

**Ready to master model evaluation? Let's build comprehensive metrics!** 🚀
