"""
Step 12: AI Ethics and Responsible AI
Understanding bias, fairness, and ethical considerations in AI
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import sys
import os

# Add parent directory to path to import plotting utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from plotting import plot_confusion_matrix_style

print("=" * 70)
print("Step 12: AI Ethics and Responsible AI")
print("=" * 70)
print()

# ============================================================================
# Part 1: Understanding Bias in AI
# ============================================================================
print("=" * 70)
print("Part 1: Understanding Bias in AI")
print("=" * 70)
print()

# Create a biased dataset
# Simulating a scenario where a model might learn biased patterns
print("Creating dataset with potential bias...")
print("Scenario: Hiring prediction model")
print("  Feature 1: Years of experience")
print("  Feature 2: Education level")
print("  Feature 3: Age (potential bias source)")
print()

# Generate synthetic data
# Note: In real scenarios, bias can come from historical data
np.random.seed(42)
n_samples = 1000

# Feature 1: Years of experience (0-20)
experience = np.random.uniform(0, 20, n_samples)

# Feature 2: Education level (1-5, where 5 is highest)
education = np.random.randint(1, 6, n_samples)

# Feature 3: Age (potential bias - older candidates might be discriminated)
age = np.random.uniform(25, 65, n_samples)

# Create target: Hired (1) or Not Hired (0)
# Simulate bias: Older candidates less likely to be hired (unfair)
# This represents historical bias in the data
bias_factor = 0.3  # Strength of age bias
hired_prob = (
    0.3 * (experience / 20) +  # More experience helps
    0.3 * (education / 5) +     # Higher education helps
    0.4 * np.random.random(n_samples) -  # Random component
    bias_factor * ((age - 25) / 40)  # Age bias (older = less likely)
)
hired = (hired_prob > 0.5).astype(int)

# Combine features
X = np.column_stack([experience, education, age])
y = hired

print(f"Dataset created: {len(X)} samples")
print(f"  Hired: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
print(f"  Not Hired: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
print()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
print("Training model on biased data...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")
print()

# ============================================================================
# Part 2: Detecting Bias
# ============================================================================
print("=" * 70)
print("Part 2: Detecting Bias in Model Predictions")
print("=" * 70)
print()

# Analyze predictions by age groups
# Split test set into age groups
age_test = X_test[:, 2]  # Age is the third feature
young_age = age_test < 35
middle_age = (age_test >= 35) & (age_test < 50)
old_age = age_test >= 50

# Calculate hiring rates by age group
young_hired_rate = np.mean(y_pred[young_age] == 1)
middle_hired_rate = np.mean(y_pred[middle_age] == 1)
old_hired_rate = np.mean(y_pred[old_age] == 1)

print("Hiring Rate by Age Group:")
print(f"  Young (<35): {young_hired_rate:.2%}")
print(f"  Middle (35-50): {middle_hired_rate:.2%}")
print(f"  Old (>50): {old_hired_rate:.2%}")
print()

# Calculate actual rates in test data
young_actual = np.mean(y_test[young_age] == 1)
middle_actual = np.mean(y_test[middle_age] == 1)
old_actual = np.mean(y_test[old_age] == 1)

print("Actual Hiring Rate by Age Group (in test data):")
print(f"  Young (<35): {young_actual:.2%}")
print(f"  Middle (35-50): {middle_actual:.2%}")
print(f"  Old (>50): {old_actual:.2%}")
print()

# Bias detection: Compare predicted vs actual
print("⚠️  BIAS DETECTED:")
print("  Model shows different hiring rates across age groups")
print("  This could indicate age discrimination")
print()

# ============================================================================
# Part 3: Fairness Metrics
# ============================================================================
print("=" * 70)
print("Part 3: Fairness Metrics")
print("=" * 70)
print()

def demographic_parity(y_pred, groups):
    """
    Demographic Parity: Equal positive prediction rates across groups
    Also called: Statistical Parity
    """
    rates = []
    for group_mask in groups:
        if np.sum(group_mask) > 0:
            rate = np.mean(y_pred[group_mask] == 1)
            rates.append(rate)
    return rates

def equalized_odds(y_pred, y_true, groups):
    """
    Equalized Odds: Equal TPR and FPR across groups
    TPR = True Positive Rate (recall)
    FPR = False Positive Rate
    """
    metrics = []
    for group_mask in groups:
        if np.sum(group_mask) > 0:
            group_pred = y_pred[group_mask]
            group_true = y_true[group_mask]
            
            # True Positive Rate (TPR)
            tp = np.sum((group_pred == 1) & (group_true == 1))
            fn = np.sum((group_pred == 0) & (group_true == 1))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # False Positive Rate (FPR)
            fp = np.sum((group_pred == 1) & (group_true == 0))
            tn = np.sum((group_pred == 0) & (group_true == 0))
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            metrics.append({'tpr': tpr, 'fpr': fpr})
    return metrics

# Calculate fairness metrics
groups = [young_age, middle_age, old_age]
group_names = ['Young', 'Middle', 'Old']

# Demographic Parity
parity_rates = demographic_parity(y_pred, groups)
print("Demographic Parity (Positive Prediction Rates):")
for name, rate in zip(group_names, parity_rates):
    print(f"  {name}: {rate:.2%}")
print()

# Calculate disparity
max_rate = max(parity_rates)
min_rate = min(parity_rates)
disparity = max_rate - min_rate
print(f"Disparity: {disparity:.2%} (lower is better, 0% = perfect fairness)")
print()

# Equalized Odds
odds_metrics = equalized_odds(y_pred, y_test, groups)
print("Equalized Odds:")
for name, metrics in zip(group_names, odds_metrics):
    print(f"  {name}: TPR={metrics['tpr']:.2%}, FPR={metrics['fpr']:.2%}")
print()

# ============================================================================
# Part 4: Mitigating Bias
# ============================================================================
print("=" * 70)
print("Part 4: Mitigating Bias")
print("=" * 70)
print()

# Strategy 1: Remove protected attribute (age) from training
print("Strategy 1: Remove Protected Attribute")
print("  Training model without age feature...")
X_train_no_age = X_train[:, :2]  # Only experience and education
X_test_no_age = X_test[:, :2]

model_no_age = LogisticRegression(max_iter=1000)
model_no_age.fit(X_train_no_age, y_train)
y_pred_no_age = model_no_age.predict(X_test_no_age)

# Check if bias is reduced
parity_rates_no_age = demographic_parity(y_pred_no_age, groups)
disparity_no_age = max(parity_rates_no_age) - min(parity_rates_no_age)

print(f"  Accuracy: {accuracy_score(y_test, y_pred_no_age):.2%}")
print(f"  Disparity: {disparity_no_age:.2%} (was {disparity:.2%})")
print()

# Strategy 2: Balanced sampling (oversample underrepresented groups)
print("Strategy 2: Balanced Sampling")
print("  Oversampling underrepresented groups...")

# Identify underrepresented group (older candidates)
old_indices = np.where(y_train == 1)[0]
old_age_train = X_train[:, 2]
old_candidates = old_indices[old_age_train[old_indices] >= 50]

if len(old_candidates) > 0:
    # Oversample old candidates
    n_oversample = len(y_train) // 3
    oversample_indices = np.random.choice(old_candidates, n_oversample, replace=True)
    
    X_train_balanced = np.vstack([X_train, X_train[oversample_indices]])
    y_train_balanced = np.hstack([y_train, y_train[oversample_indices]])
    
    model_balanced = LogisticRegression(max_iter=1000)
    model_balanced.fit(X_train_balanced[:, :2], y_train_balanced)  # Still remove age
    y_pred_balanced = model_balanced.predict(X_test_no_age)
    
    parity_rates_balanced = demographic_parity(y_pred_balanced, groups)
    disparity_balanced = max(parity_rates_balanced) - min(parity_rates_balanced)
    
    print(f"  Accuracy: {accuracy_score(y_test, y_pred_balanced):.2%}")
    print(f"  Disparity: {disparity_balanced:.2%}")
    print()

# ============================================================================
# Part 5: Model Interpretability
# ============================================================================
print("=" * 70)
print("Part 5: Model Interpretability")
print("=" * 70)
print()

# Feature importance analysis
print("Feature Importance Analysis:")
print("  Understanding which features the model relies on most")
print()

# Get model coefficients (feature weights)
if hasattr(model, 'coef_'):
    coef = model.coef_[0]
    feature_names = ['Experience', 'Education', 'Age']
    
    print("Model Coefficients (Feature Weights):")
    for name, weight in zip(feature_names, coef):
        print(f"  {name}: {weight:.4f}")
    print()
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    colors = ['green', 'blue', 'red']
    bars = plt.bar(feature_names, np.abs(coef), color=colors)
    plt.ylabel('Absolute Weight', fontsize=12)
    plt.title('Feature Importance (Absolute Weights)', fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    # Highlight age (potential bias source)
    bars[2].set_edgecolor('red')
    bars[2].set_linewidth(3)
    
    plt.subplot(1, 2, 2)
    # Show signed weights
    colors_signed = ['green' if w > 0 else 'red' for w in coef]
    bars = plt.bar(feature_names, coef, color=colors_signed, alpha=0.7)
    plt.ylabel('Weight', fontsize=12)
    plt.title('Feature Weights (Signed)', fontweight='bold')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom' if height > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    print("Saved: feature_importance.png")
    print()

# ============================================================================
# Part 6: Responsible AI Practices
# ============================================================================
print("=" * 70)
print("Part 6: Responsible AI Practices")
print("=" * 70)
print()

print("Key Principles for Responsible AI:")
print()
print("1. FAIRNESS")
print("   - Ensure models don't discriminate against protected groups")
print("   - Monitor for bias in predictions")
print("   - Use fairness metrics (demographic parity, equalized odds)")
print()

print("2. TRANSPARENCY")
print("   - Document model decisions and limitations")
print("   - Make models interpretable when possible")
print("   - Explain how models work to stakeholders")
print()

print("3. ACCOUNTABILITY")
print("   - Take responsibility for model outcomes")
print("   - Have human oversight for critical decisions")
print("   - Establish processes for addressing issues")
print()

print("4. PRIVACY")
print("   - Protect sensitive data")
print("   - Don't use protected attributes inappropriately")
print("   - Follow data protection regulations (GDPR, etc.)")
print()

print("5. ROBUSTNESS")
print("   - Test models on diverse data")
print("   - Handle edge cases gracefully")
print("   - Monitor model performance over time")
print()

print("6. HUMAN-CENTERED DESIGN")
print("   - Consider impact on people")
print("   - Involve diverse stakeholders")
print("   - Design for human well-being")
print()

# ============================================================================
# Part 7: Visualization
# ============================================================================
print("=" * 70)
print("Part 7: Visualizing Bias and Fairness")
print("=" * 70)
print()

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Hiring rates by age group
ax = axes[0, 0]
x_pos = np.arange(len(group_names))
width = 0.35
ax.bar(x_pos - width/2, [young_actual, middle_actual, old_actual], 
       width, label='Actual', alpha=0.7, color='blue')
ax.bar(x_pos + width/2, parity_rates, 
       width, label='Predicted', alpha=0.7, color='orange')
ax.set_xlabel('Age Group', fontsize=12)
ax.set_ylabel('Hiring Rate', fontsize=12)
ax.set_title('Hiring Rates: Actual vs Predicted', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(group_names)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 2. Demographic parity comparison
ax = axes[0, 1]
ax.bar(group_names, parity_rates, alpha=0.7, color=['green', 'yellow', 'red'])
ax.axhline(y=np.mean(parity_rates), color='black', linestyle='--', 
           label=f'Average: {np.mean(parity_rates):.2%}')
ax.set_ylabel('Positive Prediction Rate', fontsize=12)
ax.set_title('Demographic Parity Analysis', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 3. Equalized odds (TPR and FPR)
ax = axes[1, 0]
tpr_values = [m['tpr'] for m in odds_metrics]
fpr_values = [m['fpr'] for m in odds_metrics]
x_pos = np.arange(len(group_names))
width = 0.35
ax.bar(x_pos - width/2, tpr_values, width, label='TPR (Recall)', 
       alpha=0.7, color='green')
ax.bar(x_pos + width/2, fpr_values, width, label='FPR', 
       alpha=0.7, color='red')
ax.set_xlabel('Age Group', fontsize=12)
ax.set_ylabel('Rate', fontsize=12)
ax.set_title('Equalized Odds Analysis', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(group_names)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 4. Disparity comparison
ax = axes[1, 1]
strategies = ['Original\nModel', 'Remove\nAge', 'Balanced\nSampling']
disparities = [disparity, disparity_no_age, disparity_balanced if 'disparity_balanced' in locals() else disparity]
colors_strat = ['red', 'orange', 'green']
bars = ax.bar(strategies, disparities, color=colors_strat, alpha=0.7)
ax.set_ylabel('Disparity (Lower is Better)', fontsize=12)
ax.set_title('Bias Mitigation Strategies', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2%}',
            ha='center', va='bottom')

plt.tight_layout()
plt.savefig('bias_fairness_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: bias_fairness_analysis.png")
print()

print("=" * 70)
print("Step 12 Complete!")
print("=" * 70)
print()
print("Summary:")
print("  ✅ Understood bias in AI systems")
print("  ✅ Detected bias in model predictions")
print("  ✅ Calculated fairness metrics (demographic parity, equalized odds)")
print("  ✅ Applied bias mitigation strategies")
print("  ✅ Analyzed model interpretability")
print("  ✅ Learned responsible AI practices")
print()
print("Key Takeaways:")
print("  • AI models can perpetuate bias from training data")
print("  • Fairness metrics help detect and measure bias")
print("  • Multiple strategies exist to mitigate bias")
print("  • Responsible AI requires ongoing monitoring and evaluation")
print()
