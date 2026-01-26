"""
Project 1 - Part 2: Email Spam Classifier
Classify emails as spam or not spam using Logistic Regression

This project applies concepts from Steps 2-3:
- Binary classification
- Logistic regression
- Probability-based decisions
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import plotting utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from plotting import plot_sigmoid_function, plot_probability_curve, plot_learning_curve, plot_confusion_matrix_style

print("=" * 70)
print("Project 1 - Part 2: Email Spam Classifier")
print("=" * 70)
print()

# ============================================================================
# Step 1: Create Synthetic Email Dataset
# ============================================================================
print("Step 1: Creating Email Dataset")
print("-" * 70)

np.random.seed(42)

# Generate synthetic email features
num_emails = 100

# Feature 1: Number of exclamation marks
exclamations = np.random.poisson(3, num_emails)  # Spam has more
exclamations = np.where(exclamations > 10, 10, exclamations)  # Cap at 10

# Feature 2: Number of ALL CAPS words
all_caps = np.random.poisson(2, num_emails)  # Spam has more
all_caps = np.where(all_caps > 8, 8, all_caps)

# Feature 3: Contains "free" or "click" (binary)
contains_keywords = np.random.binomial(1, 0.3, num_emails)

# Combine features
X = np.column_stack([exclamations, all_caps, contains_keywords])

# Generate labels (spam = 1, not spam = 0)
# Spam emails tend to have more exclamations, caps, and keywords
spam_score = (exclamations * 0.4 + all_caps * 0.3 + contains_keywords * 0.3)
spam_threshold = np.percentile(spam_score, 50)  # Top 50% are spam
y = (spam_score >= spam_threshold).astype(int)

# Add some noise (not all spam follows the pattern)
noise = np.random.binomial(1, 0.1, num_emails)
y = np.where(noise == 1, 1 - y, y)  # Flip 10% of labels

print(f"Created {num_emails} emails")
print(f"Spam emails: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
print(f"Not spam: {num_emails - np.sum(y)} ({100 - np.mean(y)*100:.1f}%)")
print()
print("Feature statistics:")
print(f"  Exclamations: {exclamations.mean():.1f} avg, {exclamations.max()} max")
print(f"  ALL CAPS words: {all_caps.mean():.1f} avg, {all_caps.max()} max")
print(f"  Contains keywords: {contains_keywords.mean()*100:.1f}%")
print()

# ============================================================================
# Step 2: Visualize the Data
# ============================================================================
print("Step 2: Visualizing Data")
print("-" * 70)

# Plot exclamations vs caps, colored by spam/not spam
plt.figure(figsize=(10, 6))
colors = ['red' if label == 1 else 'green' for label in y]
plt.scatter(exclamations, all_caps, c=colors, s=100, alpha=0.6, edgecolors='black')
plt.xlabel("Number of Exclamation Marks", fontsize=12, fontweight='bold')
plt.ylabel("Number of ALL CAPS Words", fontsize=12, fontweight='bold')
plt.title("Email Features: Spam (Red) vs Not Spam (Green)", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(['Spam', 'Not Spam'])
plt.show()

print("Observation: Spam emails tend to have more exclamations and caps")
print()

# ============================================================================
# Step 3: Understand Sigmoid Function
# ============================================================================
print("Step 3: Understanding Sigmoid Function")
print("-" * 70)

plot_sigmoid_function()
print("Sigmoid converts any score into a probability (0 to 1)")
print()

# ============================================================================
# Step 4: Train Logistic Regression Model
# ============================================================================
print("Step 4: Training Logistic Regression")
print("-" * 70)

def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to avoid overflow

def binary_cross_entropy(y, y_pred):
    """Binary cross-entropy loss"""
    epsilon = 1e-9
    return -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))

# Initialize weights (one per feature + bias)
w = np.random.randn(X.shape[1])  # 3 weights for 3 features
b = 0.0

# Training parameters
learning_rate = 0.1
epochs = 2000
losses = []

print("Training logistic regression model...")
print(f"Features: {X.shape[1]} (exclamations, ALL CAPS, keywords)")
print(f"Learning rate: {learning_rate}")
print(f"Epochs: {epochs}")
print()

# Training loop
for epoch in range(epochs):
    # Forward pass
    z = X @ w + b
    y_pred = sigmoid(z)
    
    # Calculate loss
    loss = binary_cross_entropy(y, y_pred)
    losses.append(loss)
    
    # Calculate gradients
    dw = np.mean((y_pred - y).reshape(-1, 1) * X, axis=0)
    db = np.mean(y_pred - y)
    
    # Update weights
    w -= learning_rate * dw
    b -= learning_rate * db
    
    # Print progress
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

print()
print("Training complete!")
print(f"Learned weights: {w}")
print(f"Learned bias: {b:.4f}")
print()

# ============================================================================
# Step 5: Visualize Training Progress
# ============================================================================
print("Step 5: Training Progress")
print("-" * 70)

plot_learning_curve(losses,
                   title="Spam Classifier - Learning Curve",
                   ylabel="Loss (Binary Cross-Entropy)")

# ============================================================================
# Step 6: Make Predictions
# ============================================================================
print("Step 6: Making Predictions")
print("-" * 70)

# Predict on training data
z = X @ w + b
probabilities = sigmoid(z)
predictions = (probabilities >= 0.5).astype(int)

# Calculate accuracy
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy*100:.1f}%")
print()

# Show some example predictions
print("Sample Predictions:")
print("-" * 70)
print("Exclamations | ALL CAPS | Keywords | Probability | Prediction | Actual")
print("-" * 70)
for i in range(min(10, len(X))):
    exclam = int(X[i, 0])
    caps = int(X[i, 1])
    keywords = int(X[i, 2])
    prob = probabilities[i]
    pred = "SPAM" if predictions[i] == 1 else "NOT SPAM"
    actual = "SPAM" if y[i] == 1 else "NOT SPAM"
    match = "✓" if predictions[i] == y[i] else "✗"
    print(f"     {exclam:2d}     |    {caps:2d}    |    {keywords:1d}     |    {prob:.3f}    |  {pred:8s}  | {actual:8s} {match}")
print()

# ============================================================================
# Step 7: Visualize Predictions
# ============================================================================
print("Step 7: Predictions Visualization")
print("-" * 70)

plot_confusion_matrix_style(y, predictions,
                           class_names=['Not Spam (0)', 'Spam (1)'])

# ============================================================================
# Step 8: Probability Curve (for single feature)
# ============================================================================
print("Step 8: Probability Analysis")
print("-" * 70)

# Visualize how probability changes with exclamation marks
# (using average values for other features)
exclam_range = np.linspace(0, 10, 100)
avg_caps = all_caps.mean()
avg_keywords = contains_keywords.mean()

# Create feature matrix for visualization
X_viz = np.column_stack([
    exclam_range,
    np.full_like(exclam_range, avg_caps),
    np.full_like(exclam_range, avg_keywords)
])

z_viz = X_viz @ w + b
probs_viz = sigmoid(z_viz)

plot_probability_curve(exclam_range, probs_viz,
                      X=exclamations[:20], y=y[:20],  # Show some data points
                      xlabel="Number of Exclamation Marks",
                      ylabel="Spam Probability",
                      title="Spam Probability vs Exclamation Marks")

# ============================================================================
# Step 9: Evaluate Model Performance
# ============================================================================
print("Step 9: Model Evaluation")
print("-" * 70)

# Calculate detailed metrics
true_positives = np.sum((predictions == 1) & (y == 1))
true_negatives = np.sum((predictions == 0) & (y == 0))
false_positives = np.sum((predictions == 1) & (y == 0))
false_negatives = np.sum((predictions == 0) & (y == 1))

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("Detailed Metrics:")
print(f"  Accuracy: {accuracy*100:.1f}%")
print(f"  Precision: {precision:.3f} (of predicted spam, how many are actually spam)")
print(f"  Recall: {recall:.3f} (of actual spam, how many did we catch)")
print(f"  F1-Score: {f1_score:.3f} (harmonic mean of precision and recall)")
print()
print("Confusion Matrix:")
print(f"  True Positives (correctly identified spam): {true_positives}")
print(f"  True Negatives (correctly identified not spam): {true_negatives}")
print(f"  False Positives (not spam labeled as spam): {false_positives}")
print(f"  False Negatives (spam labeled as not spam): {false_negatives}")
print()

# ============================================================================
# Step 10: Classify New Emails
# ============================================================================
print("Step 10: Classifying New Emails")
print("-" * 70)

# New emails to classify
new_emails = np.array([
    [2, 1, 0],   # Email 1: Few exclamations, few caps, no keywords
    [8, 5, 1],   # Email 2: Many exclamations, many caps, has keywords
    [1, 0, 0],   # Email 3: Very few exclamations, no caps, no keywords
    [9, 7, 1],   # Email 4: Lots of exclamations, lots of caps, has keywords
])

print("New Emails to Classify:")
print("-" * 70)
for i, email in enumerate(new_emails):
    z_new = email @ w + b
    prob = sigmoid(z_new)
    pred = "SPAM" if prob >= 0.5 else "NOT SPAM"
    confidence = prob if prob >= 0.5 else 1 - prob
    
    print(f"Email {i+1}:")
    print(f"  Features: {email[0]} exclamations, {email[1]} ALL CAPS, {email[2]} keywords")
    print(f"  Spam probability: {prob:.1%}")
    print(f"  Prediction: {pred} (confidence: {confidence:.1%})")
    print()

print("=" * 70)
print("Project 1 - Part 2 Complete!")
print("You've built a spam classifier using logistic regression!")
print("=" * 70)
