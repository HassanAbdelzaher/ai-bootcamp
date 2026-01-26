"""
Step 0 — Math Foundations for AI
Goal: Build the math "language" behind AI so students can understand 
perceptrons, neural networks, and training later.
Tools: Python + NumPy + Matplotlib
"""

# Import NumPy for numerical operations (arrays, matrices, math functions)
# NumPy is the foundation for all AI/ML work in Python
import numpy as np

# Import our custom plotting function for visualizing feature contributions
from plotting import plot_feature_contributions

# ============================================================================
# 0.3 Numbers & Features
# ============================================================================
# Features are the characteristics or measurements we use to make predictions.
# In AI, everything starts with numbers - these are our "features"
print("=== 0.3 Numbers & Features ===")

# Example features from different domains:
study_hours = 4          # Integer feature: whole number of hours studied
math_score = 80          # Integer feature: test score (0-100 scale)
temperature = 28.5       # Float feature: decimal number (Celsius)
pixel_brightness = 0.92  # Float feature: normalized value (0.0 to 1.0)

# Print all features to see their values
print(study_hours, math_score, temperature, pixel_brightness)
print()

# ============================================================================
# 0.4 Vectors
# ============================================================================
# A vector is a collection of numbers in a specific order.
# Think of it as a list of features for one data point (e.g., one student)
print("=== 0.4 Vectors ===")

# Create a vector using np.array() - this is a NumPy array
# This vector represents one student's scores in three subjects
# [Math, Science, English]
x = np.array([80, 70, 75])

# Why use np.array() instead of a regular Python list?
# - Faster mathematical operations
# - Supports vectorized operations (applies operations to all elements at once)
# - Essential for AI/ML computations

print("Student vector:", x)
print("  This represents one student with scores: Math=80, Science=70, English=75")
print()

# ============================================================================
# 0.5 Weights
# ============================================================================
# Weights determine how important each feature is in the final decision.
# Higher weight = more important feature
print("=== 0.5 Weights ===")

# Create a weight vector - each weight corresponds to one feature
# [Math weight, Science weight, English weight]
# Note: These weights sum to 1.0 (0.6 + 0.3 + 0.1 = 1.0)
w = np.array([0.6, 0.3, 0.1])

print("Weights:", w)
print("  Math is most important (0.6 = 60%)")
print("  Science is moderately important (0.3 = 30%)")
print("  English is least important (0.1 = 10%)")
print()

# ============================================================================
# 0.6 Dot Product
# ============================================================================
# The dot product multiplies corresponding elements and sums them up.
# This is the core operation in neural networks!
print("=== 0.6 Dot Product ===")

# Re-define our vectors (in case they were modified)
x = np.array([80, 70, 75])  # Student's scores
w = np.array([0.6, 0.3, 0.1])  # Weights (importance)

# Calculate dot product: z = x · w
# This is equivalent to: z = (80 × 0.6) + (70 × 0.3) + (75 × 0.1)
#                      = 48 + 21 + 7.5
#                      = 76.5
z = np.dot(x, w)

# Alternative syntax: z = x @ w (matrix multiplication operator)
# Both np.dot(x, w) and x @ w do the same thing

print("Dot product score:", z)
print("  This is the weighted sum of all features")
print("  It combines all information into a single number")
print()

# Visualize how each feature contributes to the final score
# This creates a bar chart and pie chart showing feature contributions
features = np.array([80, 70, 75])
weights = np.array([0.6, 0.3, 0.1])
plot_feature_contributions(features, weights, labels=["Math", "Science", "English"])

# ============================================================================
# 0.7 Bias
# ============================================================================
# Bias is a constant value added to adjust the final score.
# It's like a starting point or threshold adjustment.
print("=== 0.7 Bias ===")

# Set a bias value (can be positive or negative)
# Negative bias makes it harder to get a high score
b = -10

# Add bias to the dot product result
# Final score = weighted sum + bias
z_with_bias = z + b

print("Score without bias:", z)
print("Bias value:", b)
print("Score with bias:", z_with_bias)
print("  Bias adjusts the final score up or down")
print("  Negative bias = harder to pass, Positive bias = easier to pass")
print()

# ============================================================================
# 0.8 Mini Neuron
# ============================================================================
# A neuron is the basic building block of AI.
# It takes inputs, applies weights and bias, and produces an output.
print("=== 0.8 Mini Neuron ===")

def neuron(x, w, b):
    """
    Simple neuron function that calculates: z = x · w + b
    
    Parameters:
    x: input features (vector) - e.g., [math_score, science_score, english_score]
    w: weights (vector) - e.g., [0.6, 0.3, 0.1]
    b: bias (scalar) - e.g., -10
    
    Returns:
    z: output score (scalar) - a single number
    """
    # Step 1: Calculate dot product (weighted sum of features)
    # np.dot(x, w) multiplies each feature by its weight and sums them
    dot_product = np.dot(x, w)
    
    # Step 2: Add bias to the dot product
    # This adjusts the final score
    result = dot_product + b
    
    return result

# Test the neuron function with our data
result = neuron(x, w, b)
print("Neuron output:", result)
print("  This is the same as: z = x · w + b")
print("  This single calculation is the foundation of all neural networks!")
print()

# ============================================================================
# 0.9 Decision Boundary
# ============================================================================
# After calculating the score, we make a decision based on a threshold.
# This converts a continuous score into a binary decision (pass/fail).
print("=== 0.9 Decision Boundary ===")

# Set a threshold - scores above this value mean "pass"
threshold = 60

# Make a decision: 1 if score >= threshold (pass), 0 otherwise (fail)
# This is a simple if-else statement
decision = 1 if z_with_bias >= threshold else 0

print("Score:", z_with_bias)
print("Threshold:", threshold)
print("Decision:", decision, "(1=Pass, 0=Fail)")
print("  If score >= threshold → Pass (1)")
print("  If score < threshold → Fail (0)")
print()

# ============================================================================
# 0.10 Multiple Students
# ============================================================================
# In real AI, we process many data points at once using matrix operations.
# This is much faster than processing one at a time!
print("=== 0.10 Multiple Students ===")

# Create a matrix where each row is one student's features
# X is a 2D array: (number of students, number of features)
X = np.array([
    [90, 85, 70],  # Student 1: Math=90, Science=85, English=70
    [40, 50, 60],  # Student 2: Math=40, Science=50, English=60
    [75, 70, 80],  # Student 3: Math=75, Science=70, English=80
    [55, 60, 58],  # Student 4: Math=55, Science=60, English=58
])

# Matrix multiplication: X @ w
# This calculates the dot product for EACH row (student) at once!
# Result: [score_student1, score_student2, score_student3, score_student4]
# 
# How it works:
# - For each row in X, calculate dot product with w
# - Then add bias to each result
# - This is MUCH faster than a loop!

scores = X @ w + b  # Matrix multiplication + broadcasting

print("Students matrix (each row is one student):")
print(X)
print(f"\nWeights: {w}")
print(f"Bias: {b}")
print("\nScores for all students:")
print(scores)
print()
print("  Matrix multiplication processes all students at once!")
print("  This is the power of NumPy - fast, efficient, and readable")
print()

# ============================================================================
# Exercises
# ============================================================================
print("=== Exercises ===")
print("Try changing weights and bias values to observe their impact!")
print("Implement a pass/fail function based on the threshold.")
print()
print("Key concepts learned:")
print("  ✓ Features: Input data (numbers)")
print("  ✓ Vectors: Collections of features")
print("  ✓ Weights: Importance of each feature")
print("  ✓ Dot Product: Combining features with weights")
print("  ✓ Bias: Constant adjustment")
print("  ✓ Neuron: z = x · w + b (the foundation of AI!)")
print("  ✓ Decision: Using threshold to make binary choices")
print("  ✓ Matrix Operations: Processing multiple examples efficiently")