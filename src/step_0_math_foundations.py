"""
Step 0 — Math Foundations for AI
Goal: Build the math "language" behind AI so students can understand 
perceptrons, neural networks, and training later.
Tools: Python + NumPy + Matplotlib
"""

import numpy as np
from plotting import plot_feature_contributions

# 0.3 Numbers & Features
print("=== 0.3 Numbers & Features ===")
study_hours = 4
math_score = 80
temperature = 28.5
pixel_brightness = 0.92

print(study_hours, math_score, temperature, pixel_brightness)
print()

# 0.4 Vectors
print("=== 0.4 Vectors ===")
x = np.array([80, 70, 75])
print("Student vector:", x)
print()

# 0.5 Weights
print("=== 0.5 Weights ===")
w = np.array([0.6, 0.3, 0.1])
print("Weights:", w)
print()

# 0.6 Dot Product
print("=== 0.6 Dot Product ===")
x = np.array([80, 70, 75])
w = np.array([0.6, 0.3, 0.1])
z = np.dot(x, w)
print("Dot product score:", z)
print()

# Visualize feature contributions
features = np.array([80, 70, 75])
weights = np.array([0.6, 0.3, 0.1])
plot_feature_contributions(features, weights, labels=["Math", "Science", "English"])

# 0.7 Bias
print("=== 0.7 Bias ===")
b = -10
z_with_bias = z + b
print("Score with bias:", z_with_bias)
print()

# 0.8 Mini Neuron
print("=== 0.8 Mini Neuron ===")
def neuron(x, w, b):
    """Simple neuron: z = x · w + b"""
    return np.dot(x, w) + b

result = neuron(x, w, b)
print("Neuron output:", result)
print()

# 0.9 Decision Boundary
print("=== 0.9 Decision Boundary ===")
threshold = 60
decision = 1 if z_with_bias >= threshold else 0
print("Decision:", decision)
print()

# 0.10 Multiple Students
print("=== 0.10 Multiple Students ===")
X = np.array([
    [90, 85, 70],
    [40, 50, 60],
    [75, 70, 80],
    [55, 60, 58],
])

scores = X @ w + b
print("Scores for all students:")
print(scores)
print()

# Exercises
print("=== Exercises ===")
print("Try changing weights and bias values to observe their impact!")
print("Implement a pass/fail function based on the threshold.")
