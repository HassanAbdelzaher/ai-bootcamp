"""
Project 1 - Part 1: House Price Predictor
Predict house prices using Linear Regression

This project applies concepts from Steps 0-1:
- Linear regression
- Gradient descent
- Model evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import plotting utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from plotting import plot_data_scatter, plot_prediction_line, plot_learning_curve, plot_error_distribution

print("=" * 70)
print("Project 1 - Part 1: House Price Predictor")
print("=" * 70)
print()

# ============================================================================
# Step 1: Create Synthetic Dataset
# ============================================================================
print("Step 1: Creating Dataset")
print("-" * 70)

# Generate synthetic house data
np.random.seed(42)  # For reproducibility

# Features
num_houses = 50
house_sizes = np.random.uniform(800, 3000, num_houses)  # sq ft
bedrooms = np.random.randint(1, 5, num_houses)
house_ages = np.random.uniform(0, 30, num_houses)  # years

# Create price based on features (with some noise)
# Price = 50 * size + 10000 * bedrooms - 500 * age + 20000 + noise
base_price = 20000
prices = (50 * house_sizes + 
          10000 * bedrooms - 
          500 * house_ages + 
          base_price + 
          np.random.normal(0, 5000, num_houses))  # Add noise

# Ensure prices are positive
prices = np.maximum(prices, 30000)

print(f"Created {num_houses} houses")
print(f"Size range: {house_sizes.min():.0f} - {house_sizes.max():.0f} sq ft")
print(f"Bedrooms: {bedrooms.min()} - {bedrooms.max()}")
print(f"Price range: ${prices.min():,.0f} - ${prices.max():,.0f}")
print()

# ============================================================================
# Step 2: Visualize the Data
# ============================================================================
print("Step 2: Visualizing Data")
print("-" * 70)

# Plot house size vs price
plot_data_scatter(house_sizes, prices, 
                 xlabel="House Size (sq ft)", 
                 ylabel="Price ($)",
                 title="House Size vs Price")

print("Observation: Larger houses tend to cost more (positive correlation)")
print()

# ============================================================================
# Step 3: Prepare Data for Training
# ============================================================================
print("Step 3: Preparing Data")
print("-" * 70)

# For simplicity, start with just house size as feature
# Later, you can add more features (bedrooms, age)
X = house_sizes.reshape(-1, 1)  # Reshape to (n_samples, n_features)
y = prices

# Normalize features (optional but helps with training)
X_mean = X.mean()
X_std = X.std()
X_normalized = (X - X_mean) / X_std

print(f"Feature shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature mean: {X_mean:.2f}, std: {X_std:.2f}")
print()

# ============================================================================
# Step 4: Train Linear Regression Model
# ============================================================================
print("Step 4: Training Model")
print("-" * 70)

# Initialize weights
w = np.random.randn(1)  # Weight for house size
b = np.random.randn()   # Bias

# Training parameters
learning_rate = 0.01
epochs = 1000
losses = []

print("Training linear regression model...")
print(f"Learning rate: {learning_rate}")
print(f"Epochs: {epochs}")
print()

# Training loop
for epoch in range(epochs):
    # Forward pass
    y_pred = w * X_normalized.flatten() + b
    
    # Calculate loss (MSE)
    loss = np.mean((y_pred - y) ** 2)
    losses.append(loss)
    
    # Calculate gradients
    dw = np.mean((y_pred - y) * X_normalized.flatten())
    db = np.mean(y_pred - y)
    
    # Update weights
    w -= learning_rate * dw
    b -= learning_rate * db
    
    # Print progress
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:,.0f}")

print()
print("Training complete!")
print(f"Final weight: {w[0]:.2f}")
print(f"Final bias: ${b:,.2f}")
print()

# ============================================================================
# Step 5: Visualize Training Progress
# ============================================================================
print("Step 5: Training Progress")
print("-" * 70)

plot_learning_curve(losses, 
                   title="House Price Prediction - Learning Curve",
                   ylabel="Loss (MSE)")

# ============================================================================
# Step 6: Make Predictions
# ============================================================================
print("Step 6: Making Predictions")
print("-" * 70)

# Predict on training data
y_pred = w * X_normalized.flatten() + b

# Plot predictions vs actual
plot_prediction_line(house_sizes, y, y_pred,
                    xlabel="House Size (sq ft)",
                    ylabel="Price ($)",
                    title="House Price Predictions",
                    label_pred="Predicted Prices",
                    color="green")

# Error analysis
plot_error_distribution(y, y_pred, 
                       title="House Price Prediction Errors")

# ============================================================================
# Step 7: Evaluate Model
# ============================================================================
print("Step 7: Model Evaluation")
print("-" * 70)

# Calculate metrics
mse = np.mean((y_pred - y) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_pred - y))

# R-squared (coefficient of determination)
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print("Performance Metrics:")
print(f"  Mean Squared Error (MSE): ${mse:,.0f}")
print(f"  Root Mean Squared Error (RMSE): ${rmse:,.0f}")
print(f"  Mean Absolute Error (MAE): ${mae:,.0f}")
print(f"  R-squared (R²): {r_squared:.3f}")
print()

# ============================================================================
# Step 8: Predict for New Houses
# ============================================================================
print("Step 8: Predictions for New Houses")
print("-" * 70)

# New houses to predict
new_houses = np.array([
    [1200],  # Small house
    [2000],  # Medium house
    [2800],  # Large house
])

# Normalize new data
new_houses_normalized = (new_houses - X_mean) / X_std

# Make predictions
predictions = w * new_houses_normalized.flatten() + b

print("Predictions:")
print("-" * 50)
for size, pred in zip(new_houses.flatten(), predictions):
    print(f"House size: {size:4.0f} sq ft → Predicted price: ${pred:,.0f}")
print()

# ============================================================================
# Step 9: Challenge - Add More Features
# ============================================================================
print("Step 9: Challenge - Multi-Feature Model")
print("-" * 70)
print("Try adding bedrooms and age as additional features!")
print("This will require:")
print("  - Multiple weights (one per feature)")
print("  - Matrix operations")
print("  - Better feature normalization")
print()
print("Expected improvement: Lower error, better predictions")
print()

print("=" * 70)
print("Project 1 - Part 1 Complete!")
print("Next: Implement spam_classifier.py for Part 2")
print("=" * 70)
