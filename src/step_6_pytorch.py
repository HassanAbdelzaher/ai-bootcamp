"""
Step 6 — PyTorch (From Scratch to Real AI)
Goal: Move from building neural networks manually to using a professional AI framework 
(PyTorch) while understanding what it automates.
Tools: Python + PyTorch + Matplotlib
"""

# Import numpy first to ensure proper initialization before PyTorch
import numpy as np
import warnings

# Suppress NumPy initialization warnings (common with PyTorch)
warnings.filterwarnings('ignore', message='.*Failed to initialize NumPy.*')

import torch
import torch.nn as nn
import torch.optim as optim
from plotting import plot_learning_curve

# Check if PyTorch is available
try:
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
except ImportError:
    print("ERROR: PyTorch is not installed!")
    print("Install with: pip install torch torchvision torchaudio")
    exit(1)
print()

# 6.3 First PyTorch Tensor
print("=== 6.3 First PyTorch Tensor ===")
x = torch.tensor([1.0, 2.0, 3.0])
print("Tensor:", x)
print("Type:", type(x))
print()

# 6.4 Autograd (Automatic Gradients)
print("=== 6.4 Autograd (Automatic Gradients) ===")
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x

y.backward()

print("x =", x.item())
print("y = x² + 3x =", y.item())
print("dy/dx =", x.grad.item())
print("(Manual calculation: 2x + 3 =", 2*2 + 3, ")")
print()

# 6.5 Dataset Example (XOR Again)
print("=== 6.5 Dataset Example (XOR) ===")
X = torch.tensor([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]])

y = torch.tensor([[0.],
                  [1.],
                  [1.],
                  [0.]])

print("Input X:")
print(X)
print("Target y:")
print(y)
print()

# 6.6 Defining a Neural Network
print("=== 6.6 Defining a Neural Network ===")
model = nn.Sequential(
    nn.Linear(2, 4),   # input → hidden
    nn.ReLU(),         # activation
    nn.Linear(4, 1),   # hidden → output
    nn.Sigmoid()       # probability
)

print("Model architecture:")
print(model)
print()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
print()

# 6.7 Loss Function & Optimizer
print("=== 6.7 Loss Function & Optimizer ===")
loss_fn = nn.BCELoss()  # Binary Cross-Entropy
optimizer = optim.SGD(model.parameters(), lr=0.1)

print("Loss function: Binary Cross-Entropy")
print("Optimizer: SGD (Stochastic Gradient Descent)")
print("Learning rate: 0.1")
print()

# 6.8 Training Loop
print("=== 6.8 Training Loop ===")
losses = []

for epoch in range(3000):
    # Forward pass
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    losses.append(loss.item())
    
    # Backward pass
    optimizer.zero_grad()  # Clear gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update weights
    
    # Print progress every 500 epochs
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}/{3000}, Loss: {loss.item():.4f}")

print("Training complete!")
print()

# 6.9 Learning Curve
print("=== 6.9 Learning Curve ===")
plot_learning_curve(losses, title="Training Loss (PyTorch)", ylabel="Loss")

# 6.10 Final Predictions
print("=== 6.10 Final Predictions ===")
with torch.no_grad():  # Disable gradient computation for inference
    probs = model(X)
    preds = (probs >= 0.5).int()

print("Probabilities:")
print(probs)
print("\nPredictions:")
print(preds)
print("\nActual values:")
print(y.int())
print("\nAccuracy:", (preds == y.int()).float().mean().item())
print("✅ PyTorch solved XOR successfully!")
print()

# 6.11 Model Saving and Loading
print("=== 6.11 Model Saving and Loading ===")
print("Saving models allows you to:")
print("  - Reuse trained models without retraining")
print("  - Share models with others")
print("  - Deploy models to production")
print("  - Resume training from checkpoints")
print()

# Save the model
# Define file path where model will be saved
# .pth extension is standard for PyTorch checkpoint files
model_path = "xor_model.pth"

# torch.save() saves data to a file
# First argument: Dictionary containing what to save
#   - 'model_state_dict': All model weights and biases (learned parameters)
#   - 'optimizer_state_dict': Optimizer state (learning rate, momentum, etc.)
#   - 'loss': Final training loss (for reference)
# Second argument: File path where to save
torch.save({
    'model_state_dict': model.state_dict(),        # model.state_dict() returns dict of all weights
    'optimizer_state_dict': optimizer.state_dict(), # optimizer.state_dict() returns optimizer state
    'loss': losses[-1],                             # losses[-1] gets last element (final loss)
}, model_path)

print(f"✅ Model saved to {model_path}")
print(f"   - Model weights: {model_path}")
# Calculate file size: sum of all parameters * 4 bytes (float32) / 1024 (KB)
# p.numel() returns number of elements in parameter tensor
# 4 bytes per float32 parameter
# / 1024 converts bytes to KB
# {:.1f} formats to 1 decimal place
print(f"   - File size: ~{sum(p.numel() * 4 for p in model.parameters()) / 1024:.1f} KB")
print()

# Create a new model instance
print("Creating a new model instance (untrained)...")
# Create a fresh model with same architecture as original
# This model has random (untrained) weights
new_model = nn.Sequential(
    nn.Linear(2, 4),   # Input layer: 2 features → 4 neurons
    nn.ReLU(),         # Activation function
    nn.Linear(4, 1),   # Hidden layer: 4 neurons → 1 output
    nn.Sigmoid()       # Output activation (probability)
)

# Test new model (should be random)
# torch.no_grad() disables gradient computation (faster, uses less memory)
# Use this during inference (prediction) when you don't need gradients
with torch.no_grad():
    # Make predictions with untrained model
    new_probs = new_model(X)  # Get probabilities (should be random ~0.5)
    # Convert probabilities to binary predictions (>= 0.5 = 1, else 0)
    # .int() converts True/False to 1/0
    new_preds = (new_probs >= 0.5).int()
    # Calculate accuracy: compare predictions with actual labels
    # (new_preds == y.int()) creates boolean tensor (True where match)
    # .float() converts True/False to 1.0/0.0
    # .mean() calculates average (accuracy)
    # .item() extracts scalar value from tensor
    new_accuracy = (new_preds == y.int()).float().mean().item()

# Print accuracy (should be around 50% - random guessing)
# {new_accuracy:.2%} formats as percentage with 2 decimal places
print(f"New model accuracy (before loading): {new_accuracy:.2%}")
print("  This is random because the model hasn't been trained yet")
print()

# Load the saved model
print(f"Loading model from {model_path}...")
# torch.load() loads data from file
# Returns the dictionary we saved earlier
checkpoint = torch.load(model_path)

# Load model weights into new model
# checkpoint['model_state_dict'] gets the saved weights dictionary
# load_state_dict() copies weights from dictionary into model
new_model.load_state_dict(checkpoint['model_state_dict'])
print("✅ Model loaded successfully!")
print()

# Test loaded model (should match original)
# Now the model should have same weights as original trained model
with torch.no_grad():
    # Make predictions with loaded model
    loaded_probs = new_model(X)  # Get probabilities (should match original)
    # Convert to binary predictions
    loaded_preds = (loaded_probs >= 0.5).int()
    # Calculate accuracy (should be 100% like original)
    loaded_accuracy = (loaded_preds == y.int()).float().mean().item()

# Print accuracy (should be 100% - same as original trained model)
print(f"Loaded model accuracy: {loaded_accuracy:.2%}")
print("  This matches the original trained model!")
print()

# Clean up
# Remove the saved model file to keep workspace clean
import os  # Import os module for file operations
# os.path.exists() checks if file exists
if os.path.exists(model_path):
    # os.remove() deletes the file
    os.remove(model_path)
    print(f"Cleaned up: Removed {model_path}")
print()

# 6.12 Comparing Scratch vs PyTorch
print("=== 6.12 Comparing Scratch vs PyTorch ===")
print("| From Scratch          | PyTorch              |")
print("|----------------------|----------------------|")
print("| Manual gradients     | Automatic            |")
print("| More code            | Less code            |")
print("| Easy to make mistakes| Safer                |")
print("| Best for learning    | Best for real projects|")
print()
print("🧠 Rule: Learn from scratch → build with PyTorch")
print()

# Mini Projects Ideas
print("=== 6.13 Mini Projects Ideas ===")
print("Project 1: Pass / Fail predictor (study hours)")
print("Project 2: Grade classifier (A/B/C)")
print("Project 3: Student performance predictor (multi-feature)")
print()

# Common Beginner Mistakes
print("=== 6.14 Common Beginner Mistakes ===")
print("❌ Using PyTorch without understanding math")
print("❌ Forgetting optimizer.zero_grad()")
print("❌ Mixing NumPy and Torch tensors")
print("❌ Ignoring loss curves")
print()

# Final Checklist
print("=== 6.15 Final Checklist (Bootcamp Complete 🎓) ===")
print("Students can:")
print("✅ Explain neurons mathematically")
print("✅ Train models from scratch")
print("✅ Use PyTorch correctly")
print("✅ Read AI code confidently")
print("✅ Build simple AI projects")
print()

print("🎉 Congratulations!")
print("You have completed:")
print("  - Step 0 → Math Foundations")
print("  - Step 1 → Linear Regression")
print("  - Step 2 → Perceptron")
print("  - Step 3 → Logistic Regression")
print("  - Step 4 → Neural Network Layers")
print("  - Step 5 → XOR & Hidden Layers")
print("  - Step 6 → PyTorch")
print()
print("🚀 You are now a Junior AI Engineer!")
print()
print("What's Next?")
print("  - CNNs (images)")
print("  - RNNs (text, sequences)")
print("  - Real datasets (CSV, images)")
print("  - AI projects & competitions")
