"""
Project 5b: Regularization and Overfitting
Demonstrate overfitting and apply regularization techniques
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import sys
import os

# Add parent directory to path to import plotting utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from plotting import plot_overfitting, plot_learning_curve

print("=" * 70)
print("Project 5b: Regularization and Overfitting")
print("=" * 70)
print()

# ============================================================================
# Step 1: Create Dataset
# ============================================================================
print("=" * 70)
print("Step 1: Creating Dataset")
print("=" * 70)
print()

# Create classification dataset
# make_classification: Generate synthetic binary classification data
# n_samples=500: Total number of samples (smaller dataset = easier to overfit)
# n_features=20: Number of input features
# n_informative=10: Features that actually help classification
# n_redundant=5: Features that are linear combinations (redundant)
# n_classes=2: Binary classification
# random_state=42: For reproducibility
X, y = make_classification(
    n_samples=500,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

# Split into training and validation sets
# train_test_split: Randomly split data
# test_size=0.3: 30% for validation, 70% for training
# Validation set: Used to detect overfitting (not used for training)
# random_state=42: Ensures same split every time
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print()

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)

# ============================================================================
# Step 2: Model Architecture (Prone to Overfitting)
# ============================================================================
print("=" * 70)
print("Step 2: Building Model (Prone to Overfitting)")
print("=" * 70)
print()

class OverfittingModel(nn.Module):
    """Model designed to overfit"""
    def __init__(self, input_size=20, hidden_size=128, output_size=1):
        # super(): Call parent class constructor (required for PyTorch modules)
        super(OverfittingModel, self).__init__()
        
        # Large hidden layers (prone to overfitting)
        # More parameters = more capacity to memorize training data
        # input_size=20: Number of input features
        # hidden_size=128: Large hidden layer (many parameters)
        # This gives model high capacity, making overfitting likely
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # Multiple large hidden layers increase model complexity
        # Each layer has 128×128 = 16,384 parameters (plus biases)
        # Total: ~50,000+ parameters for small dataset = overfitting risk
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        
        # Output layer: Maps hidden features to single output (binary classification)
        self.fc4 = nn.Linear(hidden_size, output_size)
        
        # Activation functions
        # ReLU: Rectified Linear Unit (f(x) = max(0, x))
        # Used in hidden layers for non-linearity
        self.relu = nn.ReLU()
        
        # Sigmoid: Maps output to [0, 1] for binary classification
        # Output represents probability of positive class
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Forward pass through network
        # x shape: (batch, input_size) - input features
        
        # Layer 1: Input → Hidden
        # self.fc1(x): Linear transformation (x @ W1 + b1)
        # self.relu(...): Apply ReLU activation (introduces non-linearity)
        x = self.relu(self.fc1(x))
        
        # Layer 2: Hidden → Hidden
        # Further feature transformation
        x = self.relu(self.fc2(x))
        
        # Layer 3: Hidden → Hidden
        # More feature learning (increases model capacity)
        x = self.relu(self.fc3(x))
        
        # Layer 4: Hidden → Output
        # Final classification layer
        # self.sigmoid(...): Convert to probability [0, 1]
        x = self.sigmoid(self.fc4(x))
        
        # Return: (batch, 1) - probability for each sample
        return x

# ============================================================================
# Step 3: Train Without Regularization (Overfitting)
# ============================================================================
print("=" * 70)
print("Step 3: Training Without Regularization (Overfitting)")
print("=" * 70)
print()

# Create model instance
# OverfittingModel: Model with large capacity (prone to overfitting)
model_no_reg = OverfittingModel()

# Loss function: Binary Cross-Entropy Loss
# BCELoss: Appropriate for binary classification with sigmoid output
# Formula: -[y*log(p) + (1-y)*log(1-p)]
# Penalizes confident wrong predictions more
criterion = nn.BCELoss()

# Optimizer: Adam optimizer
# optim.Adam: Adaptive learning rate optimizer
# model_no_reg.parameters(): All trainable weights and biases
# lr=0.001: Learning rate (step size for weight updates)
optimizer_no_reg = optim.Adam(model_no_reg.parameters(), lr=0.001)

# Lists to track loss during training
# train_losses: Loss on training data (should decrease)
# val_losses: Loss on validation data (should decrease, then may increase if overfitting)
train_losses_no_reg = []
val_losses_no_reg = []

epochs = 500  # Number of training iterations
print("Training...")
for epoch in range(epochs):
    # ===== TRAINING PHASE =====
    # model.train(): Set model to training mode
    # Enables dropout, batch norm updates, etc.
    model_no_reg.train()
    
    # optimizer.zero_grad(): Clear gradients from previous iteration
    # PyTorch accumulates gradients, so we need to reset them
    optimizer_no_reg.zero_grad()
    
    # Forward pass: Make predictions on training data
    # X_train_tensor: Input features (batch, features)
    # train_outputs: Predicted probabilities (batch, 1)
    train_outputs = model_no_reg(X_train_tensor)
    
    # Calculate loss: Compare predictions with true labels
    # criterion(): Computes Binary Cross-Entropy loss
    # train_loss: Scalar value representing average error
    train_loss = criterion(train_outputs, y_train_tensor)
    
    # Backward pass: Compute gradients
    # loss.backward(): Calculates gradients of loss w.r.t. all parameters
    # Uses chain rule (backpropagation) to compute gradients
    train_loss.backward()
    
    # Update weights: Move weights in direction that reduces loss
    # optimizer.step(): Updates all parameters using computed gradients
    # Formula: weight = weight - lr × gradient
    optimizer_no_reg.step()
    
    # ===== VALIDATION PHASE =====
    # model.eval(): Set model to evaluation mode
    # Disables dropout, freezes batch norm, etc.
    model_no_reg.eval()
    
    # torch.no_grad(): Disable gradient computation (saves memory, faster)
    # Not needed for validation (we're not updating weights)
    with torch.no_grad():
        # Forward pass on validation data
        val_outputs = model_no_reg(X_val_tensor)
        
        # Calculate validation loss
        # This tells us how well model generalizes to unseen data
        val_loss = criterion(val_outputs, y_val_tensor)
    
    # Store losses for visualization
    # .item(): Extract scalar value from tensor
    train_losses_no_reg.append(train_loss.item())
    val_losses_no_reg.append(val_loss.item())
    
    if (epoch + 1) % 100 == 0:
        print(f"  Epoch {epoch+1}/{epochs}: Train={train_loss.item():.4f}, Val={val_loss.item():.4f}")

print()
print(f"Final Training Loss: {train_losses_no_reg[-1]:.4f}")
print(f"Final Validation Loss: {val_losses_no_reg[-1]:.4f}")
print(f"Gap: {val_losses_no_reg[-1] - train_losses_no_reg[-1]:.4f} (Overfitting!)")
print()

# ============================================================================
# Step 4: Apply L2 Regularization
# ============================================================================
print("=" * 70)
print("Step 4: Training with L2 Regularization")
print("=" * 70)
print()

# Create new model for L2 regularization
model_l2 = OverfittingModel()

# Optimizer with L2 regularization (weight decay)
# optim.Adam: Same optimizer as before
# weight_decay=0.01: L2 regularization strength
#   This adds penalty for large weights: Loss = Original_Loss + λ × Σ(weights²)
#   λ = weight_decay = 0.01
#   Effect: Keeps weights small, prevents overfitting
#   Higher weight_decay = stronger regularization (weights stay smaller)
optimizer_l2 = optim.Adam(
    model_l2.parameters(),
    lr=0.001,
    weight_decay=0.01  # L2 regularization strength
)

train_losses_l2 = []
val_losses_l2 = []

print("Training with L2 regularization...")
for epoch in range(epochs):
    model_l2.train()
    optimizer_l2.zero_grad()
    train_outputs = model_l2(X_train_tensor)
    train_loss = criterion(train_outputs, y_train_tensor)
    train_loss.backward()
    optimizer_l2.step()
    
    model_l2.eval()
    with torch.no_grad():
        val_outputs = model_l2(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
    
    train_losses_l2.append(train_loss.item())
    val_losses_l2.append(val_loss.item())
    
    if (epoch + 1) % 100 == 0:
        print(f"  Epoch {epoch+1}/{epochs}: Train={train_loss.item():.4f}, Val={val_loss.item():.4f}")

print()
print(f"Final Training Loss: {train_losses_l2[-1]:.4f}")
print(f"Final Validation Loss: {val_losses_l2[-1]:.4f}")
print(f"Gap: {val_losses_l2[-1] - train_losses_l2[-1]:.4f} (Much better!)")
print()

# ============================================================================
# Step 5: Apply Dropout
# ============================================================================
print("=" * 70)
print("Step 5: Training with Dropout")
print("=" * 70)
print()

class DropoutModel(nn.Module):
    """Model with dropout"""
    def __init__(self, input_size=20, hidden_size=128, output_size=1, dropout=0.5):
        # super(): Call parent class constructor
        super(DropoutModel, self).__init__()
        
        # Same architecture as OverfittingModel
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Dropout layer: Randomly sets some neurons to 0 during training
        # dropout=0.5: 50% of neurons are randomly disabled each iteration
        # Purpose: Prevents co-adaptation (neurons relying too much on each other)
        # Effect: Forces model to learn more robust features
        # During inference: All neurons used (but outputs scaled by dropout rate)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Forward pass with dropout
        x = self.relu(self.fc1(x))
        
        # Apply dropout after first hidden layer
        # During training: Randomly sets 50% of values to 0
        # During evaluation: All values used (dropout automatically disabled)
        x = self.dropout(x)
        
        x = self.relu(self.fc2(x))
        
        # Apply dropout after second hidden layer
        # Different neurons disabled each time (random)
        x = self.dropout(x)
        
        x = self.relu(self.fc3(x))
        # Note: No dropout before output layer (usually not needed)
        x = self.sigmoid(self.fc4(x))
        return x

model_dropout = DropoutModel(dropout=0.5)
optimizer_dropout = optim.Adam(model_dropout.parameters(), lr=0.001)

train_losses_dropout = []
val_losses_dropout = []

print("Training with dropout...")
for epoch in range(epochs):
    model_dropout.train()
    optimizer_dropout.zero_grad()
    train_outputs = model_dropout(X_train_tensor)
    train_loss = criterion(train_outputs, y_train_tensor)
    train_loss.backward()
    optimizer_dropout.step()
    
    model_dropout.eval()
    with torch.no_grad():
        val_outputs = model_dropout(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
    
    train_losses_dropout.append(train_loss.item())
    val_losses_dropout.append(val_loss.item())
    
    if (epoch + 1) % 100 == 0:
        print(f"  Epoch {epoch+1}/{epochs}: Train={train_loss.item():.4f}, Val={val_loss.item():.4f}")

print()
print(f"Final Training Loss: {train_losses_dropout[-1]:.4f}")
print(f"Final Validation Loss: {val_losses_dropout[-1]:.4f}")
print(f"Gap: {val_losses_dropout[-1] - train_losses_dropout[-1]:.4f}")
print()

# ============================================================================
# Step 6: Early Stopping
# ============================================================================
print("=" * 70)
print("Step 6: Training with Early Stopping")
print("=" * 70)
print()

def train_with_early_stopping(model, patience=20):
    """Train with early stopping"""
    # Optimizer for training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Track best validation loss seen so far
    # float('inf'): Start with infinity (any loss will be better)
    best_val_loss = float('inf')
    
    # Patience counter: How many epochs without improvement
    # patience=20: Stop if no improvement for 20 epochs
    patience_counter = 0
    
    # Store best model state (weights and biases)
    # We'll restore this if we stop early
    best_model_state = None
    
    # Lists to track losses
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # ===== TRAINING =====
        model.train()
        optimizer.zero_grad()
        train_outputs = model(X_train_tensor)
        train_loss = criterion(train_outputs, y_train_tensor)
        train_loss.backward()
        optimizer.step()
        
        # ===== VALIDATION =====
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        
        # ===== EARLY STOPPING LOGIC =====
        # Check if validation loss improved
        if val_loss.item() < best_val_loss:
            # New best validation loss found!
            best_val_loss = val_loss.item()
            # Reset patience counter (we're improving)
            patience_counter = 0
            # Save current model state (best model so far)
            # model.state_dict(): Dictionary of all weights and biases
            # .copy(): Create a copy (not just a reference)
            best_model_state = model.state_dict().copy()
        else:
            # No improvement this epoch
            patience_counter += 1
            # Check if we've exceeded patience
            if patience_counter >= patience:
                # Stop training: validation loss not improving
                print(f"  Early stopping at epoch {epoch+1}")
                # Restore best model (from when validation loss was lowest)
                # This prevents using overfitted model
                model.load_state_dict(best_model_state)
                break
        
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: Train={train_loss.item():.4f}, Val={val_loss.item():.4f}")
    
    return train_losses, val_losses

model_early = OverfittingModel()
train_losses_early, val_losses_early = train_with_early_stopping(model_early, patience=20)
print()

# ============================================================================
# Step 7: Visualize Comparison
# ============================================================================
print("=" * 70)
print("Step 7: Visualizing Results")
print("=" * 70)
print()

# Plot overfitting comparison
plt.figure(figsize=(15, 10))

# No regularization
plt.subplot(2, 2, 1)
plt.plot(train_losses_no_reg, label='Training Loss', linewidth=2)
plt.plot(val_losses_no_reg, label='Validation Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('No Regularization (Overfitting)', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# L2 Regularization
plt.subplot(2, 2, 2)
plt.plot(train_losses_l2, label='Training Loss', linewidth=2)
plt.plot(val_losses_l2, label='Validation Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('L2 Regularization', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Dropout
plt.subplot(2, 2, 3)
plt.plot(train_losses_dropout, label='Training Loss', linewidth=2)
plt.plot(val_losses_dropout, label='Validation Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Dropout', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Early Stopping
plt.subplot(2, 2, 4)
plt.plot(train_losses_early, label='Training Loss', linewidth=2)
plt.plot(val_losses_early, label='Validation Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Early Stopping', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regularization_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: regularization_comparison.png")
print()

# Summary comparison
print("=" * 70)
print("Summary Comparison")
print("=" * 70)
print()

methods = {
    'No Regularization': (train_losses_no_reg[-1], val_losses_no_reg[-1]),
    'L2 Regularization': (train_losses_l2[-1], val_losses_l2[-1]),
    'Dropout': (train_losses_dropout[-1], val_losses_dropout[-1]),
    'Early Stopping': (train_losses_early[-1], val_losses_early[-1])
}

print(f"{'Method':<20} {'Train Loss':<15} {'Val Loss':<15} {'Gap':<15}")
print("-" * 70)
for method, (train_loss, val_loss) in methods.items():
    gap = val_loss - train_loss
    print(f"{method:<20} {train_loss:<15.4f} {val_loss:<15.4f} {gap:<15.4f}")

print()
print("=" * 70)
print("Project 5b Complete!")
print("=" * 70)
print()
print("Summary:")
print("  ✅ Demonstrated overfitting")
print("  ✅ Applied L2 regularization")
print("  ✅ Implemented dropout")
print("  ✅ Used early stopping")
print("  ✅ Compared all techniques")
print()
