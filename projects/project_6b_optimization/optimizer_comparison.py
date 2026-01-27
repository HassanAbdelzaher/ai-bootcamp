"""
Project 6b: Optimization Techniques
Compare different optimization algorithms
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
from plotting import plot_learning_curve

print("=" * 70)
print("Project 6b: Optimization Techniques")
print("=" * 70)
print()

# ============================================================================
# Step 1: Create Dataset
# ============================================================================
print("=" * 70)
print("Step 1: Creating Dataset")
print("=" * 70)
print()

X, y = make_classification(
    n_samples=1000, n_features=20, n_classes=2, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print()

# ============================================================================
# Step 2: Define Model
# ============================================================================
print("=" * 70)
print("Step 2: Defining Model")
print("=" * 70)
print()

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# ============================================================================
# Step 3: Compare Optimizers
# ============================================================================
print("=" * 70)
print("Step 3: Comparing Optimizers")
print("=" * 70)
print()

# Dictionary of optimizer configurations
# Each entry is a lambda function that creates an optimizer
# This allows us to easily compare different optimizers
optimizers_config = {
    # SGD: Basic Stochastic Gradient Descent
    # Simple: w = w - lr × gradient
    # Pros: Simple, works for most problems
    # Cons: Can be slow, sensitive to learning rate
    'SGD': lambda params: optim.SGD(params, lr=0.01),
    
    # SGD with Momentum: Adds momentum to SGD
    # Remembers previous update direction
    # Formula: velocity = momentum × velocity + gradient
    #          w = w - lr × velocity
    # momentum=0.9: How much to remember previous direction (90%)
    # Pros: Faster convergence, smoother updates
    # Cons: Need to tune momentum parameter
    'SGD+Momentum': lambda params: optim.SGD(params, lr=0.01, momentum=0.9),
    
    # Adam: Adaptive Moment Estimation
    # Combines benefits of momentum and adaptive learning rates
    # Different learning rate for each parameter
    # Pros: Fast convergence, works well out-of-the-box
    # Cons: More memory, sometimes generalizes worse than SGD
    'Adam': lambda params: optim.Adam(params, lr=0.01),
    
    # RMSprop: Root Mean Square Propagation
    # Adaptive learning rates (like Adam, but simpler)
    # Good for RNNs and non-stationary problems
    # Pros: Stable, good for recurrent networks
    # Cons: Less popular than Adam
    'RMSprop': lambda params: optim.RMSprop(params, lr=0.01)
}

# Dictionary to store results for each optimizer
# Key: optimizer name, Value: list of losses per epoch
results = {}

# Loss function: Binary Cross-Entropy for binary classification
criterion = nn.BCELoss()

epochs = 100  # Number of training epochs

# Train model with each optimizer
for opt_name, opt_class in optimizers_config.items():
    print(f"Training with {opt_name}...")
    
    # Create fresh model for each optimizer
    # This ensures fair comparison (same starting point)
    model = SimpleNN()
    
    # Create optimizer using the lambda function
    # opt_class(model.parameters()): Calls lambda with model parameters
    # Returns configured optimizer (SGD, Adam, etc.)
    optimizer = opt_class(model.parameters())
    
    # List to track loss for each epoch
    losses = []
    
    # Training loop
    for epoch in range(epochs):
        # Set model to training mode
        model.train()
        
        # Clear gradients from previous iteration
        optimizer.zero_grad()
        
        # Forward pass: Make predictions
        outputs = model(X_train_tensor)
        
        # Calculate loss
        loss = criterion(outputs, y_train_tensor)
        
        # Backward pass: Compute gradients
        loss.backward()
        
        # Update weights using optimizer
        # Each optimizer uses different update rule
        optimizer.step()
        
        # Store loss for this epoch
        losses.append(loss.item())
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss = {loss.item():.4f}")
    
    # Store results for this optimizer
    results[opt_name] = losses
    print()

# ============================================================================
# Step 4: Learning Rate Scheduling
# ============================================================================
print("=" * 70)
print("Step 4: Learning Rate Scheduling")
print("=" * 70)
print()

scheduler_results = {}

# Dictionary of learning rate scheduler configurations
# Learning rate schedulers adjust learning rate during training
# This can help converge faster and find better solutions
schedulers_config = {
    # StepLR: Reduce learning rate by factor every N epochs
    # step_size=30: Reduce LR every 30 epochs
    # gamma=0.1: Multiply LR by 0.1 (reduce to 10% of current)
    # Example: LR starts at 0.01, becomes 0.001 at epoch 30, 0.0001 at epoch 60
    'StepLR': lambda opt: optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1),
    
    # ExponentialLR: Exponential decay of learning rate
    # gamma=0.95: Multiply LR by 0.95 each epoch (5% reduction)
    # Formula: LR = LR × gamma^epoch
    # Example: LR = 0.01 × 0.95^epoch (gradual decrease)
    'ExponentialLR': lambda opt: optim.lr_scheduler.ExponentialLR(opt, gamma=0.95),
    
    # CosineAnnealingLR: Cosine annealing schedule
    # T_max=100: Period of cosine function (full cycle in 100 epochs)
    # LR follows cosine curve: starts high, decreases, then increases slightly
    # Good for fine-tuning and escaping local minima
    'CosineAnnealingLR': lambda opt: optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
}

# Train with each scheduler
for sched_name, scheduler_func in schedulers_config.items():
    print(f"Training with {sched_name} scheduler...")
    
    # Create fresh model
    model = SimpleNN()
    
    # Create optimizer (Adam with initial LR=0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Create scheduler using the lambda function
    # scheduler_func(optimizer): Calls lambda with optimizer
    # Returns configured scheduler
    scheduler = scheduler_func(optimizer)
    
    # Lists to track losses and learning rates
    losses = []
    learning_rates = []
    
    # Training loop
    for epoch in range(epochs):
        # Standard training step
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        # Store loss and current learning rate
        losses.append(loss.item())
        
        # optimizer.param_groups[0]['lr']: Get current learning rate
        # param_groups: List of parameter groups (usually just one)
        # [0]: First (and usually only) parameter group
        # ['lr']: Learning rate value
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # Update learning rate using scheduler
        # scheduler.step(): Updates learning rate according to schedule
        # This happens after optimizer.step() (update LR for next epoch)
        scheduler.step()
        
        # Print progress
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss = {loss.item():.4f}, LR = {learning_rates[-1]:.6f}")
    
    # Store results for this scheduler
    scheduler_results[sched_name] = {
        'losses': losses,
        'learning_rates': learning_rates
    }
    print()

# ============================================================================
# Step 5: Visualize Results
# ============================================================================
print("=" * 70)
print("Step 5: Visualizing Results")
print("=" * 70)
print()

# Optimizer comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for name, losses in results.items():
    plt.plot(losses, label=name, linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Optimizer Comparison', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Learning rate schedules
plt.subplot(1, 2, 2)
for name, data in scheduler_results.items():
    plt.plot(data['learning_rates'], label=name, linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.title('Learning Rate Schedules', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optimizer_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: optimizer_comparison.png")
print()

# Final losses comparison
print("=" * 70)
print("Final Losses Comparison")
print("=" * 70)
print()
print(f"{'Optimizer':<20} {'Final Loss':<15}")
print("-" * 40)
for name, losses in results.items():
    print(f"{name:<20} {losses[-1]:<15.4f}")
print()

print("=" * 70)
print("Project 6b Complete!")
print("=" * 70)
print()
print("Summary:")
print("  ✅ Compared SGD, SGD+Momentum, Adam, RMSprop")
print("  ✅ Implemented learning rate scheduling")
print("  ✅ Visualized optimizer performance")
print("  ✅ Analyzed convergence speeds")
print()
