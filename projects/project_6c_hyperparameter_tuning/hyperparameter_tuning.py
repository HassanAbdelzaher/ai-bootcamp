"""
Project 6c: Hyperparameter Tuning
Systematically find best hyperparameters
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from itertools import product
import random
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# Add parent directory to path to import plotting utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from plotting import plot_learning_curve

print("=" * 70)
print("Project 6c: Hyperparameter Tuning")
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
# Step 2: Define Tunable Model
# ============================================================================
print("=" * 70)
print("Step 2: Defining Tunable Model")
print("=" * 70)
print()

class TunableNN(nn.Module):
    def __init__(self, input_size=20, hidden_size=64, num_layers=2, dropout=0.0):
        super(TunableNN, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# ============================================================================
# Step 3: Training Function
# ============================================================================
def train_and_evaluate(params, epochs=50):
    """Train model with given hyperparameters and return test accuracy"""
    # Create model with specified hyperparameters
    # params: Dictionary with 'hidden_size', 'num_layers', 'dropout', 'learning_rate'
    # TunableNN: Model architecture that accepts these hyperparameters
    model = TunableNN(
        hidden_size=params['hidden_size'],  # Size of hidden layers
        num_layers=params['num_layers'],    # Number of hidden layers
        dropout=params['dropout']            # Dropout rate (0.0 to 1.0)
    )
    
    # Loss function: Binary Cross-Entropy for binary classification
    criterion = nn.BCELoss()
    
    # Optimizer: Adam with specified learning rate
    # params['learning_rate']: Learning rate from hyperparameter search
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    # Training loop
    # epochs=50: Train for 50 epochs (enough to see differences, not too slow)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    # Evaluation on test set
    # model.eval(): Set to evaluation mode (disables dropout, etc.)
    model.eval()
    
    # torch.no_grad(): Disable gradient computation (faster, saves memory)
    with torch.no_grad():
        # Make predictions on test set
        outputs = model(X_test_tensor)
        
        # Convert probabilities to binary predictions
        # outputs >= 0.5: Threshold at 0.5 (standard for binary classification)
        # .float(): Convert boolean to float (0.0 or 1.0)
        predictions = (outputs >= 0.5).float()
        
        # Calculate accuracy
        # predictions == y_test_tensor: Element-wise comparison (True/False)
        # .float(): Convert boolean to float (1.0 for True, 0.0 for False)
        # .mean(): Average (gives proportion of correct predictions)
        # .item(): Extract scalar value from tensor
        accuracy = (predictions == y_test_tensor).float().mean().item()
    
    # Return accuracy as evaluation metric
    # Higher accuracy = better hyperparameters
    return accuracy

# ============================================================================
# Step 4: Grid Search
# ============================================================================
print("=" * 70)
print("Step 4: Grid Search")
print("=" * 70)
print()

# Define parameter grid for grid search
# Each key is a hyperparameter, value is list of values to try
param_grid = {
    # Learning rates to try: 0.001 (small), 0.01 (medium), 0.1 (large)
    # Different learning rates can dramatically affect performance
    'learning_rate': [0.001, 0.01, 0.1],
    
    # Hidden layer sizes: 32 (small), 64 (medium), 128 (large)
    # Larger = more capacity but more parameters
    'hidden_size': [32, 64, 128],
    
    # Number of hidden layers: 1 (shallow), 2 (medium), 3 (deep)
    # More layers = deeper network, can learn more complex patterns
    'num_layers': [1, 2, 3],
    
    # Dropout rates: 0.0 (no dropout), 0.2 (light), 0.5 (moderate)
    # Dropout prevents overfitting by randomly disabling neurons
    'dropout': [0.0, 0.2, 0.5]
}

# Track best hyperparameters found so far
best_score = 0      # Best accuracy score
best_params = None  # Best hyperparameter combination
results = []        # List of (params, score) tuples

# Calculate total number of combinations
# product(*param_grid.values()): Cartesian product of all parameter lists
# Example: 3 LRs × 3 sizes × 3 layers × 3 dropouts = 81 combinations
total_combinations = len(list(product(*param_grid.values())))
print(f"Total combinations to try: {total_combinations}")
print("Running grid search...")
print()

# Grid search: Try every combination
# enumerate(): Get index and value from iterator
for idx, params in enumerate(product(*param_grid.values())):
    # Convert tuple to dictionary
    # zip(param_grid.keys(), params): Pair parameter names with values
    # Example: ('learning_rate', 'hidden_size', ...) with (0.001, 32, ...)
    # dict(...): Convert to dictionary {'learning_rate': 0.001, ...}
    param_dict = dict(zip(param_grid.keys(), params))
    
    # Train and evaluate model with these hyperparameters
    # train_and_evaluate(): Trains model and returns test accuracy
    score = train_and_evaluate(param_dict)
    
    # Store results
    # param_dict.copy(): Create copy (not reference) to avoid overwriting
    results.append((param_dict.copy(), score))
    
    # Update best if this is better
    if score > best_score:
        best_score = score
        best_params = param_dict.copy()
    
    # Print progress every 10 combinations
    if (idx + 1) % 10 == 0:
        print(f"  Progress: {idx+1}/{total_combinations}, Best Score: {best_score:.4f}")

print()
print(f"Best Params: {best_params}")
print(f"Best Score: {best_score:.4f}")
print()

# ============================================================================
# Step 5: Random Search
# ============================================================================
print("=" * 70)
print("Step 5: Random Search")
print("=" * 70)
print()

def random_search(n_trials=50):
    """Random search for hyperparameters"""
    # Random search: Randomly sample hyperparameter space
    # Often finds good hyperparameters faster than grid search
    # Especially when some hyperparameters don't matter much
    
    best_score = 0      # Track best accuracy
    best_params = None  # Track best hyperparameters
    results = []        # Store all results
    
    # Try n_trials random combinations
    for trial in range(n_trials):
        # Randomly sample hyperparameters
        # random.choice(): Randomly select from list
        # random.randint(): Random integer in range [1, 4)
        params = {
            # Learning rate: Randomly choose from options
            'learning_rate': random.choice([0.0001, 0.001, 0.01, 0.1]),
            
            # Hidden size: Randomly choose from options
            # Note: Includes 256 (not in grid search) to show flexibility
            'hidden_size': random.choice([32, 64, 128, 256]),
            
            # Number of layers: Random integer from 1 to 3
            # random.randint(1, 4): Returns 1, 2, or 3 (4 is exclusive)
            'num_layers': random.randint(1, 4),
            
            # Dropout: Randomly choose from options
            # More options than grid search (includes 0.1, 0.3)
            'dropout': random.choice([0.0, 0.1, 0.2, 0.3, 0.5])
        }
        
        # Train and evaluate with these random hyperparameters
        score = train_and_evaluate(params)
        
        # Store results
        results.append((params.copy(), score))
        
        # Update best if this is better
        if score > best_score:
            best_score = score
            best_params = params.copy()
        
        # Print progress every 10 trials
        if (trial + 1) % 10 == 0:
            print(f"  Trial {trial+1}/{n_trials}, Best Score: {best_score:.4f}")
    
    # Return best hyperparameters, best score, and all results
    return best_params, best_score, results

print("Running random search...")
best_params_rand, best_score_rand, rand_results = random_search(n_trials=50)
print()
print(f"Best Params: {best_params_rand}")
print(f"Best Score: {best_score_rand:.4f}")
print()

# ============================================================================
# Step 6: Analyze Hyperparameter Importance
# ============================================================================
print("=" * 70)
print("Step 6: Hyperparameter Importance Analysis")
print("=" * 70)
print()

# Convert results to DataFrame
df = pd.DataFrame([
    {**params, 'score': score} for params, score in results
])

print("Hyperparameter Importance (from Grid Search):")
print("=" * 50)

for param in param_grid.keys():
    importance = df.groupby(param)['score'].mean()
    print(f"\n{param}:")
    for value, mean_score in importance.items():
        print(f"  {value}: {mean_score:.4f}")

print()

# ============================================================================
# Step 7: Visualize Results
# ============================================================================
print("=" * 70)
print("Step 7: Visualizing Results")
print("=" * 70)
print()

# Top 10 configurations
top_10 = sorted(results, key=lambda x: x[1], reverse=True)[:10]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Learning rate impact
lr_impact = df.groupby('learning_rate')['score'].mean()
axes[0, 0].bar(range(len(lr_impact)), lr_impact.values)
axes[0, 0].set_xticks(range(len(lr_impact)))
axes[0, 0].set_xticklabels(lr_impact.index)
axes[0, 0].set_xlabel('Learning Rate')
axes[0, 0].set_ylabel('Mean Score')
axes[0, 0].set_title('Learning Rate Impact')
axes[0, 0].grid(True, alpha=0.3)

# Hidden size impact
hidden_impact = df.groupby('hidden_size')['score'].mean()
axes[0, 1].bar(range(len(hidden_impact)), hidden_impact.values)
axes[0, 1].set_xticks(range(len(hidden_impact)))
axes[0, 1].set_xticklabels(hidden_impact.index)
axes[0, 1].set_xlabel('Hidden Size')
axes[0, 1].set_ylabel('Mean Score')
axes[0, 1].set_title('Hidden Size Impact')
axes[0, 1].grid(True, alpha=0.3)

# Number of layers impact
layers_impact = df.groupby('num_layers')['score'].mean()
axes[1, 0].bar(range(len(layers_impact)), layers_impact.values)
axes[1, 0].set_xticks(range(len(layers_impact)))
axes[1, 0].set_xticklabels(layers_impact.index)
axes[1, 0].set_xlabel('Number of Layers')
axes[1, 0].set_ylabel('Mean Score')
axes[1, 0].set_title('Number of Layers Impact')
axes[1, 0].grid(True, alpha=0.3)

# Dropout impact
dropout_impact = df.groupby('dropout')['score'].mean()
axes[1, 1].bar(range(len(dropout_impact)), dropout_impact.values)
axes[1, 1].set_xticks(range(len(dropout_impact)))
axes[1, 1].set_xticklabels(dropout_impact.index)
axes[1, 1].set_xlabel('Dropout Rate')
axes[1, 1].set_ylabel('Mean Score')
axes[1, 1].set_title('Dropout Impact')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hyperparameter_importance.png', dpi=150, bbox_inches='tight')
print("Saved: hyperparameter_importance.png")
print()

# Comparison
print("=" * 70)
print("Grid Search vs Random Search")
print("=" * 70)
print()
print(f"Grid Search:")
print(f"  Evaluations: {len(results)}")
print(f"  Best Score: {best_score:.4f}")
print(f"  Best Params: {best_params}")
print()
print(f"Random Search:")
print(f"  Evaluations: {len(rand_results)}")
print(f"  Best Score: {best_score_rand:.4f}")
print(f"  Best Params: {best_params_rand}")
print()

print("=" * 70)
print("Project 6c Complete!")
print("=" * 70)
print()
print("Summary:")
print("  ✅ Implemented grid search")
print("  ✅ Implemented random search")
print("  ✅ Analyzed hyperparameter importance")
print("  ✅ Found optimal hyperparameters")
print()
