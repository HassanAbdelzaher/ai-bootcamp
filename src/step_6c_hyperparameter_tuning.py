"""
Step 6c — Hyperparameter Tuning
Goal: Learn systematic methods to find the best hyperparameters for your models
Tools: Python + PyTorch + NumPy + Matplotlib
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', message='.*Failed to initialize NumPy.*')

from plotting import plot_learning_curve

print("=" * 70)
print("Step 6c: Hyperparameter Tuning")
print("=" * 70)
print()
print("Goal: Learn systematic methods to find optimal hyperparameters")
print()

# ============================================================================
# 6c.1 What are Hyperparameters?
# ============================================================================
print("=== 6c.1 What are Hyperparameters? ===")
print()
print("Hyperparameters are settings that control how the model learns:")
print("  • Learning rate")
print("  • Number of layers")
print("  • Number of neurons per layer")
print("  • Batch size")
print("  • Optimizer type")
print("  • Regularization strength (λ)")
print("  • Dropout rate")
print()
print("Unlike model parameters (weights), hyperparameters are set BEFORE training")
print("and don't change during training.")
print()

# ============================================================================
# 6c.2 Why Hyperparameter Tuning Matters
# ============================================================================
print("=== 6c.2 Why Hyperparameter Tuning Matters ===")
print()
print("Good hyperparameters can:")
print("  ✅ Improve model accuracy significantly")
print("  ✅ Reduce training time")
print("  ✅ Prevent overfitting")
print("  ✅ Make training more stable")
print()
print("Bad hyperparameters can:")
print("  ❌ Cause training to fail")
print("  ❌ Lead to poor performance")
print("  ❌ Waste computational resources")
print()

# ============================================================================
# 6c.3 Create a Test Problem
# ============================================================================
print("=== 6c.3 Creating Test Problem ===")

np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic classification data
# num_samples: Number of data points
num_samples = 200
# X: Input features - random 2D points
# torch.randn(num_samples, 2) creates random values from standard normal distribution
# * 2 scales the values (makes them larger)
# Shape: (200, 2) - 200 samples, 2 features
X = torch.randn(num_samples, 2) * 2

# y: Target labels - circular pattern (non-linear boundary)
# X[:, 0]**2 + X[:, 1]**2 calculates distance squared from origin
# > 2.0 creates boolean tensor: True if point is outside circle of radius sqrt(2.0)
# .float() converts boolean to float (True→1.0, False→0.0)
# .unsqueeze(1) adds dimension: (200,) → (200, 1) for compatibility
y = ((X[:, 0]**2 + X[:, 1]**2) > 2.0).float().unsqueeze(1)

# Split into train and validation sets
# train_size: 70% of data for training
# int() converts float to integer (140.0 → 140)
train_size = int(0.7 * num_samples)  # 70% for training = 140 samples

# Training set: First 70% of data
# [:train_size] gets first train_size rows
X_train = X[:train_size]  # Shape: (140, 2)
y_train = y[:train_size]  # Shape: (140, 1)

# Validation set: Remaining 30% of data
# [train_size:] gets rows from train_size to end
X_val = X[train_size:]     # Shape: (60, 2)
y_val = y[train_size:]     # Shape: (60, 1)

print(f"Training set: {train_size} samples")
print(f"Validation set: {num_samples - train_size} samples")
print()

# ============================================================================
# 6c.4 Manual Hyperparameter Tuning (Baseline)
# ============================================================================
print("=== 6c.4 Manual Tuning (Baseline) ===")
print("First, let's see what happens with manual tuning...")
print()

def create_model(hidden_size=8, dropout_rate=0.0):
    """Create a simple neural network"""
    return nn.Sequential(
        nn.Linear(2, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_size, 1),
        nn.Sigmoid()
    )

def train_and_evaluate(lr=0.01, hidden_size=8, dropout_rate=0.0, epochs=500):
    """Train model and return validation accuracy"""
    # Create model with specified hyperparameters
    # hidden_size: Number of neurons in hidden layer
    # dropout_rate: Probability of dropping neurons (0.0 = no dropout)
    model = create_model(hidden_size, dropout_rate)
    
    # Create optimizer with specified learning rate
    # optim.Adam: Adaptive optimizer (good default choice)
    # model.parameters(): All trainable weights and biases
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Loss function for binary classification
    loss_fn = nn.BCELoss()  # Binary Cross-Entropy Loss
    
    # Training loop
    for epoch in range(epochs):
        # ===== TRAINING MODE =====
        # model.train() enables dropout and batch normalization training behavior
        model.train()
        
        # Forward pass: make predictions on training data
        y_pred = model(X_train)  # Shape: (140, 1) - probabilities
        # Calculate loss: how wrong are predictions?
        loss = loss_fn(y_pred, y_train)
        
        # Backward pass: calculate gradients
        optimizer.zero_grad()  # Clear gradients from previous iteration
        loss.backward()        # Calculate gradients
        optimizer.step()       # Update weights using gradients
    
    # ===== EVALUATION MODE =====
    # model.eval() disables dropout and batch normalization (use for inference)
    model.eval()
    # torch.no_grad() disables gradient computation (faster, uses less memory)
    with torch.no_grad():
        # Make predictions on validation set
        val_pred = model(X_val)  # Shape: (60, 1) - probabilities
        # Convert probabilities to binary predictions
        # (val_pred >= 0.5) creates boolean tensor: True if prob >= 0.5
        # .float() converts True/False to 1.0/0.0
        val_pred_binary = (val_pred >= 0.5).float()
        # Calculate accuracy: percentage of correct predictions
        # (val_pred_binary == y_val) creates boolean tensor: True where predictions match
        # .float() converts to 0.0/1.0, .mean() calculates average, .item() gets scalar
        accuracy = (val_pred_binary == y_val).float().mean().item()
    
    # Return validation accuracy (this is what we optimize!)
    return accuracy

# Try a few manual configurations
print("Trying manual configurations...")
manual_results = []

configs = [
    {"lr": 0.01, "hidden_size": 8, "dropout_rate": 0.0, "name": "Default"},
    {"lr": 0.001, "hidden_size": 8, "dropout_rate": 0.0, "name": "Lower LR"},
    {"lr": 0.01, "hidden_size": 16, "dropout_rate": 0.0, "name": "Larger Network"},
    {"lr": 0.01, "hidden_size": 8, "dropout_rate": 0.5, "name": "With Dropout"},
]

for config in configs:
    acc = train_and_evaluate(
        lr=config["lr"],
        hidden_size=config["hidden_size"],
        dropout_rate=config["dropout_rate"]
    )
    manual_results.append((config["name"], acc))
    print(f"  {config['name']}: Accuracy = {acc:.3f}")

print()
print("Manual tuning is slow and may miss optimal values!")
print()

# ============================================================================
# 6c.5 Grid Search
# ============================================================================
print("=== 6c.5 Grid Search ===")
print()
print("Grid Search: Try all combinations of hyperparameters")
print("  • Systematic: Tests every combination")
print("  • Exhaustive: Guaranteed to find best in grid")
print("  • Slow: Can be very time-consuming")
print()

def grid_search(lr_values, hidden_sizes, dropout_rates, epochs=300):
    """Perform grid search over hyperparameter space"""
    # Track best configuration found so far
    best_acc = 0        # Best accuracy seen
    best_params = None  # Best hyperparameters found
    # List to store all results (for analysis)
    results = []
    
    # Calculate total number of combinations to test
    # Grid search tests EVERY combination: lr_values × hidden_sizes × dropout_rates
    total_combinations = len(lr_values) * len(hidden_sizes) * len(dropout_rates)
    print(f"Testing {total_combinations} combinations...")
    print()
    
    # Nested loops: test every combination
    # Outer loop: try each learning rate
    for lr in lr_values:
        # Middle loop: try each hidden layer size
        for hidden_size in hidden_sizes:
            # Inner loop: try each dropout rate
            for dropout_rate in dropout_rates:
                # Train model with this combination of hyperparameters
                # Returns validation accuracy (what we want to maximize)
                acc = train_and_evaluate(lr, hidden_size, dropout_rate, epochs)
                
                # Store result for later analysis
                results.append({
                    'lr': lr,                    # Learning rate used
                    'hidden_size': hidden_size,  # Hidden layer size used
                    'dropout_rate': dropout_rate, # Dropout rate used
                    'accuracy': acc              # Validation accuracy achieved
                })
                
                # Update best if this is better
                if acc > best_acc:
                    best_acc = acc  # Update best accuracy
                    # Save best hyperparameters
                    best_params = {'lr': lr, 'hidden_size': hidden_size, 'dropout_rate': dropout_rate}
                
                print(f"  LR={lr:.4f}, Hidden={hidden_size}, Dropout={dropout_rate:.1f} → Acc={acc:.3f}")
    
    return best_params, best_acc, results

# Define search space
lr_values = [0.001, 0.01, 0.1]
hidden_sizes = [4, 8, 16]
dropout_rates = [0.0, 0.5]

print("Grid Search Configuration:")
print(f"  Learning rates: {lr_values}")
print(f"  Hidden sizes: {hidden_sizes}")
print(f"  Dropout rates: {dropout_rates}")
print()

best_grid_params, best_grid_acc, grid_results = grid_search(lr_values, hidden_sizes, dropout_rates)

print()
print(f"Best Grid Search Result:")
print(f"  Parameters: {best_grid_params}")
print(f"  Accuracy: {best_grid_acc:.3f}")
print()

# Visualize grid search results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap for learning rate vs hidden size (with dropout=0)
lr_hidden_results = {}
for result in grid_results:
    if result['dropout_rate'] == 0.0:
        key = (result['lr'], result['hidden_size'])
        if key not in lr_hidden_results or result['accuracy'] > lr_hidden_results[key]:
            lr_hidden_results[key] = result['accuracy']

# Create heatmap data
heatmap_data = np.zeros((len(lr_values), len(hidden_sizes)))
for i, lr in enumerate(lr_values):
    for j, hidden_size in enumerate(hidden_sizes):
        key = (lr, hidden_size)
        if key in lr_hidden_results:
            heatmap_data[i, j] = lr_hidden_results[key]

im = axes[0].imshow(heatmap_data, cmap='viridis', aspect='auto')
axes[0].set_xticks(range(len(hidden_sizes)))
axes[0].set_xticklabels(hidden_sizes)
axes[0].set_yticks(range(len(lr_values)))
axes[0].set_yticklabels([f'{lr:.3f}' for lr in lr_values])
axes[0].set_xlabel('Hidden Size', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
axes[0].set_title('Grid Search: LR vs Hidden Size (Dropout=0)', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=axes[0], label='Accuracy')

# Add text annotations
for i in range(len(lr_values)):
    for j in range(len(hidden_sizes)):
        text = axes[0].text(j, i, f'{heatmap_data[i, j]:.2f}',
                           ha="center", va="center", color="white", fontweight='bold')

# Bar chart of top results
top_results = sorted(grid_results, key=lambda x: x['accuracy'], reverse=True)[:5]
top_names = [f"LR={r['lr']:.3f}\nH={r['hidden_size']}\nD={r['dropout_rate']:.1f}" 
            for r in top_results]
top_accs = [r['accuracy'] for r in top_results]

bars = axes[1].barh(range(len(top_names)), top_accs, color='steelblue', alpha=0.7, edgecolor='black')
axes[1].set_yticks(range(len(top_names)))
axes[1].set_yticklabels(top_names, fontsize=9)
axes[1].set_xlabel('Validation Accuracy', fontsize=12, fontweight='bold')
axes[1].set_title('Top 5 Grid Search Results', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

for i, (bar, acc) in enumerate(zip(bars, top_accs)):
    axes[1].text(acc + 0.01, i, f'{acc:.3f}', va='center', fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================================
# 6c.6 Random Search
# ============================================================================
print("=== 6c.6 Random Search ===")
print()
print("Random Search: Try random combinations of hyperparameters")
print("  • Faster: Tests fewer combinations")
print("  • Efficient: Often finds good solutions quickly")
print("  • Less exhaustive: May miss optimal values")
print()

def random_search(lr_range, hidden_size_range, dropout_rate_range, n_trials=20, epochs=300):
    """Perform random search over hyperparameter space"""
    # Track best configuration found so far
    best_acc = 0        # Best accuracy seen
    best_params = None  # Best hyperparameters found
    # List to store all results (for analysis)
    results = []
    
    print(f"Testing {n_trials} random combinations...")
    print()
    
    # Try n_trials random combinations (much fewer than grid search!)
    for trial in range(n_trials):
        # ===== SAMPLE RANDOM HYPERPARAMETERS =====
        # Learning rate: random value in specified range
        # np.random.uniform(a, b) returns random float between a and b
        lr = np.random.uniform(lr_range[0], lr_range[1])
        
        # Hidden size: random choice from list of possible values
        # np.random.choice(array) picks one element randomly
        # int() converts to integer (hidden size must be whole number)
        hidden_size = int(np.random.choice(hidden_size_range))
        
        # Dropout rate: random value in specified range
        dropout_rate = np.random.uniform(dropout_rate_range[0], dropout_rate_range[1])
        
        # ===== TRAIN AND EVALUATE =====
        # Train model with these random hyperparameters
        acc = train_and_evaluate(lr, hidden_size, dropout_rate, epochs)
        
        # Store result
        results.append({
            'lr': lr,                    # Learning rate used
            'hidden_size': hidden_size,  # Hidden layer size used
            'dropout_rate': dropout_rate, # Dropout rate used
            'accuracy': acc              # Validation accuracy achieved
        })
        
        # Update best if this is better
        if acc > best_acc:
            best_acc = acc  # Update best accuracy
            # Save best hyperparameters
            best_params = {'lr': lr, 'hidden_size': hidden_size, 'dropout_rate': dropout_rate}
        
        print(f"  Trial {trial+1}/{n_trials}: LR={lr:.4f}, Hidden={hidden_size}, "
              f"Dropout={dropout_rate:.2f} → Acc={acc:.3f}")
    
    return best_params, best_acc, results

# Define search ranges
lr_range = (0.0001, 0.1)
hidden_size_range = [4, 8, 16, 32]
dropout_rate_range = (0.0, 0.7)

print("Random Search Configuration:")
print(f"  Learning rate range: {lr_range}")
print(f"  Hidden sizes: {hidden_size_range}")
print(f"  Dropout rate range: {dropout_rate_range}")
print(f"  Number of trials: 20")
print()

best_random_params, best_random_acc, random_results = random_search(
    lr_range, hidden_size_range, dropout_rate_range, n_trials=20
)

print()
print(f"Best Random Search Result:")
print(f"  Parameters: {best_random_params}")
print(f"  Accuracy: {best_random_acc:.3f}")
print()

# Compare Grid vs Random
print("=== Comparison: Grid Search vs Random Search ===")
print()
print(f"Grid Search:")
print(f"  Combinations tested: {len(grid_results)}")
print(f"  Best accuracy: {best_grid_acc:.3f}")
print(f"  Time: Exhaustive (tests all combinations)")
print()
print(f"Random Search:")
print(f"  Combinations tested: {len(random_results)}")
print(f"  Best accuracy: {best_random_acc:.3f}")
print(f"  Time: Faster (tests random subset)")
print()

# Visualize random search results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot: Learning rate vs Accuracy
lrs = [r['lr'] for r in random_results]
accs = [r['accuracy'] for r in random_results]
colors = [r['hidden_size'] for r in random_results]

scatter = axes[0].scatter(lrs, accs, c=colors, cmap='viridis', s=100, 
                         alpha=0.7, edgecolors='black', linewidth=1)
axes[0].set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
axes[0].set_title('Random Search: LR vs Accuracy', fontsize=13, fontweight='bold')
axes[0].set_xscale('log')
axes[0].grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=axes[0])
cbar.set_label('Hidden Size', fontsize=11, fontweight='bold')

# Convergence plot
sorted_random = sorted(random_results, key=lambda x: x['accuracy'], reverse=True)
best_so_far = []
for i in range(len(sorted_random)):
    best_so_far.append(max([r['accuracy'] for r in sorted_random[:i+1]]))

axes[1].plot(range(1, len(best_so_far)+1), best_so_far, 'o-', linewidth=2, 
            markersize=8, color='steelblue', label='Best So Far')
axes[1].axhline(y=best_grid_acc, color='red', linestyle='--', linewidth=2, 
               label=f'Grid Search Best ({best_grid_acc:.3f})')
axes[1].set_xlabel('Number of Trials', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Best Accuracy', fontsize=12, fontweight='bold')
axes[1].set_title('Random Search Convergence', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# 6c.7 Learning Rate Tuning
# ============================================================================
print("=== 6c.7 Learning Rate Tuning ===")
print()
print("Learning rate is often the most important hyperparameter!")
print("Let's find the optimal learning rate...")
print()

def find_optimal_lr(lr_values, epochs=300):
    """Find optimal learning rate"""
    results = []
    
    for lr in lr_values:
        acc = train_and_evaluate(lr=lr, hidden_size=8, dropout_rate=0.0, epochs=epochs)
        results.append({'lr': lr, 'accuracy': acc})
        print(f"  LR={lr:.4f} → Accuracy={acc:.3f}")
    
    best_lr_result = max(results, key=lambda x: x['accuracy'])
    return best_lr_result, results

# Test different learning rates
lr_candidates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

print("Testing learning rates:")
best_lr_result, lr_results = find_optimal_lr(lr_candidates)

print()
print(f"Optimal Learning Rate: {best_lr_result['lr']:.4f}")
print(f"Accuracy: {best_lr_result['accuracy']:.3f}")
print()

# Visualize learning rate tuning
fig, ax = plt.subplots(figsize=(10, 6))
lrs = [r['lr'] for r in lr_results]
accs = [r['accuracy'] for r in lr_results]

ax.plot(lrs, accs, 'o-', linewidth=2, markersize=10, color='steelblue')
ax.axvline(x=best_lr_result['lr'], color='red', linestyle='--', linewidth=2, 
          label=f'Optimal LR ({best_lr_result["lr"]:.4f})')
ax.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Learning Rate Tuning', fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

# Add value labels
for lr, acc in zip(lrs, accs):
    ax.text(lr, acc + 0.01, f'{acc:.3f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

print("Key Insight: Learning rate has a 'sweet spot' - too high or too low hurts performance!")
print()

# ============================================================================
# 6c.8 Hyperparameter Importance
# ============================================================================
print("=== 6c.8 Hyperparameter Importance ===")
print()
print("Which hyperparameters matter most?")
print()

# Analyze impact of each hyperparameter
def analyze_hyperparameter_importance():
    """Analyze which hyperparameters have the most impact"""
    
    # Fix other params, vary learning rate
    lr_impact = []
    for lr in [0.001, 0.01, 0.1]:
        acc = train_and_evaluate(lr=lr, hidden_size=8, dropout_rate=0.0)
        lr_impact.append(acc)
    lr_variance = np.var(lr_impact)
    
    # Fix other params, vary hidden size
    hidden_impact = []
    for hidden_size in [4, 8, 16]:
        acc = train_and_evaluate(lr=0.01, hidden_size=hidden_size, dropout_rate=0.0)
        hidden_impact.append(acc)
    hidden_variance = np.var(hidden_impact)
    
    # Fix other params, vary dropout
    dropout_impact = []
    for dropout_rate in [0.0, 0.3, 0.6]:
        acc = train_and_evaluate(lr=0.01, hidden_size=8, dropout_rate=dropout_rate)
        dropout_impact.append(acc)
    dropout_variance = np.var(dropout_impact)
    
    return {
        'learning_rate': lr_variance,
        'hidden_size': hidden_variance,
        'dropout_rate': dropout_variance
    }

print("Analyzing hyperparameter importance...")
importance = analyze_hyperparameter_importance()

print()
print("Hyperparameter Impact (variance in accuracy):")
for param, var in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"  {param.replace('_', ' ').title()}: {var:.4f}")

print()
print("Higher variance = more important hyperparameter")
print()

# Visualize importance
fig, ax = plt.subplots(figsize=(8, 5))
params = list(importance.keys())
vars = list(importance.values())
colors = ['steelblue', 'coral', 'lightgreen']

bars = ax.barh(params, vars, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Variance in Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Hyperparameter Importance', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

for bar, var in zip(bars, vars):
    ax.text(var + 0.0001, bar.get_y() + bar.get_height()/2,
           f'{var:.4f}', va='center', fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================================
# 6c.9 Bayesian Optimization (Conceptual)
# ============================================================================
print("=== 6c.9 Bayesian Optimization (Conceptual) ===")
print()
print("Bayesian Optimization: Smart search using probability")
print()
print("How it works:")
print("  1. Build a probabilistic model of the objective function")
print("  2. Use this model to suggest promising hyperparameters")
print("  3. Update model with new results")
print("  4. Repeat until convergence")
print()
print("Advantages:")
print("  ✅ More efficient than random search")
print("  ✅ Fewer evaluations needed")
print("  ✅ Good for expensive evaluations")
print()
print("Tools:")
print("  • scikit-optimize (skopt)")
print("  • Optuna")
print("  • Hyperopt")
print()
print("Example (conceptual):")
print("  Instead of trying random values, Bayesian optimization")
print("  uses past results to intelligently choose next values to try.")
print()

# ============================================================================
# 6c.10 Best Practices
# ============================================================================
print("=== 6c.10 Best Practices ===")
print()

print("✅ Start with defaults:")
print("   • Use recommended values from literature")
print("   • PyTorch defaults are often good starting points")
print()

print("✅ Tune one at a time (initially):")
print("   • Start with learning rate (most important)")
print("   • Then architecture (layers, neurons)")
print("   • Then regularization")
print()

print("✅ Use validation set:")
print("   • Never tune on test set!")
print("   • Use separate validation set for tuning")
print("   • Keep test set for final evaluation")
print()

print("✅ Start coarse, then fine:")
print("   • First: Wide ranges (e.g., LR: 0.001 to 0.1)")
print("   • Then: Narrow around best (e.g., LR: 0.01 to 0.03)")
print()

print("✅ Use appropriate search method:")
print("   • Grid search: Small search space (< 100 combinations)")
print("   • Random search: Large search space")
print("   • Bayesian: Expensive evaluations")
print()

print("✅ Track experiments:")
print("   • Log all hyperparameters and results")
print("   • Use tools like MLflow, Weights & Biases")
print()

# ============================================================================
# 6c.11 Summary
# ============================================================================
print("=== 6c.11 Summary ===")
print()

print("✅ You've learned:")
print("  • What hyperparameters are")
print("  • Grid search (exhaustive)")
print("  • Random search (efficient)")
print("  • Learning rate tuning")
print("  • Hyperparameter importance")
print("  • Bayesian optimization (conceptual)")
print()

print("🎯 Key Takeaways:")
print("  1. Learning rate is often most important")
print("  2. Random search often beats grid search")
print("  3. Start with defaults, tune systematically")
print("  4. Use validation set, not test set")
print("  5. Track all experiments")
print()

print("=" * 70)
print("Step 6c Complete! You understand hyperparameter tuning!")
print("=" * 70)
