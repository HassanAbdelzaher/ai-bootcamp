"""
Step 6b — Optimization Techniques
Goal: Understand different optimizers and when to use each one
Tools: Python + PyTorch + Matplotlib
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
print("Step 6b: Optimization Techniques")
print("=" * 70)
print()
print("Goal: Compare different optimizers and understand when to use each")
print()

# ============================================================================
# 6b.1 Understanding the Problem
# ============================================================================
print("=== 6b.1 Why Different Optimizers? ===")
print("Different optimizers use different strategies to update weights:")
print("  • SGD: Simple gradient descent")
print("  • Momentum: Adds velocity to gradient descent")
print("  • Adam: Adaptive learning rates per parameter")
print("  • RMSprop: Adaptive learning rates (simpler than Adam)")
print()
print("Each has strengths for different problems!")
print()

# ============================================================================
# 6b.2 Create a Test Problem
# ============================================================================
print("=== 6b.2 Creating Test Problem ===")

# Create a non-trivial problem (XOR-like but more complex)
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic data
num_samples = 200
X = torch.randn(num_samples, 2) * 2
y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).float().unsqueeze(1)

print(f"Dataset: {num_samples} samples, 2 features")
print(f"Task: Binary classification (XOR-like)")
print()

# ============================================================================
# 6b.3 Define Model Architecture
# ============================================================================
print("=== 6b.3 Model Architecture ===")

def create_model():
    """Create a simple neural network"""
    return nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
        nn.Sigmoid()
    )

print("Model: 2 → 8 → 4 → 1 (with ReLU activations)")
print()

# ============================================================================
# 6b.4 Training Function
# ============================================================================
print("=== 6b.4 Training Function ===")

def train_with_optimizer(optimizer_name, optimizer_class, lr=0.01, epochs=1000):
    """Train model with specified optimizer"""
    model = create_model()
    loss_fn = nn.BCELoss()
    
    # Create optimizer
    if optimizer_name == "SGD":
        optimizer = optimizer_class(model.parameters(), lr=lr)
    elif optimizer_name == "SGD-Momentum":
        optimizer = optimizer_class(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "Adam":
        optimizer = optimizer_class(model.parameters(), lr=lr)
    elif optimizer_name == "RMSprop":
        optimizer = optimizer_class(model.parameters(), lr=lr)
    else:
        optimizer = optimizer_class(model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(epochs):
        # Forward pass
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        losses.append(loss.item())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Final accuracy
    with torch.no_grad():
        predictions = (model(X) >= 0.5).float()
        accuracy = (predictions == y).float().mean().item()
    
    return losses, accuracy

print("Training function created")
print()

# ============================================================================
# 6b.5 Compare Optimizers
# ============================================================================
print("=== 6b.5 Comparing Optimizers ===")
print("Training with different optimizers...")
print()

results = {}
optimizers_to_test = [
    ("SGD", optim.SGD, 0.1),
    ("SGD-Momentum", optim.SGD, 0.1),
    ("Adam", optim.Adam, 0.001),
    ("RMSprop", optim.RMSprop, 0.001),
]

for opt_name, opt_class, lr in optimizers_to_test:
    print(f"Training with {opt_name} (lr={lr})...")
    losses, accuracy = train_with_optimizer(opt_name, opt_class, lr=lr, epochs=1000)
    results[opt_name] = {
        'losses': losses,
        'accuracy': accuracy,
        'final_loss': losses[-1]
    }
    print(f"  Final loss: {losses[-1]:.4f}, Accuracy: {accuracy*100:.1f}%")
print()

# ============================================================================
# 6b.6 Visualize Comparison
# ============================================================================
print("=== 6b.6 Learning Curves Comparison ===")

plt.figure(figsize=(14, 8))

# Plot 1: Learning curves
plt.subplot(2, 2, 1)
for opt_name, result in results.items():
    plt.plot(result['losses'], label=opt_name, linewidth=2)
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Loss', fontsize=12, fontweight='bold')
plt.title('Learning Curves: Loss Over Time', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Log scale to see differences better

# Plot 2: Final performance
plt.subplot(2, 2, 2)
opt_names = list(results.keys())
final_losses = [results[opt]['final_loss'] for opt in opt_names]
accuracies = [results[opt]['accuracy'] * 100 for opt in opt_names]

x = np.arange(len(opt_names))
width = 0.35

ax1 = plt.gca()
ax2 = ax1.twinx()

bars1 = ax1.bar(x - width/2, final_losses, width, label='Final Loss', color='skyblue', alpha=0.7)
bars2 = ax2.bar(x + width/2, accuracies, width, label='Accuracy (%)', color='lightcoral', alpha=0.7)

ax1.set_xlabel('Optimizer', fontsize=12, fontweight='bold')
ax1.set_ylabel('Final Loss', fontsize=12, fontweight='bold', color='skyblue')
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color='lightcoral')
ax1.set_xticks(x)
ax1.set_xticklabels(opt_names, rotation=45, ha='right')
ax1.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 3: Convergence speed (first 200 epochs)
plt.subplot(2, 2, 3)
for opt_name, result in results.items():
    plt.plot(result['losses'][:200], label=opt_name, linewidth=2, alpha=0.8)
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Loss', fontsize=12, fontweight='bold')
plt.title('Early Training (First 200 Epochs)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Plot 4: Accuracy comparison
plt.subplot(2, 2, 4)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
bars = plt.bar(opt_names, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
plt.xlabel('Optimizer', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Accuracy Comparison', fontsize=14, fontweight='bold')
plt.ylim([0, 100])
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.1f}%',
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print("Visualization complete!")
print()

# ============================================================================
# 6b.7 Understanding Each Optimizer
# ============================================================================
print("=== 6b.7 Understanding Each Optimizer ===")
print()

print("1. SGD (Stochastic Gradient Descent):")
print("   • Simple: w = w - lr * gradient")
print("   • Pros: Simple, predictable")
print("   • Cons: Can be slow, sensitive to learning rate")
print("   • Best for: Simple problems, when you want control")
print()

print("2. SGD with Momentum:")
print("   • Adds velocity: v = momentum * v + gradient")
print("   • Pros: Faster convergence, helps escape local minima")
print("   • Cons: Still needs careful learning rate tuning")
print("   • Best for: When SGD is too slow")
print()

print("3. Adam (Adaptive Moment Estimation):")
print("   • Adaptive learning rate per parameter")
print("   • Tracks both momentum and squared gradients")
print("   • Pros: Fast convergence, less sensitive to learning rate")
print("   • Cons: Can sometimes generalize worse than SGD")
print("   • Best for: Most deep learning problems (default choice)")
print()

print("4. RMSprop:")
print("   • Adaptive learning rate based on gradient magnitude")
print("   • Simpler than Adam")
print("   • Pros: Good for non-stationary problems")
print("   • Cons: Less popular than Adam")
print("   • Best for: RNNs, when Adam doesn't work well")
print()

# ============================================================================
# 6b.8 Learning Rate Scheduling
# ============================================================================
print("=== 6b.8 Learning Rate Scheduling ===")
print("Sometimes we want to change learning rate during training")
print()

# Test with learning rate scheduler
model = create_model()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
loss_fn = nn.BCELoss()

losses_scheduled = []
learning_rates = []

print("Training with learning rate scheduler (reduces LR every 200 epochs)...")
for epoch in range(1000):
    # Forward pass
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    losses_scheduled.append(loss.item())
    learning_rates.append(scheduler.get_last_lr()[0])
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

print("Training complete!")
print()

# Visualize learning rate scheduling
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(losses_scheduled, label='With LR Scheduler', linewidth=2, color='purple')
plt.plot(results['Adam']['losses'], label='Without Scheduler', linewidth=2, color='orange', linestyle='--')
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Loss', fontsize=12, fontweight='bold')
plt.title('Effect of Learning Rate Scheduling', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.subplot(1, 2, 2)
plt.plot(learning_rates, linewidth=2, color='green')
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Learning Rate', fontsize=12, fontweight='bold')
plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.tight_layout()
plt.show()

print("Learning rate scheduling helps fine-tune convergence!")
print()

# ============================================================================
# 6b.9 When to Use Which Optimizer
# ============================================================================
print("=== 6b.9 When to Use Which Optimizer? ===")
print()

print("📊 Decision Guide:")
print()
print("Use SGD when:")
print("  • You want full control over training")
print("  • Problem is simple or well-understood")
print("  • You need reproducible results")
print("  • Training from scratch on large datasets")
print()

print("Use SGD with Momentum when:")
print("  • SGD is too slow")
print("  • You want faster convergence")
print("  • Problem has many local minima")
print()

print("Use Adam when:")
print("  • Default choice for most problems")
print("  • You want fast convergence")
print("  • Problem is complex")
print("  • Limited computational budget")
print("  • Working with CNNs, Transformers")
print()

print("Use RMSprop when:")
print("  • Working with RNNs")
print("  • Adam doesn't work well")
print("  • Non-stationary problems")
print()

print("💡 Pro Tip: Start with Adam (lr=0.001), then try others if needed!")
print()

# ============================================================================
# 6b.10 Common Pitfalls
# ============================================================================
print("=== 6b.10 Common Pitfalls ===")
print()

print("❌ Using wrong learning rate:")
print("   • SGD: Usually 0.01 - 0.1")
print("   • Adam: Usually 0.0001 - 0.001")
print("   • Too high: Training unstable")
print("   • Too low: Training too slow")
print()

print("❌ Forgetting to zero gradients:")
print("   • Always call optimizer.zero_grad() before backward()")
print("   • Otherwise gradients accumulate!")
print()

print("❌ Not using learning rate scheduling:")
print("   • Can help fine-tune convergence")
print("   • Especially useful for long training")
print()

print("❌ Using same optimizer for everything:")
print("   • Different problems need different optimizers")
print("   • Experiment to find what works best")
print()

# ============================================================================
# 6b.11 Summary
# ============================================================================
print("=== 6b.11 Summary ===")
print()

print("✅ You've learned:")
print("  • Different optimizers (SGD, Adam, RMSprop)")
print("  • How they compare in practice")
print("  • Learning rate scheduling")
print("  • When to use each optimizer")
print()

print("🎯 Key Takeaways:")
print("  1. Adam is usually the best default choice")
print("  2. SGD gives more control but needs tuning")
print("  3. Learning rate scheduling can help")
print("  4. Experiment to find what works for your problem")
print()

print("=" * 70)
print("Step 6b Complete! You understand optimization techniques!")
print("=" * 70)
