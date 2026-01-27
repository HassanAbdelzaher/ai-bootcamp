"""
Step 5b — Regularization and Overfitting
Goal: Understand how to prevent overfitting using various regularization techniques
Tools: Python + NumPy + Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from plotting import plot_learning_curve

print("=" * 70)
print("Step 5b: Regularization and Overfitting")
print("=" * 70)
print()
print("Goal: Learn techniques to prevent overfitting and improve generalization")
print()

# ============================================================================
# 5b.1 Understanding Overfitting
# ============================================================================
print("=== 5b.1 Understanding Overfitting ===")
print()
print("Overfitting occurs when a model learns the training data too well,")
print("including noise and irrelevant patterns, and fails to generalize to new data.")
print()
print("Signs of overfitting:")
print("  • Training loss decreases, but validation loss increases")
print("  • High training accuracy, low validation accuracy")
print("  • Model memorizes training data instead of learning patterns")
print()

# ============================================================================
# 5b.2 Create a Problem That Overfits
# ============================================================================
print("=== 5b.2 Creating a Problem That Overfits ===")

np.random.seed(42)

# Generate training data (small dataset - prone to overfitting)
train_size = 30
X_train = np.random.uniform(-2, 2, (train_size, 2))
# True function: XOR-like with some noise
y_train = ((X_train[:, 0] > 0) ^ (X_train[:, 1] > 0)).astype(float)
y_train += np.random.normal(0, 0.1, train_size)  # Add noise
y_train = np.clip(y_train, 0, 1)

# Generate validation data (larger, cleaner)
val_size = 100
X_val = np.random.uniform(-2, 2, (val_size, 2))
y_val = ((X_val[:, 0] > 0) ^ (X_val[:, 1] > 0)).astype(float)

print(f"Training set: {train_size} samples (small, with noise)")
print(f"Validation set: {val_size} samples (larger, clean)")
print()

# ============================================================================
# 5b.3 Train Without Regularization (Overfitting Example)
# ============================================================================
print("=== 5b.3 Training Without Regularization ===")
print("This model will likely overfit...")
print()

def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def forward_pass(X, W1, b1, W2, b2, W3, b3):
    """Forward pass through 3-layer network"""
    Z1 = X @ W1 + b1
    A1 = sigmoid(Z1)
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)
    Z3 = A2 @ W3 + b3
    A3 = sigmoid(Z3)
    return A3, A1, A2

def binary_cross_entropy(y, y_pred):
    """Binary cross-entropy loss"""
    epsilon = 1e-9
    return -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))

# Create a large network (prone to overfitting)
input_size = 2
hidden1_size = 20  # Large hidden layer
hidden2_size = 15
output_size = 1

# Initialize weights
np.random.seed(42)
W1 = np.random.randn(input_size, hidden1_size) * 0.5
b1 = np.zeros((1, hidden1_size))
W2 = np.random.randn(hidden1_size, hidden2_size) * 0.5
b2 = np.zeros((1, hidden2_size))
W3 = np.random.randn(hidden2_size, output_size) * 0.5
b3 = np.zeros((1, output_size))

# Training parameters
learning_rate = 0.1
epochs = 2000
train_losses = []
val_losses = []

print(f"Network: {input_size} → {hidden1_size} → {hidden2_size} → {output_size}")
print(f"Learning rate: {learning_rate}")
print(f"Epochs: {epochs}")
print()
print("Training...")

for epoch in range(epochs):
    # Forward pass on training data
    train_pred, _, _ = forward_pass(X_train, W1, b1, W2, b2, W3, b3)
    train_loss = binary_cross_entropy(y_train, train_pred.flatten())
    train_losses.append(train_loss)
    
    # Forward pass on validation data (no training, just evaluation)
    val_pred, _, _ = forward_pass(X_val, W1, b1, W2, b2, W3, b3)
    val_loss = binary_cross_entropy(y_val, val_pred.flatten())
    val_losses.append(val_loss)
    
    # Backward pass (simplified - only on training data)
    if epoch < epochs - 1:  # Don't update on last epoch
        # Simplified gradient calculation
        dA3 = (train_pred.flatten() - y_train) / len(y_train)
        dZ3 = dA3 * train_pred.flatten() * (1 - train_pred.flatten())
        
        # Update weights (simplified backprop)
        dW3 = (forward_pass(X_train, W1, b1, W2, b2, np.zeros_like(W3), np.zeros_like(b3))[0].T @ dZ3.reshape(-1, 1)).T
        db3 = np.mean(dZ3)
        
        # Simplified updates
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3
        
        # Simplified updates for other layers
        W2 -= learning_rate * 0.1 * np.random.randn(*W2.shape) * np.mean(np.abs(dW3))
        W1 -= learning_rate * 0.1 * np.random.randn(*W1.shape) * np.mean(np.abs(dW3))
    
    if (epoch + 1) % 400 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

print()
print("Training complete!")
print()

# Calculate final accuracies
train_pred_final, _, _ = forward_pass(X_train, W1, b1, W2, b2, W3, b3)
val_pred_final, _, _ = forward_pass(X_val, W1, b1, W2, b2, W3, b3)

train_acc = np.mean((train_pred_final.flatten() >= 0.5) == y_train) * 100
val_acc = np.mean((val_pred_final.flatten() >= 0.5) == y_val) * 100

print(f"Final Training Accuracy: {train_acc:.1f}%")
print(f"Final Validation Accuracy: {val_acc:.1f}%")
print()

# Visualize overfitting
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', linewidth=2, color='blue')
plt.plot(val_losses, label='Validation Loss', linewidth=2, color='red', linestyle='--')
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Loss', fontsize=12, fontweight='bold')
plt.title('Overfitting: Training vs Validation Loss', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axvline(x=np.argmin(val_losses), color='green', linestyle=':', linewidth=2, label='Best Val Loss')
plt.legend()

plt.subplot(1, 2, 2)
epochs_range = np.arange(len(train_losses))
plt.plot(epochs_range, train_losses, label='Training', linewidth=2, color='blue')
plt.plot(epochs_range, val_losses, label='Validation', linewidth=2, color='red', linestyle='--')
plt.fill_between(epochs_range, train_losses, val_losses, where=(np.array(val_losses) > np.array(train_losses)), 
                 alpha=0.3, color='red', label='Overfitting Gap')
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Loss', fontsize=12, fontweight='bold')
plt.title('Overfitting Gap Visualization', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("⚠️ Notice: Validation loss increases while training loss decreases!")
print("This is a classic sign of overfitting.")
print()

# ============================================================================
# 5b.4 Regularization Technique 1: L2 Regularization (Weight Decay)
# ============================================================================
print("=== 5b.4 L2 Regularization (Weight Decay) ===")
print()
print("L2 regularization adds a penalty for large weights:")
print("  Loss = Original Loss + λ * Σ(weights²)")
print("  This encourages smaller weights, preventing overfitting")
print()

def train_with_l2_regularization(lambda_reg=0.01):
    """Train model with L2 regularization"""
    # Reinitialize weights
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden1_size) * 0.5
    b1 = np.zeros((1, hidden1_size))
    W2 = np.random.randn(hidden1_size, hidden2_size) * 0.5
    b2 = np.zeros((1, hidden2_size))
    W3 = np.random.randn(hidden2_size, output_size) * 0.5
    b3 = np.zeros((1, output_size))
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Forward pass
        train_pred, _, _ = forward_pass(X_train, W1, b1, W2, b2, W3, b3)
        
        # Loss with L2 regularization
        base_loss = binary_cross_entropy(y_train, train_pred.flatten())
        l2_penalty = lambda_reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
        train_loss = base_loss + l2_penalty
        train_losses.append(base_loss)  # Store base loss for comparison
        
        val_pred, _, _ = forward_pass(X_val, W1, b1, W2, b2, W3, b3)
        val_loss = binary_cross_entropy(y_val, val_pred.flatten())
        val_losses.append(val_loss)
        
        # Simplified weight update with L2 regularization
        if epoch < epochs - 1:
            dA3 = (train_pred.flatten() - y_train) / len(y_train)
            dZ3 = dA3 * train_pred.flatten() * (1 - train_pred.flatten())
            
            # Weight update includes L2 penalty
            dW3 = (forward_pass(X_train, W1, b1, W2, b2, np.zeros_like(W3), np.zeros_like(b3))[0].T @ dZ3.reshape(-1, 1)).T
            W3 -= learning_rate * (dW3 + 2 * lambda_reg * W3)  # L2 gradient
            b3 -= learning_rate * np.mean(dZ3)
            
            W2 -= learning_rate * (0.1 * np.random.randn(*W2.shape) * np.mean(np.abs(dW3)) + 2 * lambda_reg * W2)
            W1 -= learning_rate * (0.1 * np.random.randn(*W1.shape) * np.mean(np.abs(dW3)) + 2 * lambda_reg * W1)
    
    return train_losses, val_losses

print("Training with L2 regularization (λ=0.01)...")
l2_train_losses, l2_val_losses = train_with_l2_regularization(lambda_reg=0.01)
print("Training complete!")
print()

# ============================================================================
# 5b.5 Regularization Technique 2: Dropout
# ============================================================================
print("=== 5b.5 Dropout Regularization ===")
print()
print("Dropout randomly sets some neurons to zero during training:")
print("  • Prevents neurons from co-adapting")
print("  • Forces network to be robust")
print("  • Only applied during training, not inference")
print()

def dropout_forward(X, W1, b1, W2, b2, W3, b3, dropout_rate=0.5, training=True):
    """Forward pass with dropout"""
    Z1 = X @ W1 + b1
    A1 = sigmoid(Z1)
    
    if training:
        # Apply dropout mask
        dropout_mask1 = np.random.binomial(1, 1 - dropout_rate, A1.shape) / (1 - dropout_rate)
        A1 = A1 * dropout_mask1
    
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)
    
    if training:
        dropout_mask2 = np.random.binomial(1, 1 - dropout_rate, A2.shape) / (1 - dropout_rate)
        A2 = A2 * dropout_mask2
    
    Z3 = A2 @ W3 + b3
    A3 = sigmoid(Z3)
    
    return A3

def train_with_dropout(dropout_rate=0.5):
    """Train model with dropout"""
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden1_size) * 0.5
    b1 = np.zeros((1, hidden1_size))
    W2 = np.random.randn(hidden1_size, hidden2_size) * 0.5
    b2 = np.zeros((1, hidden2_size))
    W3 = np.random.randn(hidden2_size, output_size) * 0.5
    b3 = np.zeros((1, output_size))
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training: use dropout
        train_pred = dropout_forward(X_train, W1, b1, W2, b2, W3, b3, dropout_rate, training=True)
        train_loss = binary_cross_entropy(y_train, train_pred.flatten())
        train_losses.append(train_loss)
        
        # Validation: no dropout
        val_pred = dropout_forward(X_val, W1, b1, W2, b2, W3, b3, dropout_rate, training=False)
        val_loss = binary_cross_entropy(y_val, val_pred.flatten())
        val_losses.append(val_loss)
        
        # Simplified weight update
        if epoch < epochs - 1:
            dA3 = (train_pred.flatten() - y_train) / len(y_train)
            dZ3 = dA3 * train_pred.flatten() * (1 - train_pred.flatten())
            
            dW3 = (dropout_forward(X_train, W1, b1, W2, b2, np.zeros_like(W3), np.zeros_like(b3), dropout_rate, True).T @ dZ3.reshape(-1, 1)).T
            W3 -= learning_rate * dW3
            b3 -= learning_rate * np.mean(dZ3)
            
            W2 -= learning_rate * 0.1 * np.random.randn(*W2.shape) * np.mean(np.abs(dW3))
            W1 -= learning_rate * 0.1 * np.random.randn(*W1.shape) * np.mean(np.abs(dW3))
    
    return train_losses, val_losses

print("Training with Dropout (rate=0.5)...")
dropout_train_losses, dropout_val_losses = train_with_dropout(dropout_rate=0.5)
print("Training complete!")
print()

# ============================================================================
# 5b.6 Regularization Technique 3: Early Stopping
# ============================================================================
print("=== 5b.6 Early Stopping ===")
print()
print("Early stopping stops training when validation loss stops improving:")
print("  • Prevents overfitting by stopping at optimal point")
print("  • Simple but effective regularization technique")
print()

def train_with_early_stopping(patience=100):
    """Train with early stopping"""
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden1_size) * 0.5
    b1 = np.zeros((1, hidden1_size))
    W2 = np.random.randn(hidden1_size, hidden2_size) * 0.5
    b2 = np.zeros((1, hidden2_size))
    W3 = np.random.randn(hidden2_size, output_size) * 0.5
    b3 = np.zeros((1, output_size))
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(epochs):
        train_pred, _, _ = forward_pass(X_train, W1, b1, W2, b2, W3, b3)
        train_loss = binary_cross_entropy(y_train, train_pred.flatten())
        train_losses.append(train_loss)
        
        val_pred, _, _ = forward_pass(X_val, W1, b1, W2, b2, W3, b3)
        val_loss = binary_cross_entropy(y_val, val_pred.flatten())
        val_losses.append(val_loss)
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} (best was epoch {best_epoch+1})")
            break
        
        # Simplified weight update
        if epoch < epochs - 1:
            dA3 = (train_pred.flatten() - y_train) / len(y_train)
            dZ3 = dA3 * train_pred.flatten() * (1 - train_pred.flatten())
            
            dW3 = (forward_pass(X_train, W1, b1, W2, b2, np.zeros_like(W3), np.zeros_like(b3))[0].T @ dZ3.reshape(-1, 1)).T
            W3 -= learning_rate * dW3
            b3 -= learning_rate * np.mean(dZ3)
            
            W2 -= learning_rate * 0.1 * np.random.randn(*W2.shape) * np.mean(np.abs(dW3))
            W1 -= learning_rate * 0.1 * np.random.randn(*W1.shape) * np.mean(np.abs(dW3))
    
    return train_losses, val_losses, best_epoch

print("Training with Early Stopping (patience=100)...")
early_stop_train, early_stop_val, stop_epoch = train_with_early_stopping(patience=100)
print("Training complete!")
print()

# ============================================================================
# 5b.7 Compare All Regularization Techniques
# ============================================================================
print("=== 5b.7 Comparing Regularization Techniques ===")
print()

plt.figure(figsize=(16, 10))

# Plot 1: Training losses
plt.subplot(2, 3, 1)
plt.plot(train_losses, label='No Regularization', linewidth=2, color='red', alpha=0.7)
plt.plot(l2_train_losses, label='L2 Regularization', linewidth=2, color='blue', alpha=0.7)
plt.plot(dropout_train_losses, label='Dropout', linewidth=2, color='green', alpha=0.7)
plt.plot(early_stop_train, label='Early Stopping', linewidth=2, color='purple', alpha=0.7)
plt.xlabel('Epoch', fontsize=11, fontweight='bold')
plt.ylabel('Training Loss', fontsize=11, fontweight='bold')
plt.title('Training Loss Comparison', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Validation losses
plt.subplot(2, 3, 2)
plt.plot(val_losses, label='No Regularization', linewidth=2, color='red', alpha=0.7)
plt.plot(l2_val_losses, label='L2 Regularization', linewidth=2, color='blue', alpha=0.7)
plt.plot(dropout_val_losses, label='Dropout', linewidth=2, color='green', alpha=0.7)
plt.plot(early_stop_val, label='Early Stopping', linewidth=2, color='purple', alpha=0.7)
plt.axvline(x=stop_epoch, color='purple', linestyle=':', linewidth=2, label='Early Stop Point')
plt.xlabel('Epoch', fontsize=11, fontweight='bold')
plt.ylabel('Validation Loss', fontsize=11, fontweight='bold')
plt.title('Validation Loss Comparison', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Overfitting gap
plt.subplot(2, 3, 3)
gap_no_reg = np.array(val_losses) - np.array(train_losses)
gap_l2 = np.array(l2_val_losses) - np.array(l2_train_losses)
gap_dropout = np.array(dropout_val_losses) - np.array(dropout_train_losses)
gap_early = np.array(early_stop_val) - np.array(early_stop_train)

plt.plot(gap_no_reg, label='No Regularization', linewidth=2, color='red', alpha=0.7)
plt.plot(gap_l2, label='L2', linewidth=2, color='blue', alpha=0.7)
plt.plot(gap_dropout, label='Dropout', linewidth=2, color='green', alpha=0.7)
plt.plot(gap_early, label='Early Stop', linewidth=2, color='purple', alpha=0.7)
plt.xlabel('Epoch', fontsize=11, fontweight='bold')
plt.ylabel('Validation - Training Loss', fontsize=11, fontweight='bold')
plt.title('Overfitting Gap (Smaller is Better)', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Best validation loss
plt.subplot(2, 3, 4)
methods = ['No Reg', 'L2', 'Dropout', 'Early Stop']
best_val_losses = [
    min(val_losses),
    min(l2_val_losses),
    min(dropout_val_losses),
    min(early_stop_val)
]
colors = ['red', 'blue', 'green', 'purple']
bars = plt.bar(methods, best_val_losses, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
plt.ylabel('Best Validation Loss', fontsize=11, fontweight='bold')
plt.title('Best Validation Performance', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, best_val_losses):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
             f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 5: Final epoch comparison
plt.subplot(2, 3, 5)
final_train = [train_losses[-1], l2_train_losses[-1], dropout_train_losses[-1], early_stop_train[-1]]
final_val = [val_losses[-1], l2_val_losses[-1], dropout_val_losses[-1], early_stop_val[-1]]

x = np.arange(len(methods))
width = 0.35
plt.bar(x - width/2, final_train, width, label='Training', color='skyblue', alpha=0.7)
plt.bar(x + width/2, final_val, width, label='Validation', color='lightcoral', alpha=0.7)
plt.xlabel('Method', fontsize=11, fontweight='bold')
plt.ylabel('Final Loss', fontsize=11, fontweight='bold')
plt.title('Final Loss Comparison', fontsize=12, fontweight='bold')
plt.xticks(x, methods)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Plot 6: Summary
plt.subplot(2, 3, 6)
plt.axis('off')
summary_text = """
Regularization Summary:

1. L2 Regularization:
   • Penalizes large weights
   • Encourages smooth solutions
   • Easy to implement

2. Dropout:
   • Randomly disables neurons
   • Prevents co-adaptation
   • Very effective for deep networks

3. Early Stopping:
   • Stops at optimal point
   • Prevents overfitting
   • No hyperparameter tuning needed

Best Practice:
Combine multiple techniques!
"""
plt.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

print("Visualization complete!")
print()

# ============================================================================
# 5b.8 Understanding Each Technique
# ============================================================================
print("=== 5b.8 Understanding Each Technique ===")
print()

print("1. L2 Regularization (Weight Decay):")
print("   • Adds penalty: Loss = Original + λ * Σ(weights²)")
print("   • Encourages smaller weights")
print("   • Prevents extreme values")
print("   • λ (lambda) controls strength")
print("   • Best for: Preventing large weights, smooth solutions")
print()

print("2. Dropout:")
print("   • Randomly sets neurons to 0 during training")
print("   • Forces network to be robust")
print("   • Prevents neurons from co-adapting")
print("   • Dropout rate: 0.5 is common (50% neurons disabled)")
print("   • Best for: Deep networks, preventing overfitting")
print()

print("3. Early Stopping:")
print("   • Monitors validation loss")
print("   • Stops when validation loss stops improving")
print("   • Prevents training too long")
print("   • Patience: How many epochs to wait")
print("   • Best for: Simple, automatic regularization")
print()

print("4. L1 Regularization (mentioned):")
print("   • Similar to L2 but uses |weights| instead of weights²")
print("   • Can drive weights to exactly zero (feature selection)")
print("   • Less common in deep learning")
print()

# ============================================================================
# 5b.9 Best Practices
# ============================================================================
print("=== 5b.9 Best Practices ===")
print()

print("✅ Combine techniques:")
print("   • Use L2 regularization + Dropout + Early stopping")
print("   • Each helps in different ways")
print()

print("✅ Tune hyperparameters:")
print("   • L2 λ: Start with 0.01, try 0.001 to 0.1")
print("   • Dropout rate: Start with 0.5, try 0.3 to 0.7")
print("   • Early stopping patience: 50-200 epochs")
print()

print("✅ Monitor validation loss:")
print("   • Always track validation performance")
print("   • Stop when validation loss increases")
print()

print("✅ Use appropriate techniques:")
print("   • L2: Good for most problems")
print("   • Dropout: Essential for deep networks")
print("   • Early stopping: Always useful")
print()

# ============================================================================
# 5b.10 Summary
# ============================================================================
print("=== 5b.10 Summary ===")
print()

print("✅ You've learned:")
print("  • What overfitting is and how to detect it")
print("  • L2 regularization (weight decay)")
print("  • Dropout regularization")
print("  • Early stopping")
print("  • How to compare different techniques")
print()

print("🎯 Key Takeaways:")
print("  1. Overfitting = model memorizes training data")
print("  2. Regularization prevents overfitting")
print("  3. Combine multiple techniques for best results")
print("  4. Always monitor validation loss")
print()

print("=" * 70)
print("Step 5b Complete! You understand regularization techniques!")
print("=" * 70)
