"""
Example: Visualizing the Vanishing Gradient Problem
Demonstrates how gradients diminish through deep networks
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore', message='.*Failed to initialize NumPy.*')

from plotting import plot_vanishing_gradient, plot_gradient_flow

print("=" * 70)
print("Example: Vanishing Gradient Problem")
print("=" * 70)
print()

# 1. Understanding the Problem
print("=== 1. Understanding the Vanishing Gradient Problem ===")
print()
print("What is it?")
print("  - Gradients become very small as they propagate backward")
print("  - Early layers receive tiny gradients")
print("  - Makes it hard to train deep networks")
print()
print("Why does it happen?")
print("  - Chain rule multiplies many small values")
print("  - Activation functions (like sigmoid) compress values")
print("  - Deep networks = many multiplications = exponential decay")
print()

# 2. Simulate Gradient Flow Through Layers
print("=== 2. Simulating Gradient Flow Through Layers ===")
print()

def sigmoid_derivative(x):
    """Derivative of sigmoid function"""
    s = 1 / (1 + np.exp(-x))
    return s * (1 - s)

# Simulate a deep network with sigmoid activations
num_layers = 8
print(f"Simulating {num_layers}-layer network with sigmoid activations...")
print()

# Start with a gradient at the output layer
initial_gradient = 1.0
print(f"Initial gradient at output layer: {initial_gradient}")
print()

# Propagate backward through layers
gradients = [initial_gradient]
layer_names = ["Output"]

# Typical sigmoid derivative values (small, around 0.25)
sigmoid_deriv_avg = 0.25  # Average value of sigmoid'(x)

print("Propagating gradient backward through layers:")
print("-" * 60)
print(f"Layer {'Output':>15s}: Gradient = {initial_gradient:.6f}")

for i in range(num_layers - 1):
    # Each layer multiplies by sigmoid derivative (typically small)
    # Also multiply by weight (assume weights around 0.5)
    weight_factor = 0.5
    layer_gradient = gradients[-1] * sigmoid_deriv_avg * weight_factor
    gradients.append(layer_gradient)
    layer_names.append(f"Hidden {num_layers - i - 1}")
    
    reduction = gradients[-2] / gradients[-1] if gradients[-1] > 0 else float('inf')
    print(f"Layer {layer_names[-1]:>15s}: Gradient = {layer_gradient:.6f} "
          f"(reduced by {reduction:.1f}x)")

layer_names.append("Input")
print(f"Layer {'Input':>15s}: Gradient = {gradients[-1]:.6f}")
print()

# 3. Visualize Vanishing Gradient
print("=== 3. Visualizing Vanishing Gradient ===")
print()
print("The visualization shows:")
print("  - Left plot: Gradient magnitude by layer (bar chart)")
print("  - Right plot: Exponential decay through layers")
print("  - Warning if gradient reduction > 100x")
print()

plot_vanishing_gradient(gradients, layer_names, 
                       title="Vanishing Gradient in Deep Network")

# 4. Compare Different Activation Functions
print("=== 4. Comparing Different Activation Functions ===")
print()

# ReLU derivative (typically 1.0 for positive inputs)
relu_deriv_avg = 1.0

print("Gradient flow with ReLU activation:")
print("-" * 60)
relu_gradients = [initial_gradient]
for i in range(num_layers - 1):
    layer_gradient = relu_gradients[-1] * relu_deriv_avg * weight_factor
    relu_gradients.append(layer_gradient)
    print(f"Layer {i+1:2d}: Gradient = {layer_gradient:.6f}")

print()
print("Comparison:")
print(f"  Sigmoid (last layer): {gradients[-1]:.6f}")
print(f"  ReLU (last layer):    {relu_gradients[-1]:.6f}")
print(f"  ReLU is {relu_gradients[-1] / gradients[-1]:.1f}x better!")
print()

# 5. Solutions to Vanishing Gradient
print("=== 5. Solutions to Vanishing Gradient ===")
print()
print("✅ Use ReLU instead of sigmoid:")
print("   - ReLU derivative = 1 for positive inputs")
print("   - Gradients don't vanish as quickly")
print()
print("✅ Batch Normalization:")
print("   - Normalizes activations")
print("   - Helps gradients flow better")
print()
print("✅ Residual Connections (ResNet):")
print("   - Skip connections allow direct gradient flow")
print("   - Gradients can bypass layers")
print()
print("✅ Gradient Clipping:")
print("   - Prevents gradients from becoming too small")
print("   - Also prevents exploding gradients")
print()
print("✅ Better Weight Initialization:")
print("   - Xavier/He initialization")
print("   - Keeps activations in good range")
print()

# 6. Gradient Flow Over Time
print("=== 6. Gradient Flow During Training ===")
print()
print("Simulating gradient changes during training...")
print()

# Simulate gradient diminishing over training
epochs = 100
gradient_history = []
current_grad = initial_gradient

for epoch in range(epochs):
    # Gradient gradually decreases (common in deep networks)
    decay_rate = 0.995
    current_grad *= decay_rate
    # Add some noise
    noise = np.random.normal(0, current_grad * 0.1)
    current_grad = max(current_grad + noise, 1e-8)
    gradient_history.append(current_grad)

print(f"Initial gradient: {gradient_history[0]:.6f}")
print(f"Final gradient:   {gradient_history[-1]:.6f}")
print(f"Reduction:        {gradient_history[0] / gradient_history[-1]:.1f}x")
print()

plot_gradient_flow(gradient_history, layer_idx=0, 
                  title="Gradient Vanishing During Training")

# 7. Real-World Impact
print("=== 7. Real-World Impact ===")
print()
print("Vanishing gradient affects:")
print("  ❌ Very deep networks (>10 layers) with sigmoid/tanh")
print("  ❌ Recurrent Neural Networks (RNNs)")
print("  ❌ Long sequences in RNNs/LSTMs")
print()
print("Modern solutions:")
print("  ✅ ResNet (2015): Residual connections")
print("  ✅ Batch Normalization (2015)")
print("  ✅ ReLU and variants (Leaky ReLU, ELU)")
print("  ✅ Transformer architecture (2017)")
print()

print("=" * 70)
print("Key Takeaway:")
print("  Vanishing gradients make it hard to train deep networks.")
print("  Modern techniques (ReLU, ResNet, BatchNorm) help solve this!")
print("=" * 70)
