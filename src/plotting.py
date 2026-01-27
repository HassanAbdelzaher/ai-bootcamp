"""
Common plotting utilities for AI Codecamp steps
Contains all visualization functions used across different steps
Enhanced with better styling, annotations, and educational elements
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap


def plot_feature_contributions(features, weights, labels=None):
    """Plot feature contributions (Step 0) - Enhanced"""
    contrib = features * weights
    if labels is None:
        labels = [f"Feature {i+1}" for i in range(len(features))]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart with colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    bars = ax1.bar(labels, contrib, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_title("Feature Contributions", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Contribution", fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(contrib) * 1.1)
    
    # Add value labels on bars
    for bar, val in zip(bars, contrib):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Pie chart showing relative contributions
    total = np.sum(np.abs(contrib))
    if total > 0:
        sizes = np.abs(contrib) / total * 100
        ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
               colors=colors, textprops={'fontsize': 10})
        ax2.set_title("Relative Contributions", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_data_scatter(X, y, xlabel="X", ylabel="Y", title="Data Scatter Plot"):
    """Plot scatter plot of data (Step 1) - Enhanced"""
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Enhanced scatter plot
    scatter = ax.scatter(X, y, s=150, alpha=0.7, c=y, cmap='viridis', 
                        edgecolors='black', linewidth=1.5, zorder=3)
    
    # Add trend line
    if len(X) > 1:
        z = np.polyfit(X, y, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(X.min(), X.max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.5, linewidth=2, 
               label="Trend Line", zorder=1)
    
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    ax.legend(fontsize=10)
    
    # Add statistics text
    stats_text = f"Mean: {np.mean(y):.1f} | Std: {np.std(y):.1f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


def plot_prediction_line(X, y, y_pred, xlabel="X", ylabel="Y", title="Prediction", 
                        label_data="Real Data", label_pred="Prediction", color="green"):
    """Plot data with prediction line (Step 1) - Enhanced"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot data points
    ax.scatter(X, y, label=label_data, s=150, alpha=0.7, c='steelblue',
              edgecolors='black', linewidth=1.5, zorder=3)
    
    # Plot prediction line
    ax.plot(X, y_pred, label=label_pred, color=color, linewidth=3, 
           linestyle='-', marker='o', markersize=8, zorder=2)
    
    # Add error bars/connections
    for i in range(len(X)):
        ax.plot([X[i], X[i]], [y[i], y_pred[i]], 'r--', alpha=0.4, linewidth=1, zorder=1)
    
    # Calculate and display error
    mse = np.mean((y_pred - y) ** 2)
    rmse = np.sqrt(mse)
    
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    ax.legend(fontsize=11, loc='best')
    
    # Add error metrics
    error_text = f"MSE: {mse:.2f} | RMSE: {rmse:.2f}"
    ax.text(0.02, 0.02, error_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.show()


def plot_learning_curve(errors, title="Learning Curve", xlabel="Epoch", ylabel="Error"):
    """Plot learning curve (Steps 1, 3, 4, 6) - Enhanced"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Main learning curve
    epochs = np.arange(len(errors))
    ax1.plot(epochs, errors, linewidth=2, color='steelblue', label='Loss')
    ax1.fill_between(epochs, errors, alpha=0.3, color='steelblue')
    ax1.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax1.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10)
    
    # Add improvement annotation
    if len(errors) > 10:
        initial_error = errors[0]
        final_error = errors[-1]
        improvement = ((initial_error - final_error) / initial_error) * 100
        ax1.annotate(f'Improvement: {improvement:.1f}%', 
                    xy=(len(errors)*0.7, errors[int(len(errors)*0.7)]),
                    xytext=(len(errors)*0.5, errors[0]*0.8),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Log scale view (if errors vary widely)
    if max(errors) / min(errors) > 10:
        ax2.semilogy(epochs, errors, linewidth=2, color='coral', label='Loss (log scale)')
        ax2.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax2.set_ylabel(f"{ylabel} (log scale)", fontsize=12, fontweight='bold')
        ax2.set_title("Learning Curve (Log Scale)", fontsize=14, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, linestyle='--', which='both')
        ax2.legend(fontsize=10)
    else:
        # Show first and last 100 epochs comparison
        window = min(100, len(errors) // 4)
        ax2.plot(epochs[:window], errors[:window], linewidth=2, color='green', 
               label='First 100 epochs', alpha=0.7)
        if len(errors) > window:
            ax2.plot(epochs[-window:], errors[-window:], linewidth=2, color='red',
                   label='Last 100 epochs', alpha=0.7)
        ax2.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax2.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax2.set_title("Learning Progress (Zoom)", fontsize=14, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()


def plot_perceptron_boundary(X, y, boundary, xlabel="Study Hours", ylabel="Pass (0/1)", 
                            title="Perceptron Decision Boundary", color="red", label="Decision Boundary"):
    """Plot perceptron decision boundary (Step 2) - Enhanced"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color code points by class
    colors_map = ['red' if yi == 0 else 'green' for yi in y]
    scatter = ax.scatter(X, y, s=200, alpha=0.7, c=colors_map,
                        edgecolors='black', linewidth=2, zorder=3)
    
    # Decision boundary with shaded regions
    ax.axvline(boundary, linestyle="--", color=color, label=label, 
              linewidth=3, zorder=2)
    
    # Shade decision regions
    x_min, x_max = ax.get_xlim()
    ax.axvspan(x_min, boundary, alpha=0.2, color='red', label='Class 0 Region')
    ax.axvspan(boundary, x_max, alpha=0.2, color='green', label='Class 1 Region')
    
    # Add boundary annotation
    ax.annotate(f'Boundary: {boundary:.2f}', 
               xy=(boundary, 0.5), xytext=(boundary, 0.3),
               arrowprops=dict(arrowstyle='->', color='blue', lw=2),
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
               ha='center')
    
    # Add class labels
    for i, (xi, yi) in enumerate(zip(X, y)):
        ax.text(xi, yi + 0.05, f'({xi}, {int(yi)})', 
               fontsize=9, ha='center', fontweight='bold')
    
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(-0.2, 1.2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['0 (Fail)', '1 (Pass)'])
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    
    plt.tight_layout()
    plt.show()


def plot_sigmoid_function(z_vals=None, title="Sigmoid Function"):
    """Plot sigmoid function (Step 3) - Enhanced"""
    if z_vals is None:
        z_vals = np.linspace(-10, 10, 200)
    s_vals = 1 / (1 + np.exp(-z_vals))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Main sigmoid plot
    ax1.plot(z_vals, s_vals, linewidth=3, color='steelblue', label='Sigmoid(z)')
    ax1.fill_between(z_vals, s_vals, alpha=0.3, color='steelblue')
    ax1.axhline(0.5, linestyle="--", color="red", linewidth=2, 
               alpha=0.7, label="Decision Threshold (0.5)")
    ax1.axvline(0, linestyle="--", color="gray", linewidth=1, alpha=0.5)
    ax1.axhline(0, linestyle="--", color="gray", linewidth=1, alpha=0.5)
    ax1.set_xlabel("z (input)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Probability", fontsize=12, fontweight='bold')
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11)
    ax1.set_ylim(-0.1, 1.1)
    
    # Add key points annotation
    key_points = [-5, -2, 0, 2, 5]
    for z in key_points:
        s = 1 / (1 + np.exp(-z))
        ax1.plot(z, s, 'ro', markersize=10, zorder=5)
        ax1.annotate(f'({z}, {s:.2f})', xy=(z, s), 
                    xytext=(z, s + 0.15), fontsize=9, fontweight='bold',
                    ha='center', arrowprops=dict(arrowstyle='->', alpha=0.5))
    
    # Comparison with step function
    step_vals = np.where(z_vals >= 0, 1, 0)
    ax2.plot(z_vals, s_vals, linewidth=3, color='steelblue', 
            label='Sigmoid (smooth)', alpha=0.8)
    ax2.plot(z_vals, step_vals, linewidth=2, color='red', 
            linestyle='--', label='Step Function (hard)', alpha=0.8)
    ax2.axhline(0.5, linestyle=":", color="gray", linewidth=1, alpha=0.5)
    ax2.set_xlabel("z (input)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Output", fontsize=12, fontweight='bold')
    ax2.set_title("Sigmoid vs Step Function", fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=11)
    ax2.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.show()


def plot_probability_curve(x_vals, probs, X=None, y=None, threshold=0.5, 
                          xlabel="Study Hours", ylabel="Pass Probability", 
                          title="Logistic Regression Curve"):
    """Plot probability curve (Step 3) - Enhanced"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Probability curve with gradient fill
    ax.plot(x_vals, probs, linewidth=3, color='steelblue', label="Probability Curve", zorder=2)
    ax.fill_between(x_vals, probs, alpha=0.3, color='steelblue', zorder=1)
    
    # Decision threshold
    ax.axhline(threshold, linestyle="--", color="red", linewidth=2, 
              label=f"Decision Threshold ({threshold})", zorder=3)
    
    # Shade decision regions
    ax.fill_between(x_vals, 0, threshold, alpha=0.2, color='red', label='Class 0 Region')
    ax.fill_between(x_vals, threshold, 1, alpha=0.2, color='green', label='Class 1 Region')
    
    # Data points
    if X is not None and y is not None:
        colors = ['red' if yi == 0 else 'green' for yi in y]
        ax.scatter(X, y, s=200, alpha=0.8, c=colors, edgecolors='black', 
                  linewidth=2, label="Data Points", zorder=5)
        
        # Add labels for data points
        for xi, yi in zip(X, y):
            ax.annotate(f'({xi}, {int(yi)})', xy=(xi, yi), 
                       xytext=(xi, yi + 0.1), fontsize=9, fontweight='bold',
                       ha='center', arrowprops=dict(arrowstyle='->', alpha=0.5))
    
    # Find decision point (where curve crosses threshold)
    crossing_idx = np.argmin(np.abs(probs - threshold))
    if len(x_vals) > crossing_idx:
        crossing_x = x_vals[crossing_idx]
        ax.plot(crossing_x, threshold, 'ro', markersize=12, zorder=6)
        ax.annotate(f'Decision Point\nx ≈ {crossing_x:.2f}', 
                   xy=(crossing_x, threshold), xytext=(crossing_x, threshold + 0.2),
                   fontsize=10, fontweight='bold', ha='center',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    
    plt.tight_layout()
    plt.show()


def plot_neuron_outputs(A, title="Outputs of Neurons in One Layer", 
                        xlabel="Student Index", ylabel="Activation"):
    """Plot outputs of multiple neurons (Step 4) - Enhanced"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Line plot with different colors
    colors = plt.cm.tab10(np.linspace(0, 1, A.shape[1]))
    for i in range(A.shape[1]):
        ax1.plot(A[:, i], marker='o', label=f"Neuron {i+1}", 
                linewidth=2.5, markersize=8, color=colors[i], alpha=0.8)
    ax1.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax1.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Heatmap view
    im = ax2.imshow(A.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax2.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax2.set_ylabel("Neuron", fontsize=12, fontweight='bold')
    ax2.set_title("Neuron Activations (Heatmap)", fontsize=14, fontweight='bold', pad=15)
    ax2.set_yticks(range(A.shape[1]))
    ax2.set_yticklabels([f"Neuron {i+1}" for i in range(A.shape[1])])
    plt.colorbar(im, ax=ax2, label='Activation Value')
    
    plt.tight_layout()
    plt.show()


def plot_xor_data(X, y, title="XOR Problem (No single line can separate)"):
    """Plot XOR data points (Step 5) - Enhanced"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Enhanced scatter plot
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y.flatten(), s=300, cmap='RdYlGn', 
                        edgecolors='black', linewidth=3, zorder=3, alpha=0.8)
    
    # Add labels for each point
    labels_map = {0: 'Class 0', 1: 'Class 1'}
    for i, (point, label) in enumerate(zip(X, y)):
        ax.annotate(f'({int(point[0])}, {int(point[1])})\n{labels_map[label[0]]}', 
                   xy=(point[0], point[1]), xytext=(point[0], point[1] + 0.15),
                   fontsize=11, fontweight='bold', ha='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   arrowprops=dict(arrowstyle='->', alpha=0.5))
    
    # Try to show why single line fails - draw example lines
    x_line = np.linspace(-0.5, 1.5, 100)
    # Example: horizontal line
    ax.plot(x_line, [0.5]*len(x_line), 'b--', linewidth=2, alpha=0.5, 
           label='Example Line 1 (fails)', zorder=1)
    # Example: diagonal line
    ax.plot(x_line, x_line, 'r--', linewidth=2, alpha=0.5, 
           label='Example Line 2 (fails)', zorder=1)
    
    ax.set_xlabel("x1", fontsize=12, fontweight='bold')
    ax.set_ylabel("x2", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.legend(fontsize=9, loc='upper left')
    
    # Add explanation text
    explanation = "No single straight line\ncan separate the classes!"
    ax.text(0.5, -0.3, explanation, transform=ax.transAxes,
           fontsize=11, fontweight='bold', ha='center',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.show()


def plot_learning_curves_comparison(losses_single, losses_deep, 
                                   title1="Single Layer (Fails)", 
                                   title2="Hidden Layer (Succeeds)"):
    """Plot comparison of learning curves (Step 5) - Enhanced"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    epochs_single = np.arange(len(losses_single))
    epochs_deep = np.arange(len(losses_deep))
    
    # Single layer
    axes[0].plot(epochs_single, losses_single, label="Single Layer", 
                color="red", linewidth=2.5)
    axes[0].fill_between(epochs_single, losses_single, alpha=0.3, color="red")
    axes[0].set_title(title1, fontsize=13, fontweight='bold', pad=15)
    axes[0].set_xlabel("Epoch", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Loss", fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].text(0.5, 0.95, '❌ Cannot learn', transform=axes[0].transAxes,
                fontsize=12, fontweight='bold', ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    # Deep network
    axes[1].plot(epochs_deep, losses_deep, label="Hidden Layer", 
                color="green", linewidth=2.5)
    axes[1].fill_between(epochs_deep, losses_deep, alpha=0.3, color="green")
    axes[1].set_title(title2, fontsize=13, fontweight='bold', pad=15)
    axes[1].set_xlabel("Epoch", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Loss", fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].text(0.5, 0.95, '✅ Learns successfully', transform=axes[1].transAxes,
                fontsize=12, fontweight='bold', ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
    
    # Side-by-side comparison
    min_len = min(len(losses_single), len(losses_deep))
    axes[2].plot(epochs_single[:min_len], losses_single[:min_len], 
                label="Single Layer", color="red", linewidth=2, alpha=0.7)
    axes[2].plot(epochs_deep[:min_len], losses_deep[:min_len], 
                label="Hidden Layer", color="green", linewidth=2, alpha=0.7)
    axes[2].set_title("Direct Comparison", fontsize=13, fontweight='bold', pad=15)
    axes[2].set_xlabel("Epoch", fontsize=11, fontweight='bold')
    axes[2].set_ylabel("Loss", fontsize=11, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()


def plot_decision_regions(xx, yy, Z, X, y, title="XOR Decision Regions (Deep Network)"):
    """Plot decision regions contour (Step 5) - Enhanced"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Enhanced contour plot
    contour = ax.contourf(xx, yy, Z, levels=30, alpha=0.8, cmap='RdYlGn', zorder=1)
    contour_lines = ax.contour(xx, yy, Z, levels=10, colors='black', 
                              linewidths=0.5, alpha=0.3, zorder=2)
    cbar = plt.colorbar(contour, ax=ax, label='Probability', pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    
    # Data points with labels
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y.flatten(), s=400, 
                        cmap='RdYlGn', edgecolors='black', linewidth=3, 
                        zorder=5, alpha=0.9)
    
    # Add point labels
    for i, (point, label) in enumerate(zip(X, y)):
        ax.annotate(f'({int(point[0])}, {int(point[1])})', 
                   xy=(point[0], point[1]), xytext=(point[0], point[1] + 0.2),
                   fontsize=11, fontweight='bold', ha='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                   arrowprops=dict(arrowstyle='->', alpha=0.6))
    
    # Decision boundary (0.5 probability)
    cs = ax.contour(xx, yy, Z, levels=[0.5], colors='blue', 
                   linewidths=3, linestyles='--', zorder=4)
    ax.clabel(cs, inline=True, fontsize=12, fmt='Decision Boundary')
    
    ax.set_xlabel("x1", fontsize=13, fontweight='bold')
    ax.set_ylabel("x2", fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.grid(True, alpha=0.2, linestyle='--', zorder=0)
    
    plt.tight_layout()
    plt.show()


def plot_weight_evolution(weights_history=None, biases_history=None, title="Weight Evolution During Training"):
    """Plot how weights and biases change during training"""
    if weights_history is None:
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot weights
    if isinstance(weights_history, list) and len(weights_history) > 0:
        if isinstance(weights_history[0], np.ndarray):
            # Multiple weights
            for i in range(len(weights_history[0].flatten())):
                w_vals = [w.flatten()[i] for w in weights_history]
                axes[0].plot(w_vals, label=f'Weight {i+1}', linewidth=2, alpha=0.8)
        else:
            # Single weight
            axes[0].plot(weights_history, label='Weight', linewidth=2, color='steelblue')
    
    axes[0].set_xlabel("Epoch", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Weight Value", fontsize=12, fontweight='bold')
    axes[0].set_title("Weight Evolution", fontsize=13, fontweight='bold', pad=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # Plot biases
    if biases_history is not None:
        if isinstance(biases_history[0], np.ndarray):
            for i in range(len(biases_history[0].flatten())):
                b_vals = [b.flatten()[i] for b in biases_history]
                axes[1].plot(b_vals, label=f'Bias {i+1}', linewidth=2, alpha=0.8)
        else:
            axes[1].plot(biases_history, label='Bias', linewidth=2, color='coral')
    
    axes[1].set_xlabel("Epoch", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Bias Value", fontsize=12, fontweight='bold')
    axes[1].set_title("Bias Evolution", fontsize=13, fontweight='bold', pad=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_style(y_true, y_pred, class_names=None):
    """Plot predictions vs actual in a confusion matrix style"""
    if class_names is None:
        class_names = ['Class 0', 'Class 1']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create comparison
    correct = y_true == y_pred
    colors = ['green' if c else 'red' for c in correct]
    
    x_pos = np.arange(len(y_true))
    width = 0.35
    
    ax.bar(x_pos - width/2, y_true, width, label='Actual', color='steelblue', alpha=0.7)
    ax.bar(x_pos + width/2, y_pred, width, label='Predicted', color='coral', alpha=0.7)
    
    # Add correctness indicators
    for i, (is_correct, true_val, pred_val) in enumerate(zip(correct, y_true, y_pred)):
        if is_correct:
            ax.plot(i, max(true_val, pred_val) + 0.1, 'g^', markersize=12)
        else:
            ax.plot(i, max(true_val, pred_val) + 0.1, 'rx', markersize=12)
    
    ax.set_xlabel("Sample Index", fontsize=12, fontweight='bold')
    ax.set_ylabel("Class", fontsize=12, fontweight='bold')
    ax.set_title("Predictions vs Actual", fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(class_names)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    accuracy = np.mean(correct) * 100
    ax.text(0.5, 0.95, f'Accuracy: {accuracy:.1f}%', transform=ax.transAxes,
           fontsize=12, fontweight='bold', ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='lightgreen' if accuracy > 80 else 'lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def plot_error_distribution(y_true, y_pred, title="Error Distribution"):
    """Plot distribution of prediction errors"""
    errors = y_pred - y_true
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(errors, bins=20, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax1.axvline(np.mean(errors), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
    ax1.set_xlabel("Error (Predicted - Actual)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Frequency", fontsize=12, fontweight='bold')
    ax1.set_title("Error Histogram", fontsize=13, fontweight='bold', pad=15)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Scatter plot: Actual vs Predicted
    ax2.scatter(y_true, y_pred, alpha=0.6, s=100, edgecolors='black', linewidth=1.5)
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax2.set_xlabel("Actual Values", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Predicted Values", fontsize=12, fontweight='bold')
    ax2.set_title("Actual vs Predicted", fontsize=13, fontweight='bold', pad=15)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()


def plot_vanishing_gradient(gradients_by_layer, layer_names=None, title="Vanishing Gradient Problem"):
    """
    Visualize how gradients diminish through layers (vanishing gradient problem)
    
    Parameters:
    gradients_by_layer: list of gradient values for each layer (from output to input)
    layer_names: optional list of layer names
    title: plot title
    """
    if layer_names is None:
        layer_names = [f"Layer {i+1}" for i in range(len(gradients_by_layer))]
    
    # Reverse to show from input to output
    gradients = gradients_by_layer[::-1]
    layer_names = layer_names[::-1]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Bar chart showing gradient magnitudes
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(gradients)))
    bars = ax1.barh(layer_names, gradients, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar, grad) in enumerate(zip(bars, gradients)):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2,
                f'{grad:.6f}', ha='left' if width > max(gradients) * 0.1 else 'right',
                va='center', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel("Gradient Magnitude", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Layer (Input → Output)", fontsize=12, fontweight='bold')
    ax1.set_title("Gradient Magnitude by Layer", fontsize=13, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax1.set_xscale('log')  # Log scale to better show the difference
    
    # Add warning annotation for vanishing gradients
    if gradients[0] / gradients[-1] > 100:
        ax1.text(0.5, 0.95, '⚠️ Vanishing Gradient Detected!', 
                transform=ax1.transAxes, fontsize=12, fontweight='bold',
                ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Plot 2: Line plot showing exponential decay
    ax2.plot(range(len(gradients)), gradients, marker='o', linewidth=3, 
            markersize=10, color='steelblue', label='Gradient Magnitude')
    ax2.fill_between(range(len(gradients)), gradients, alpha=0.3, color='steelblue')
    
    # Add exponential decay curve for comparison
    if len(gradients) > 1:
        # Fit exponential decay
        x_fit = np.arange(len(gradients))
        # Simple exponential: y = a * exp(-b*x)
        try:
            from scipy.optimize import curve_fit
            def exp_decay(x, a, b):
                return a * np.exp(-b * x)
            popt, _ = curve_fit(exp_decay, x_fit, gradients, p0=[gradients[0], 0.5])
            y_fit = exp_decay(x_fit, *popt)
            ax2.plot(x_fit, y_fit, 'r--', linewidth=2, alpha=0.7, 
                    label=f'Exponential Decay (rate={popt[1]:.2f})')
        except:
            # If scipy not available, just show the data
            pass
    
    ax2.set_xlabel("Layer Number (Input → Output)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Gradient Magnitude", fontsize=12, fontweight='bold')
    ax2.set_title("Gradient Decay Through Layers", fontsize=13, fontweight='bold', pad=15)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=10)
    ax2.set_xticks(range(len(gradients)))
    ax2.set_xticklabels([f'L{i+1}' for i in range(len(gradients))])
    
    # Add ratio annotation
    if len(gradients) > 1:
        ratio = gradients[0] / gradients[-1]
        ax2.text(0.02, 0.98, f'Input/Output Ratio: {ratio:.1e}x', 
                transform=ax2.transAxes, fontsize=11, fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.show()


def plot_gradient_flow(gradients_history, layer_idx=0, title="Gradient Flow Over Time"):
    """
    Plot how gradients change during training for a specific layer
    
    Parameters:
    gradients_history: list of gradient values over epochs for a layer
    layer_idx: which layer to visualize
    title: plot title
    """
    if not gradients_history:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    epochs = np.arange(len(gradients_history))
    
    # Plot 1: Gradient magnitude over time
    ax1.plot(epochs, gradients_history, linewidth=2, color='steelblue', label='Gradient Magnitude')
    ax1.fill_between(epochs, gradients_history, alpha=0.3, color='steelblue')
    ax1.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Zero')
    
    # Add vanishing threshold
    threshold = 1e-6
    ax1.axhline(threshold, color='orange', linestyle='--', linewidth=1, 
               alpha=0.7, label=f'Vanishing Threshold ({threshold})')
    
    ax1.set_xlabel("Epoch", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Gradient Magnitude", fontsize=12, fontweight='bold')
    ax1.set_title(f"{title} - Layer {layer_idx+1}", fontsize=13, fontweight='bold', pad=15)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=10)
    
    # Plot 2: Gradient change rate
    if len(gradients_history) > 1:
        gradient_changes = np.diff(gradients_history)
        ax2.plot(epochs[1:], gradient_changes, linewidth=2, color='coral', 
                label='Gradient Change')
        ax2.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax2.fill_between(epochs[1:], gradient_changes, alpha=0.3, color='coral')
        
        ax2.set_xlabel("Epoch", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Gradient Change", fontsize=12, fontweight='bold')
        ax2.set_title("Gradient Change Rate", fontsize=13, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()


def plot_roc_curve(fpr, tpr, auc_score=None, title="ROC Curve"):
    """Plot ROC curve with AUC"""
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(fpr, tpr, linewidth=3, color='steelblue', 
           label=f'ROC Curve' + (f' (AUC = {auc_score:.3f})' if auc_score else ''))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.5)', alpha=0.5)
    if auc_score:
        ax.fill_between(fpr, 0, tpr, alpha=0.3, color='steelblue', label='AUC Area')
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.show()


def plot_pr_curve(recalls, precisions, pr_auc=None, title="Precision-Recall Curve"):
    """Plot Precision-Recall curve"""
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(recalls, precisions, linewidth=3, color='coral',
           label=f'PR Curve' + (f' (AUC = {pr_auc:.3f})' if pr_auc else ''))
    if pr_auc:
        ax.fill_between(recalls, 0, precisions, alpha=0.3, color='coral', label='PR-AUC Area')
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.show()


def plot_train_test_split(X_train, y_train, X_test, y_test, y_pred_train=None, y_pred_test=None,
                          xlabel="X", ylabel="Y", title="Train/Test Split Visualization"):
    """Visualize train/test split with optional predictions"""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot training data
    ax.scatter(X_train, y_train, c='blue', marker='o', s=100, alpha=0.6, 
              edgecolors='darkblue', linewidths=1.5, label='Training Data', zorder=3)
    
    # Plot test data
    ax.scatter(X_test, y_test, c='red', marker='s', s=100, alpha=0.6, 
              edgecolors='darkred', linewidths=1.5, label='Test Data', zorder=3)
    
    # Plot predictions if provided
    if y_pred_train is not None:
        ax.plot(X_train, y_pred_train, 'b--', linewidth=2, alpha=0.7, label='Train Predictions')
    if y_pred_test is not None:
        ax.plot(X_test, y_pred_test, 'r--', linewidth=2, alpha=0.7, label='Test Predictions')
    
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add text annotation
    train_ratio = len(X_train) / (len(X_train) + len(X_test)) * 100
    test_ratio = len(X_test) / (len(X_train) + len(X_test)) * 100
    info_text = f'Train: {len(X_train)} ({train_ratio:.1f}%)\nTest: {len(X_test)} ({test_ratio:.1f}%)'
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()


def plot_overfitting(train_losses, val_losses, title="Training vs Validation Loss (Overfitting Detection)"):
    """Visualize overfitting by comparing training and validation losses"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, 'b-', linewidth=2.5, label='Training Loss', alpha=0.8)
    ax.plot(epochs, val_losses, 'r-', linewidth=2.5, label='Validation Loss', alpha=0.8)
    
    # Highlight overfitting region (where val loss starts increasing)
    if len(val_losses) > 10:
        min_val_idx = np.argmin(val_losses)
        if min_val_idx < len(val_losses) - 5:
            ax.axvspan(min_val_idx + 1, len(val_losses), alpha=0.2, color='red', 
                      label='Overfitting Region')
    
    ax.set_xlabel("Epoch", fontsize=12, fontweight='bold')
    ax.set_ylabel("Loss", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add annotation
    if len(val_losses) > 0:
        min_val_loss = min(val_losses)
        min_val_epoch = val_losses.index(min_val_loss) + 1
        ax.annotate(f'Best Val Loss: {min_val_loss:.4f}\n(at epoch {min_val_epoch})',
                   xy=(min_val_epoch, min_val_loss), xytext=(min_val_epoch + len(epochs)*0.2, min_val_loss + max(val_losses)*0.1),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2),
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.show()


def plot_word_embeddings(embeddings, words=None, title="Word Embeddings Visualization (2D Projection)"):
    """Visualize word embeddings in 2D using PCA or t-SNE"""
    from sklearn.decomposition import PCA
    
    # Use PCA to reduce to 2D for visualization
    if embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        explained_var = sum(pca.explained_variance_ratio_)
    else:
        embeddings_2d = embeddings
        explained_var = 1.0
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot embeddings
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                        s=150, alpha=0.6, c=range(len(embeddings_2d)), 
                        cmap='viridis', edgecolors='black', linewidths=1)
    
    # Add word labels if provided
    if words is not None:
        for i, word in enumerate(words):
            ax.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                       fontsize=9, ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax.set_xlabel(f'First Principal Component', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Second Principal Component', fontsize=12, fontweight='bold')
    ax.set_title(f'{title}\n(Explained Variance: {explained_var:.1%})', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.colorbar(scatter, ax=ax, label='Word Index')
    plt.tight_layout()
    plt.show()


def plot_model_architecture(model, input_shape=None, title="Model Architecture"):
    """Create a simple text-based model architecture diagram"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Get model structure
    layers = []
    if hasattr(model, 'named_children'):
        for name, module in model.named_children():
            layers.append((name, str(type(module).__name__), 
                          sum(p.numel() for p in module.parameters())))
    else:
        layers = [("Model", str(type(model).__name__), 
                  sum(p.numel() for p in model.parameters()))]
    
    # Create visualization
    y_positions = np.linspace(0.9, 0.1, len(layers))
    box_height = 0.08
    box_width = 0.4
    
    for i, (name, layer_type, num_params) in enumerate(layers):
        y_pos = y_positions[i]
        
        # Draw box
        rect = mpatches.Rectangle((0.3, y_pos - box_height/2), box_width, box_height,
                                 linewidth=2, edgecolor='steelblue', facecolor='lightblue',
                                 alpha=0.7)
        ax.add_patch(rect)
        
        # Add text
        ax.text(0.5, y_pos, f'{name}\n{layer_type}', 
               ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Add parameter count
        if num_params > 0:
            ax.text(0.75, y_pos, f'{num_params:,} params', 
                   ha='left', va='center', fontsize=9, style='italic')
        
        # Draw arrow
        if i < len(layers) - 1:
            ax.arrow(0.5, y_pos - box_height/2 - 0.02, 0, -0.05,
                    head_width=0.02, head_length=0.01, fc='black', ec='black')
    
    # Add title and info
    ax.text(0.5, 0.95, title, ha='center', va='top', 
           fontsize=16, fontweight='bold', transform=ax.transAxes)
    
    total_params = sum(p.numel() for p in model.parameters())
    if input_shape:
        info_text = f'Input Shape: {input_shape}\nTotal Parameters: {total_params:,}'
    else:
        info_text = f'Total Parameters: {total_params:,}'
    
    ax.text(0.5, 0.05, info_text, ha='center', va='bottom', 
           fontsize=11, transform=ax.transAxes,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
