"""
Common plotting utilities for AI Codecamp steps
Contains all visualization functions used across different steps
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_feature_contributions(features, weights, labels=None):
    """Plot feature contributions (Step 0)"""
    contrib = features * weights
    if labels is None:
        labels = [f"Feature {i+1}" for i in range(len(features))]
    
    plt.figure(figsize=(8, 5))
    plt.bar(labels, contrib)
    plt.title("Feature Contributions")
    plt.ylabel("Contribution")
    plt.show()


def plot_data_scatter(X, y, xlabel="X", ylabel="Y", title="Data Scatter Plot"):
    """Plot scatter plot of data (Step 1)"""
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, s=100, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_prediction_line(X, y, y_pred, xlabel="X", ylabel="Y", title="Prediction", 
                        label_data="Real Data", label_pred="Prediction", color="green"):
    """Plot data with prediction line (Step 1)"""
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, label=label_data, s=100, alpha=0.7)
    plt.plot(X, y_pred, label=label_pred, color=color, linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_learning_curve(errors, title="Learning Curve", xlabel="Epoch", ylabel="Error"):
    """Plot learning curve (Steps 1, 3, 4, 6)"""
    plt.figure(figsize=(8, 5))
    plt.plot(errors)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_perceptron_boundary(X, y, boundary, xlabel="Study Hours", ylabel="Pass (0/1)", 
                            title="Perceptron Decision Boundary", color="red", label="Decision Boundary"):
    """Plot perceptron decision boundary (Step 2)"""
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, s=100, alpha=0.7, label="Data")
    plt.axvline(boundary, linestyle="--", color=color, label=label, linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_sigmoid_function(z_vals=None, title="Sigmoid Function"):
    """Plot sigmoid function (Step 3)"""
    if z_vals is None:
        z_vals = np.linspace(-10, 10, 200)
    s_vals = 1 / (1 + np.exp(-z_vals))
    
    plt.figure(figsize=(8, 5))
    plt.plot(z_vals, s_vals, linewidth=2)
    plt.title(title)
    plt.xlabel("z")
    plt.ylabel("Probability")
    plt.grid(True, alpha=0.3)
    plt.axhline(0.5, linestyle="--", color="red", alpha=0.5, label="0.5 threshold")
    plt.legend()
    plt.show()


def plot_probability_curve(x_vals, probs, X=None, y=None, threshold=0.5, 
                          xlabel="Study Hours", ylabel="Pass Probability", 
                          title="Logistic Regression Curve"):
    """Plot probability curve (Step 3)"""
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, probs, linewidth=2, label="Probability Curve")
    plt.axhline(threshold, linestyle="--", color="red", label=f"Decision Threshold ({threshold})")
    if X is not None and y is not None:
        plt.scatter(X, y, s=100, alpha=0.7, color="green", label="Data Points", zorder=5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_neuron_outputs(A, title="Outputs of Neurons in One Layer", 
                        xlabel="Student Index", ylabel="Activation"):
    """Plot outputs of multiple neurons (Step 4)"""
    plt.figure(figsize=(8, 5))
    for i in range(A.shape[1]):
        plt.plot(A[:, i], marker='o', label=f"Neuron {i+1}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_xor_data(X, y, title="XOR Problem (No single line can separate)"):
    """Plot XOR data points (Step 5)"""
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), s=200, cmap='coolwarm', 
                edgecolors='black', linewidth=2)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.show()


def plot_learning_curves_comparison(losses_single, losses_deep, 
                                   title1="Single Layer (Fails)", 
                                   title2="Hidden Layer (Succeeds)"):
    """Plot comparison of learning curves (Step 5)"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses_single, label="Single Layer", color="red")
    plt.title(title1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(losses_deep, label="Hidden Layer", color="green")
    plt.title(title2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_decision_regions(xx, yy, Z, X, y, title="XOR Decision Regions (Deep Network)"):
    """Plot decision regions contour (Step 5)"""
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, levels=20, alpha=0.7, cmap='coolwarm')
    plt.colorbar(label='Probability')
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), s=200, cmap='coolwarm', 
                edgecolors='black', linewidth=2, zorder=5)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.show()
