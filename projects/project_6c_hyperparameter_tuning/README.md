# Project 6c: Hyperparameter Tuning

> **Systematically find the best hyperparameters for your models**

**Difficulty**: ⭐⭐⭐ Advanced  
**Time**: 3-4 hours  
**Prerequisites**: Steps 0-6 (Especially Step 6: PyTorch)

---

## 📚 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem: Find Optimal Hyperparameters](#problem-find-optimal-hyperparameters)
3. [Key Concepts](#key-concepts)
4. [Step-by-Step Guide](#step-by-step-guide)
5. [Expected Results](#expected-results)
6. [Extension Ideas](#extension-ideas)

---

## 🎯 Project Overview

This project teaches you systematic approaches to find the best hyperparameters. You'll learn to:

- Use grid search for hyperparameter tuning
- Implement random search
- Understand hyperparameter importance
- Optimize learning rates, batch sizes, architectures

### Why Hyperparameter Tuning?

- **Huge impact**: Can improve accuracy by 10-20%
- **Systematic approach**: Better than random guessing
- **Time-saving**: Find good params faster
- **Essential skill**: Required for production models

---

## 📋 Problem: Find Optimal Hyperparameters

### Task

Find the best hyperparameters for a neural network using:
1. **Grid Search**: Exhaustive search over parameter grid
2. **Random Search**: Random sampling of parameter space
3. **Learning Rate Tuning**: Find optimal learning rate
4. **Architecture Search**: Find best network size

### Learning Objectives

- Understand hyperparameter impact
- Implement grid search
- Use random search effectively
- Analyze hyperparameter importance
- Optimize model performance

---

## 🧠 Key Concepts

### 1. Hyperparameters vs Parameters

**Parameters**: Learned during training (weights, biases)
**Hyperparameters**: Set before training (learning rate, batch size, architecture)

### 2. Grid Search

**Method**: Try all combinations of hyperparameters

**Pros**: Guaranteed to find best in grid
**Cons**: Computationally expensive

### 3. Random Search

**Method**: Randomly sample hyperparameter space

**Pros**: Faster, often finds better params
**Cons**: No guarantee of finding best

### 4. Hyperparameter Importance

**Some matter more**: Learning rate > batch size
**Interactions**: Some params work well together
**Problem-dependent**: Best params vary by task

---

## 🚀 Step-by-Step Guide

### Step 1: Create Dataset

```python
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=1000, n_features=20, n_classes=2, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
```

### Step 2: Define Model with Hyperparameters

```python
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
```

### Step 3: Grid Search

```python
from itertools import product

def train_and_evaluate(params):
    """Train model with given hyperparameters"""
    model = TunableNN(
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    )
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params['learning_rate']
    )
    
    # Training
    epochs = 50
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = (outputs >= 0.5).float()
        accuracy = (predictions == y_test_tensor).float().mean().item()
    
    return accuracy

# Define parameter grid
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'hidden_size': [32, 64, 128],
    'num_layers': [1, 2, 3],
    'dropout': [0.0, 0.2, 0.5]
}

# Grid search
best_score = 0
best_params = None
results = []

print("Running Grid Search...")
for params in product(*param_grid.values()):
    param_dict = dict(zip(param_grid.keys(), params))
    score = train_and_evaluate(param_dict)
    results.append((param_dict, score))
    
    if score > best_score:
        best_score = score
        best_params = param_dict
    
    print(f"Params: {param_dict}, Score: {score:.4f}")

print(f"\nBest Params: {best_params}")
print(f"Best Score: {best_score:.4f}")
```

### Step 4: Random Search

```python
import random

def random_search(n_trials=50):
    """Random search for hyperparameters"""
    best_score = 0
    best_params = None
    results = []
    
    for trial in range(n_trials):
        # Randomly sample hyperparameters
        params = {
            'learning_rate': random.choice([0.0001, 0.001, 0.01, 0.1]),
            'hidden_size': random.choice([32, 64, 128, 256]),
            'num_layers': random.randint(1, 4),
            'dropout': random.choice([0.0, 0.1, 0.2, 0.3, 0.5])
        }
        
        score = train_and_evaluate(params)
        results.append((params, score))
        
        if score > best_score:
            best_score = score
            best_params = params
        
        if (trial + 1) % 10 == 0:
            print(f"Trial {trial+1}/{n_trials}, Best Score: {best_score:.4f}")
    
    return best_params, best_score, results

print("Running Random Search...")
best_params_rand, best_score_rand, rand_results = random_search(n_trials=50)
print(f"\nBest Params: {best_params_rand}")
print(f"Best Score: {best_score_rand:.4f}")
```

### Step 5: Learning Rate Tuning

```python
def find_optimal_lr(model, criterion, train_loader, lr_range=(1e-5, 1e-1)):
    """Find optimal learning rate using learning rate finder"""
    lrs = []
    losses = []
    
    for lr in np.logspace(np.log10(lr_range[0]), np.log10(lr_range[1]), 50):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # One epoch training
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        lrs.append(lr)
        losses.append(avg_loss)
    
    # Find LR with steepest decrease
    best_idx = np.argmin(losses)
    optimal_lr = lrs[best_idx]
    
    return lrs, losses, optimal_lr

# Find optimal learning rate
model = TunableNN()
criterion = nn.BCELoss()
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
    batch_size=32
)

lrs, losses, optimal_lr = find_optimal_lr(model, criterion, train_loader)
print(f"Optimal Learning Rate: {optimal_lr:.6f}")

# Visualize
plt.plot(lrs, losses)
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('Learning Rate Finder')
plt.axvline(optimal_lr, color='r', linestyle='--', label=f'Optimal LR: {optimal_lr:.6f}')
plt.legend()
plt.show()
```

### Step 6: Analyze Hyperparameter Importance

```python
import pandas as pd

# Convert results to DataFrame
df = pd.DataFrame([
    {**params, 'score': score} for params, score in results
])

# Analyze importance
print("\nHyperparameter Importance:")
print("=" * 50)

for param in param_grid.keys():
    importance = df.groupby(param)['score'].mean()
    print(f"\n{param}:")
    for value, mean_score in importance.items():
        print(f"  {value}: {mean_score:.4f}")
```

---

## 📊 Expected Results

### Grid Search Results

```
Total combinations: 81
Best Params:
  learning_rate: 0.01
  hidden_size: 128
  num_layers: 2
  dropout: 0.2
Best Score: 0.9123
```

### Random Search Results

```
Trials: 50
Best Params:
  learning_rate: 0.01
  hidden_size: 128
  num_layers: 2
  dropout: 0.2
Best Score: 0.9156
(Found similar result with fewer evaluations!)
```

### Learning Rate Finder

```
Optimal Learning Rate: 0.007943
(Steepest loss decrease at this LR)
```

---

## 💡 Extension Ideas

1. **Bayesian Optimization**
   - Use libraries like Optuna
   - More efficient than random search
   - Learn from previous trials

2. **Automated Hyperparameter Tuning**
   - AutoML frameworks
   - Neural architecture search
   - Automated feature engineering

3. **Multi-Objective Optimization**
   - Optimize accuracy and speed
   - Balance performance and efficiency
   - Pareto frontier analysis

---

## ✅ Success Criteria

- ✅ Implement grid search correctly
- ✅ Use random search effectively
- ✅ Find optimal learning rate
- ✅ Analyze hyperparameter importance
- ✅ Improve model performance

---

**Ready to find the best hyperparameters? Let's tune your models!** 🚀
