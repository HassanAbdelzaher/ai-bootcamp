# Project 6b: Optimization Techniques

> **Compare and master different optimization algorithms for neural networks**

**Difficulty**: ⭐⭐ Intermediate  
**Time**: 2-3 hours  
**Prerequisites**: Steps 0-6 (Especially Step 6: PyTorch)

---

## 📚 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem: Optimizer Comparison](#problem-optimizer-comparison)
3. [Key Concepts](#key-concepts)
4. [Step-by-Step Guide](#step-by-step-guide)
5. [Expected Results](#expected-results)
6. [Extension Ideas](#extension-ideas)

---

## 🎯 Project Overview

This project teaches you to compare and use different optimization algorithms. You'll learn to:

- Compare SGD, Adam, RMSprop optimizers
- Use learning rate scheduling
- Understand momentum and adaptive methods
- Choose the right optimizer for your problem

### Why Different Optimizers?

- **Different problems need different optimizers**: Some converge faster
- **Learning rate matters**: Wrong LR = slow or unstable training
- **Real-world impact**: Can make training 10x faster
- **Essential skill**: Every ML engineer needs this knowledge

---

## 📋 Problem: Optimizer Comparison

### Task

Train the same model with different optimizers and compare:
1. **SGD**: Basic gradient descent
2. **SGD with Momentum**: Faster convergence
3. **Adam**: Adaptive learning rates
4. **RMSprop**: Adaptive for RNNs
5. **Learning Rate Scheduling**: Adjust LR during training

### Learning Objectives

- Understand how optimizers work
- Compare convergence speeds
- Use learning rate schedulers
- Choose optimal optimizer for your task

---

## 🧠 Key Concepts

### 1. SGD (Stochastic Gradient Descent)

**Basic optimizer**: Updates weights using gradient

**Formula**:
```
w = w - lr × gradient
```

**Pros**: Simple, works for most problems
**Cons**: Can be slow, sensitive to learning rate

### 2. SGD with Momentum

**Adds momentum**: Remembers previous updates

**Formula**:
```
velocity = momentum × velocity + gradient
w = w - lr × velocity
```

**Pros**: Faster convergence, smoother updates
**Cons**: Need to tune momentum parameter

### 3. Adam (Adaptive Moment Estimation)

**Adaptive learning rates**: Different LR for each parameter

**Pros**: Fast convergence, works well out-of-the-box
**Cons**: More memory, can sometimes generalize worse

### 4. Learning Rate Scheduling

**Adjust LR during training**: Start high, decrease over time

**Strategies**:
- StepLR: Reduce every N epochs
- ExponentialLR: Exponential decay
- CosineAnnealingLR: Cosine schedule

---

## 🚀 Step-by-Step Guide

### Step 1: Create Dataset

```python
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create classification dataset
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
```

### Step 2: Define Model

```python
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
```

### Step 3: Compare Optimizers

```python
import torch.optim as optim

optimizers = {
    'SGD': optim.SGD,
    'SGD+Momentum': lambda params, lr: optim.SGD(params, lr, momentum=0.9),
    'Adam': optim.Adam,
    'RMSprop': optim.RMSprop
}

results = {}

for opt_name, opt_class in optimizers.items():
    # Create fresh model
    model = SimpleNN()
    criterion = nn.BCELoss()
    optimizer = opt_class(model.parameters(), lr=0.01)
    
    losses = []
    epochs = 100
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 20 == 0:
            print(f"{opt_name} - Epoch {epoch+1}: Loss = {loss.item():.4f}")
    
    results[opt_name] = losses
```

### Step 4: Visualize Comparison

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
for name, losses in results.items():
    plt.plot(losses, label=name)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Optimizer Comparison')
plt.legend()
plt.grid(True)
plt.show()
```

### Step 5: Learning Rate Scheduling

```python
# Model with learning rate scheduler
model = SimpleNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Different schedulers
schedulers = {
    'StepLR': optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1),
    'ExponentialLR': optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95),
    'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
}

scheduler_results = {}

for sched_name, scheduler in schedulers.items():
    model = SimpleNN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = schedulers[sched_name]
    
    losses = []
    learning_rates = []
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        scheduler.step()
    
    scheduler_results[sched_name] = {
        'losses': losses,
        'learning_rates': learning_rates
    }
```

### Step 6: Visualize Learning Rates

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, (name, data) in enumerate(scheduler_results.items()):
    axes[0, idx].plot(data['losses'])
    axes[0, idx].set_title(f'{name} - Loss')
    axes[0, idx].set_xlabel('Epoch')
    axes[0, idx].set_ylabel('Loss')
    
    axes[1, idx].plot(data['learning_rates'])
    axes[1, idx].set_title(f'{name} - Learning Rate')
    axes[1, idx].set_xlabel('Epoch')
    axes[1, idx].set_ylabel('LR')

plt.tight_layout()
plt.show()
```

---

## 📊 Expected Results

### Optimizer Comparison

```
SGD:
  Final Loss: 0.2345
  Convergence: Slow, steady

SGD+Momentum:
  Final Loss: 0.1987
  Convergence: Faster, smoother

Adam:
  Final Loss: 0.1234
  Convergence: Fastest, stable

RMSprop:
  Final Loss: 0.1456
  Convergence: Fast, good for RNNs
```

### Learning Rate Schedules

```
StepLR: Reduces LR every 30 epochs
ExponentialLR: Gradual exponential decay
CosineAnnealingLR: Smooth cosine curve
```

---

## 💡 Extension Ideas

1. **Hyperparameter Tuning**
   - Find optimal learning rates
   - Tune momentum values
   - Optimize scheduler parameters

2. **Advanced Optimizers**
   - AdamW (weight decay)
   - AdaGrad
   - AdaDelta

3. **Warm-up Schedules**
   - Linear warm-up
   - Cosine warm-up
   - Compare with/without warm-up

---

## ✅ Success Criteria

- ✅ Compare all major optimizers
- ✅ Understand when to use each
- ✅ Implement learning rate scheduling
- ✅ Visualize optimizer performance
- ✅ Choose best optimizer for task

---

**Ready to optimize your training? Let's compare optimizers!** 🚀
