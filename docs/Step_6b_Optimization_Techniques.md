# Step 6b: Optimization Techniques

> **Learn about different optimizers and when to use each one**

**Time**: ~60 minutes  
**Prerequisites**: Step 6 (PyTorch basics)

---

## 🎯 Learning Objectives

By the end of this step, you'll understand:
- Different optimization algorithms (SGD, Adam, RMSprop)
- How optimizers compare in practice
- Learning rate scheduling
- When to use each optimizer
- Common pitfalls and best practices

---

## 📚 What Are Optimizers?

Optimizers determine **how** neural networks update their weights during training. Different optimizers use different strategies to find the best weights.

### The Basic Idea

All optimizers try to minimize the loss function by updating weights:
```
new_weight = old_weight - learning_rate * gradient
```

But different optimizers calculate this update differently!

---

## 🔍 Optimizer Types

### 1. SGD (Stochastic Gradient Descent)

**The simplest optimizer**

```python
optimizer = optim.SGD(model.parameters(), lr=0.1)
```

**How it works:**
- Directly uses the gradient
- Updates: `w = w - lr * gradient`

**Pros:**
- Simple and predictable
- Good for large datasets
- Often better generalization

**Cons:**
- Can be slow
- Sensitive to learning rate
- May get stuck in local minima

**Best for:**
- Simple problems
- When you want full control
- Training from scratch on large datasets

---

### 2. SGD with Momentum

**SGD with "velocity"**

```python
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
```

**How it works:**
- Adds velocity to gradient descent
- Formula: `v = momentum * v + gradient`
- Update: `w = w - lr * v`

**Pros:**
- Faster convergence than SGD
- Helps escape local minima
- Smoother updates

**Cons:**
- Still needs learning rate tuning
- Can overshoot minimum

**Best for:**
- When SGD is too slow
- Problems with many local minima

---

### 3. Adam (Adaptive Moment Estimation)

**The most popular optimizer**

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

**How it works:**
- Adaptive learning rate per parameter
- Tracks both momentum and squared gradients
- Combines benefits of momentum and RMSprop

**Pros:**
- Fast convergence
- Less sensitive to learning rate
- Works well for most problems
- Default choice for deep learning

**Cons:**
- Can sometimes generalize worse than SGD
- More memory usage

**Best for:**
- Most deep learning problems (default)
- CNNs, Transformers
- When you want fast results

---

### 4. RMSprop

**Adaptive learning rates**

```python
optimizer = optim.RMSprop(model.parameters(), lr=0.001)
```

**How it works:**
- Adapts learning rate based on gradient magnitude
- Divides learning rate by moving average of squared gradients

**Pros:**
- Good for non-stationary problems
- Works well with RNNs

**Cons:**
- Less popular than Adam
- Can be sensitive to hyperparameters

**Best for:**
- RNNs
- When Adam doesn't work well
- Non-stationary problems

---

## 📊 Comparing Optimizers

The step includes a comprehensive comparison showing:
- Learning curves (loss over time)
- Final performance (loss and accuracy)
- Convergence speed
- Visual comparisons

**Typical Results:**
- **Adam**: Fastest convergence, good accuracy
- **SGD**: Slower but sometimes better generalization
- **RMSprop**: Good for specific problems
- **SGD-Momentum**: Faster than SGD, slower than Adam

---

## 🎛️ Learning Rate Scheduling

Sometimes you want to **change the learning rate during training**.

### Why?

- Start with higher LR for fast initial learning
- Reduce LR later for fine-tuning
- Helps escape local minima early
- Better final convergence

### Types of Schedulers

**1. StepLR** - Reduce LR at fixed intervals
```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
# Reduces LR by 50% every 200 epochs
```

**2. ExponentialLR** - Exponential decay
```python
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
# Multiplies LR by 0.95 each epoch
```

**3. CosineAnnealingLR** - Cosine schedule
```python
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# Smooth cosine curve
```

---

## 🎯 When to Use Which Optimizer?

### Decision Guide

**Start with Adam** (lr=0.001)
- Default choice for most problems
- Fast convergence
- Less tuning needed

**Use SGD** when:
- You need reproducible results
- Training from scratch on large datasets
- You want better generalization
- Problem is simple

**Use RMSprop** when:
- Working with RNNs
- Adam doesn't work well
- Non-stationary problems

**Use SGD-Momentum** when:
- SGD is too slow
- You want faster convergence than SGD
- But more control than Adam

---

## ⚠️ Common Pitfalls

### 1. Wrong Learning Rate

**Problem:**
- Too high: Training unstable, loss explodes
- Too low: Training too slow, may not converge

**Solution:**
- SGD: Usually 0.01 - 0.1
- Adam: Usually 0.0001 - 0.001
- Start with defaults, then tune

### 2. Forgetting to Zero Gradients

**Problem:**
```python
# WRONG - gradients accumulate!
loss.backward()
optimizer.step()
```

**Solution:**
```python
# CORRECT - always zero gradients first
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 3. Not Using Learning Rate Scheduling

**Problem:**
- Fixed LR may not be optimal
- Can get stuck in local minima

**Solution:**
- Use schedulers for long training
- Helps fine-tune convergence

### 4. Using Same Optimizer for Everything

**Problem:**
- Different problems need different optimizers
- One size doesn't fit all

**Solution:**
- Experiment with different optimizers
- Compare results
- Choose based on your specific problem

---

## 💻 Code Example

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Create model
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

# Choose optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Optional: Add learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

# Training loop
loss_fn = nn.BCELoss()
for epoch in range(1000):
    # Forward pass
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    
    # Backward pass
    optimizer.zero_grad()  # Always zero first!
    loss.backward()
    optimizer.step()
    
    # Update learning rate (if using scheduler)
    scheduler.step()
```

---

## 📈 Visualizations

The step includes:
1. **Learning curves comparison** - See how each optimizer converges
2. **Final performance** - Compare loss and accuracy
3. **Early training** - See convergence speed
4. **Accuracy comparison** - Bar chart of final accuracies
5. **Learning rate schedule** - Visualize LR changes over time

---

## ✅ Key Takeaways

1. **Adam is usually the best default** - Fast convergence, less tuning
2. **SGD gives more control** - But needs careful tuning
3. **Learning rate scheduling helps** - Especially for long training
4. **Experiment to find what works** - Different problems need different optimizers

---

## 🚀 Next Steps

After this step, you can:
- Choose the right optimizer for your problem
- Use learning rate scheduling
- Understand why training behaves the way it does
- Tune hyperparameters more effectively

**Continue to**: Step 7 (RNNs) or try the projects!

---

## 📚 Additional Resources

- [PyTorch Optimizers Documentation](https://pytorch.org/docs/stable/optim.html)
- [Understanding Adam Optimizer](https://ruder.io/optimizing-gradient-descent/)
- [An Overview of Gradient Descent Optimization Algorithms](https://ruder.io/optimizing-gradient-descent/index.html)

---

**Happy Optimizing!** 🎯
