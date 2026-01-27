# Step 6c: Hyperparameter Tuning

> **Learn systematic methods to find optimal hyperparameters for your models**

**Time**: ~75 minutes  
**Prerequisites**: Step 6 (PyTorch basics), Step 6b (Optimization Techniques)

---

## 🎯 Learning Objectives

By the end of this step, you'll understand:
- What hyperparameters are and why they matter
- Grid search methodology
- Random search methodology
- Learning rate tuning
- Hyperparameter importance analysis
- Bayesian optimization (conceptual)
- Best practices for systematic tuning

---

## 📚 What are Hyperparameters?

**Hyperparameters** are settings that control how the model learns, set **before** training begins.

### Common Hyperparameters

- **Learning rate**: How fast the model learns
- **Number of layers**: Model depth
- **Number of neurons**: Model width
- **Batch size**: Number of samples per update
- **Optimizer type**: SGD, Adam, RMSprop, etc.
- **Regularization strength (λ)**: L2 penalty
- **Dropout rate**: Fraction of neurons to disable
- **Activation functions**: ReLU, sigmoid, tanh, etc.

### Hyperparameters vs Parameters

| Hyperparameters | Parameters (Weights) |
|----------------|---------------------|
| Set before training | Learned during training |
| Don't change during training | Updated via backpropagation |
| Control learning process | Store learned knowledge |
| Examples: LR, layers, dropout | Examples: W, b (weights, biases) |

---

## 🔍 Why Hyperparameter Tuning Matters

### Impact of Good Hyperparameters

✅ **Significantly improve accuracy**  
✅ **Reduce training time**  
✅ **Prevent overfitting**  
✅ **Make training more stable**

### Impact of Bad Hyperparameters

❌ **Training may fail** (e.g., LR too high)  
❌ **Poor performance** (e.g., LR too low)  
❌ **Waste computational resources**  
❌ **Model doesn't converge**

---

## 🔧 Hyperparameter Tuning Methods

### 1. Manual Tuning

**What**: Try different values manually

**Pros**:
- Simple
- Good for learning
- Quick for small changes

**Cons**:
- Time-consuming
- May miss optimal values
- Not systematic

**When to use**: Initial exploration, small adjustments

---

### 2. Grid Search

**What**: Try all combinations of hyperparameters

**How it works**:
1. Define ranges for each hyperparameter
2. Create grid of all combinations
3. Train model for each combination
4. Select best result

**Example**:
```python
learning_rates = [0.001, 0.01, 0.1]
hidden_sizes = [8, 16, 32]
dropout_rates = [0.0, 0.5]

# Tests 3 × 3 × 2 = 18 combinations
```

**Pros**:
- Systematic and exhaustive
- Guaranteed to find best in grid
- Easy to understand

**Cons**:
- Can be very slow (exponential growth)
- May miss values between grid points
- Wastes time on poor combinations

**When to use**: Small search space (< 100 combinations)

**Visualization**: Heatmaps showing performance across combinations

---

### 3. Random Search

**What**: Try random combinations of hyperparameters

**How it works**:
1. Define ranges for each hyperparameter
2. Randomly sample combinations
3. Train model for each sample
4. Select best result

**Example**:
```python
# Sample 20 random combinations instead of testing all
for trial in range(20):
    lr = random.uniform(0.0001, 0.1)
    hidden_size = random.choice([8, 16, 32, 64])
    dropout = random.uniform(0.0, 0.7)
    # Train and evaluate...
```

**Pros**:
- Faster than grid search
- Often finds good solutions quickly
- Better for large search spaces
- Can explore more of the space

**Cons**:
- May miss optimal values
- Less systematic
- Results vary between runs

**When to use**: Large search space, limited time

**Research**: Random search often outperforms grid search with same budget!

---

### 4. Bayesian Optimization

**What**: Smart search using probability models

**How it works**:
1. Build probabilistic model of objective function
2. Use model to suggest promising hyperparameters
3. Update model with new results
4. Repeat until convergence

**Pros**:
- More efficient than random search
- Fewer evaluations needed
- Good for expensive evaluations
- Learns from past results

**Cons**:
- More complex to implement
- Requires specialized libraries
- May get stuck in local optima

**When to use**: Expensive evaluations, limited budget

**Tools**:
- **scikit-optimize (skopt)**: Simple Bayesian optimization
- **Optuna**: Advanced hyperparameter optimization
- **Hyperopt**: Distributed hyperparameter optimization

---

## 🎯 Learning Rate Tuning

**Learning rate is often the most important hyperparameter!**

### Why Learning Rate Matters

- **Too high**: Training unstable, loss explodes
- **Too low**: Training too slow, may not converge
- **Just right**: Fast convergence, stable training

### Finding Optimal Learning Rate

**Method 1: Logarithmic Search**
```python
lr_candidates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
# Test each and find best
```

**Method 2: Learning Rate Range Test**
- Start with very small LR
- Gradually increase
- Plot loss vs LR
- Find point where loss starts increasing

**Method 3: Cyclical Learning Rates**
- Vary LR during training
- Helps escape local minima

### Typical Learning Rates

- **SGD**: 0.01 - 0.1
- **Adam**: 0.0001 - 0.001
- **RMSprop**: 0.001 - 0.01

---

## 📊 Hyperparameter Importance

**Not all hyperparameters matter equally!**

### How to Measure Importance

1. **Fix other hyperparameters**
2. **Vary one hyperparameter**
3. **Measure variance in performance**
4. **Higher variance = more important**

### Typical Importance Order

1. **Learning rate** (usually most important)
2. **Architecture** (layers, neurons)
3. **Regularization** (dropout, L2)
4. **Batch size**
5. **Optimizer type** (often less important)

### Why This Matters

- **Focus tuning effort** on important hyperparameters
- **Save time** by tuning less important ones last
- **Understand** which settings really matter

---

## ✅ Best Practices

### 1. Start with Defaults

- Use recommended values from literature
- PyTorch defaults are often good starting points
- Don't reinvent the wheel

### 2. Tune One at a Time (Initially)

**Order of tuning**:
1. **Learning rate** (most important)
2. **Architecture** (layers, neurons)
3. **Regularization** (dropout, L2)
4. **Other hyperparameters**

**Then**: Fine-tune combinations

### 3. Use Validation Set

**Critical**: Never tune on test set!

- **Training set**: Train model
- **Validation set**: Tune hyperparameters
- **Test set**: Final evaluation only

### 4. Start Coarse, Then Fine

**Phase 1: Coarse Search**
- Wide ranges (e.g., LR: 0.001 to 0.1)
- Fewer epochs per trial
- Find promising region

**Phase 2: Fine Search**
- Narrow ranges around best (e.g., LR: 0.01 to 0.03)
- More epochs per trial
- Refine optimal value

### 5. Use Appropriate Search Method

- **Grid search**: Small search space (< 100 combinations)
- **Random search**: Large search space
- **Bayesian**: Expensive evaluations

### 6. Track Experiments

**Log everything**:
- All hyperparameters tried
- Results (accuracy, loss)
- Training time
- Best configuration

**Tools**:
- **MLflow**: Experiment tracking
- **Weights & Biases**: Advanced tracking
- **TensorBoard**: Visualization
- **Simple CSV/JSON**: Manual logging

### 7. Consider Computational Budget

- **Time**: How long can you wait?
- **Resources**: GPU hours available?
- **Priority**: What matters most?

**Trade-offs**:
- More trials = better results but more time
- Fewer trials = faster but may miss optimal

---

## 💻 Code Examples

### Grid Search

```python
from itertools import product

def grid_search(lr_values, hidden_sizes, dropout_rates):
    best_acc = 0
    best_params = None
    
    for lr, hidden_size, dropout_rate in product(lr_values, hidden_sizes, dropout_rates):
        acc = train_and_evaluate(lr, hidden_size, dropout_rate)
        if acc > best_acc:
            best_acc = acc
            best_params = {'lr': lr, 'hidden_size': hidden_size, 'dropout_rate': dropout_rate}
    
    return best_params, best_acc
```

### Random Search

```python
import random

def random_search(n_trials=50):
    best_acc = 0
    best_params = None
    
    for _ in range(n_trials):
        lr = random.uniform(0.0001, 0.1)
        hidden_size = random.choice([8, 16, 32, 64])
        dropout_rate = random.uniform(0.0, 0.7)
        
        acc = train_and_evaluate(lr, hidden_size, dropout_rate)
        if acc > best_acc:
            best_acc = acc
            best_params = {'lr': lr, 'hidden_size': hidden_size, 'dropout_rate': dropout_rate}
    
    return best_params, best_acc
```

### Learning Rate Tuning

```python
def find_optimal_lr(lr_candidates):
    best_lr = None
    best_acc = 0
    
    for lr in lr_candidates:
        acc = train_and_evaluate(lr=lr)
        if acc > best_acc:
            best_acc = acc
            best_lr = lr
    
    return best_lr, best_acc
```

---

## 📈 Visualizations

The step includes:
1. **Grid Search Heatmap** - Performance across combinations
2. **Top Results Bar Chart** - Best configurations
3. **Random Search Scatter** - LR vs Accuracy
4. **Convergence Plot** - Best accuracy over trials
5. **Learning Rate Tuning** - Accuracy vs LR
6. **Hyperparameter Importance** - Variance analysis

---

## ✅ Key Takeaways

1. **Learning rate is often most important** - Start here
2. **Random search often beats grid search** - More efficient
3. **Start with defaults, tune systematically** - Don't guess
4. **Use validation set, not test set** - Critical!
5. **Track all experiments** - Learn from history
6. **Consider computational budget** - Balance time vs quality

---

## 🚀 Next Steps

After this step, you can:
- Systematically tune hyperparameters
- Choose appropriate search method
- Find optimal learning rates
- Understand hyperparameter importance
- Use best practices for tuning

**Continue to**: Step 7 (RNNs) or apply tuning to previous steps!

---

## 📚 Additional Resources

- [Scikit-learn Grid Search](https://scikit-learn.org/stable/modules/grid_search.html)
- [Optuna Documentation](https://optuna.org/)
- [Hyperparameter Tuning Guide](https://www.jeremyjordan.me/hyperparameter-tuning/)
- [Random Search for Hyperparameter Optimization](https://www.jmlr.org/papers/v13/bergstra12a.html)

---

## 🎓 Summary

**Hyperparameter tuning is essential** for getting the best performance from your models. The main methods are:

1. **Grid Search** - Exhaustive but slow
2. **Random Search** - Efficient and often better
3. **Bayesian Optimization** - Smart and efficient

**Best practice**: Start with learning rate, use random search for large spaces, track everything!

---

**Happy Tuning!** 🎯
