# Step 6 — PyTorch (From Scratch to Real AI)

> **Goal:** Move from building neural networks manually to using a **professional AI framework** (PyTorch) while understanding what it automates.  
> **Tools:** Python + PyTorch + Matplotlib

---

## 6.1 Big Idea
So far, **you built everything by hand**:
- Weights
- Forward pass
- Backpropagation
- Updates

Real AI engineers use frameworks like **PyTorch** to:
- Write less code
- Avoid bugs
- Train large models faster

🧠 **Important:**  
PyTorch does NOT replace understanding — it **automates math you already know**.

---

## 6.2 Installing PyTorch

Visit: https://pytorch.org  
Choose:
- OS: your system
- Package: pip
- Language: Python

Example:
```bash
pip install torch torchvision torchaudio
```

---

## 6.3 First PyTorch Tensor

A **tensor** is like a NumPy array, but smarter.

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
print(x)
```

### Tensor vs NumPy
| NumPy | PyTorch |
|-----|--------|
| array | tensor |
| CPU only | CPU / GPU |
| manual gradients | automatic gradients |

---

## 6.4 Autograd (Automatic Gradients)

PyTorch can calculate gradients **automatically**.

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x

y.backward()

print("dy/dx =", x.grad)
```

🧠 **Key idea:**  
This replaces manual derivative calculations.

---

## 6.5 Dataset Example (XOR Again)

We reuse XOR to prove PyTorch works.

```python
X = torch.tensor([[0.,0.],
                  [0.,1.],
                  [1.,0.],
                  [1.,1.]])

y = torch.tensor([[0.],
                  [1.],
                  [1.],
                  [0.]])
```

---

## 6.6 Defining a Neural Network

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 4),   # input → hidden
    nn.ReLU(),         # activation
    nn.Linear(4, 1),   # hidden → output
    nn.Sigmoid()       # probability
)

print(model)
```

🧠 **Connection to previous steps:**
- Linear = weights + bias
- ReLU/Sigmoid = activation
- Layers = matrices

---

## 6.7 Loss Function & Optimizer

```python
loss_fn = nn.BCELoss()              # Binary Cross-Entropy
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)
```

| Concept | PyTorch |
|------|--------|
| Loss | `nn.BCELoss()` |
| Gradient Descent | `optim.SGD` |
| Weights | `model.parameters()` |

---

## 6.8 Training Loop (Very Important)

```python
losses = []

for epoch in range(3000):
    # Forward
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    losses.append(loss.item())

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

🧠 This loop replaces **everything you coded manually** before.

---

## 6.9 Learning Curve

```python
import matplotlib.pyplot as plt

plt.plot(losses)
plt.title("Training Loss (PyTorch)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.show()
```

✅ Loss decreases → model is learning.

---

## 6.10 Final Predictions

```python
with torch.no_grad():
    probs = model(X)
    preds = (probs >= 0.5).int()

print("Probabilities:\n", probs)
print("Predictions:\n", preds)
```

🎉 PyTorch solved XOR successfully.

---

## 6.11 Model Saving and Loading

### Why Save Models?

Saving models allows you to:
- **Reuse trained models** without retraining
- **Share models** with others
- **Deploy models** to production
- **Resume training** from checkpoints

### Saving a Model

```python
# Save the model
model_path = "xor_model.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': losses[-1],
}, model_path)

print(f"✅ Model saved to {model_path}")
print(f"   - Model weights: {model_path}")
print(f"   - File size: ~{sum(p.numel() * 4 for p in model.parameters()) / 1024:.1f} KB")
```

**Code Explanation:**
- `torch.save()`: Saves data to file
- `model.state_dict()`: Dictionary containing all model weights
- `optimizer.state_dict()`: Optimizer state (learning rate, momentum, etc.)
- `losses[-1]`: Final training loss
- `.pth` extension: PyTorch checkpoint file

**What gets saved:**
- Model architecture (weights and biases)
- Optimizer state
- Training loss
- Any other metadata you include

### Loading a Model

```python
# Create a new model instance (untrained)
new_model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

# Test new model (should be random)
with torch.no_grad():
    new_probs = new_model(X)
    new_preds = (new_probs >= 0.5).int()
    new_accuracy = (new_preds == y.int()).float().mean().item()

print(f"New model accuracy (before loading): {new_accuracy:.2%}")
print("  This is random because the model hasn't been trained yet")

# Load the saved model
print(f"Loading model from {model_path}...")
checkpoint = torch.load(model_path)
new_model.load_state_dict(checkpoint['model_state_dict'])
print("✅ Model loaded successfully!")

# Test loaded model (should match original)
with torch.no_grad():
    loaded_probs = new_model(X)
    loaded_preds = (loaded_probs >= 0.5).int()
    loaded_accuracy = (loaded_preds == y.int()).float().mean().item()

print(f"Loaded model accuracy: {loaded_accuracy:.2%}")
print("  This matches the original trained model!")
```

**Code Explanation:**
- `new_model`: Create fresh model instance (same architecture)
- `torch.load()`: Load checkpoint from file
- `checkpoint['model_state_dict']`: Extract model weights
- `new_model.load_state_dict()`: Load weights into model
- `torch.no_grad()`: Disable gradient computation (faster inference)

**Expected Output:**
```
New model accuracy (before loading): 50.00%
  This is random because the model hasn't been trained yet
Loading model from xor_model.pth...
✅ Model loaded successfully!
Loaded model accuracy: 100.00%
  This matches the original trained model!
```

### Best Practices

1. **Save checkpoints during training**:
   ```python
   if epoch % 100 == 0:
       torch.save({
           'epoch': epoch,
           'model_state_dict': model.state_dict(),
           'optimizer_state_dict': optimizer.state_dict(),
           'loss': loss,
       }, f'checkpoint_epoch_{epoch}.pth')
   ```

2. **Save best model**:
   ```python
   best_loss = float('inf')
   for epoch in range(num_epochs):
       # ... training ...
       if loss < best_loss:
           best_loss = loss
           torch.save(model.state_dict(), 'best_model.pth')
   ```

3. **Save only weights** (smaller file):
   ```python
   torch.save(model.state_dict(), 'model_weights.pth')
   ```

4. **Load with error handling**:
   ```python
   try:
       checkpoint = torch.load('model.pth')
       model.load_state_dict(checkpoint['model_state_dict'])
   except FileNotFoundError:
       print("Model file not found!")
   ```

### File Formats

- **`.pth`**: PyTorch checkpoint (recommended)
- **`.pt`**: Alternative PyTorch extension
- **`.pkl`**: Pickle format (less common)

---

## 6.12 Comparing Scratch vs PyTorch

| From Scratch | PyTorch |
|------------|---------|
| Manual gradients | Automatic |
| More code | Less code |
| Easy to make mistakes | Safer |
| Best for learning | Best for real projects |

🧠 **Rule:**  
Learn from scratch → build with PyTorch.

---

## 6.13 Mini Projects (Choose One)

### Project 1
Pass / Fail predictor (study hours)

### Project 2
Grade classifier (A/B/C)

### Project 3
Student performance predictor (multi-feature)

---

## 6.14 Common Beginner Mistakes

❌ Using PyTorch without understanding math  
❌ Forgetting `optimizer.zero_grad()`  
❌ Mixing NumPy and Torch tensors  
❌ Ignoring loss curves  

---

## 6.15 Final Checklist (Bootcamp Complete 🎓)

Students can:
- Explain neurons mathematically
- Train models from scratch
- Use PyTorch correctly
- Read AI code confidently
- Build simple AI projects

---

## 🎉 Congratulations!
You have completed:
- Step 0 → Math Foundations
- Step 1 → Linear Regression
- Step 2 → Perceptron
- Step 3 → Logistic Regression
- Step 4 → Neural Network Layers
- Step 5 → XOR & Hidden Layers
- Step 6 → PyTorch

🚀 You are now a **Junior AI Engineer**.

---

## What’s Next?
Possible next tracks:
- CNNs (images)
- RNNs (text, sequences)
- Real datasets (CSV, images)
- AI projects & competitions

**End of Bootcamp** 🧠🤖
