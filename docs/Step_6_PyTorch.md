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

## 6.11 Comparing Scratch vs PyTorch

| From Scratch | PyTorch |
|------------|---------|
| Manual gradients | Automatic |
| More code | Less code |
| Easy to make mistakes | Safer |
| Best for learning | Best for real projects |

🧠 **Rule:**  
Learn from scratch → build with PyTorch.

---

## 6.12 Mini Projects (Choose One)

### Project 1
Pass / Fail predictor (study hours)

### Project 2
Grade classifier (A/B/C)

### Project 3
Student performance predictor (multi-feature)

---

## 6.13 Common Beginner Mistakes

❌ Using PyTorch without understanding math  
❌ Forgetting `optimizer.zero_grad()`  
❌ Mixing NumPy and Torch tensors  
❌ Ignoring loss curves  

---

## 6.14 Final Checklist (Bootcamp Complete 🎓)

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
