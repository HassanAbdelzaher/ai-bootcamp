# Step 5 — Hidden Layers & XOR (Why Deep Learning Exists)

> **Goal:** Prove why a single layer is not enough and how **hidden layers** solve complex problems.  
> **Tools:** Python + NumPy + Matplotlib

---

## 5.1 Big Idea (The WOW Moment)
Some problems **cannot** be solved with a single straight line.

This famous problem is called **XOR**.

> If perceptron fails ❌ and a deeper network succeeds ✅,  
> then depth really matters.

---

## 5.2 The XOR Problem

XOR truth table:

| x1 | x2 | XOR |
|----|----|-----|
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 0 |

```python
import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([[0], [1], [1], [0]])
```

---

## 5.3 Visualizing XOR (Why One Line Fails)

```python
plt.scatter(X[:,0], X[:,1], c=y.flatten(), s=100)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("XOR Problem")
plt.grid(True, alpha=0.3)
plt.show()
```

🧠 **Observation:**  
No single straight line can separate the blue and orange points.

---

## 5.4 Try a Single-Layer Model (It Will Fail)

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

W = np.random.randn(2, 1)
b = np.zeros((1,))

lr = 0.1
losses = []

for epoch in range(2000):
    z = X @ W + b
    y_pred = sigmoid(z)

    loss = -np.mean(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))
    losses.append(loss)

    dW = X.T @ (y_pred - y)
    db = np.mean(y_pred - y)

    W -= lr * dW
    b -= lr * db

print("Single-layer predictions:", (y_pred >= 0.5).astype(int))
```

❌ The model cannot solve XOR.

---

## 5.5 Adding a Hidden Layer (Deep Learning)

Now we add **one hidden layer** with non-linearity.

Architecture:
- Input layer (2 neurons)
- Hidden layer (4 neurons)
- Output layer (1 neuron)

---

## 5.6 Initialize the Network

```python
W1 = np.random.randn(2, 4)
b1 = np.zeros((1, 4))

W2 = np.random.randn(4, 1)
b2 = np.zeros((1, 1))

lr = 0.1
losses = []
```

**Code Explanation:**
- `W1 = np.random.randn(2, 4)`: First layer weights (input → hidden)
  - Shape `(2, 4)`: 2 input features → 4 hidden neurons
  - Random initialization (will be learned)
- `b1 = np.zeros((1, 4))`: First layer biases
  - Shape `(1, 4)`: One bias per hidden neuron (4 biases)
- `W2 = np.random.randn(4, 1)`: Second layer weights (hidden → output)
  - Shape `(4, 1)`: 4 hidden neurons → 1 output neuron
  - Combines hidden layer outputs
- `b2 = np.zeros((1, 1))`: Second layer bias
  - Shape `(1, 1)`: Single bias for output neuron
- `lr = 0.1`: Learning rate for gradient descent
- `losses = []`: Track loss during training
- **Architecture:** Input(2) → Hidden(4) → Output(1)

---

## 5.7 Forward Pass

```python
def forward(X):
    Z1 = X @ W1 + b1
    A1 = sigmoid(Z1)

    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)

    return A1, A2
```

---

## 5.8 Training with Backpropagation

```python
for epoch in range(5000):
    A1, y_pred = forward(X)

    loss = -np.mean(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))
    losses.append(loss)

    # Backprop
    dZ2 = y_pred - y
    dW2 = A1.T @ dZ2
    db2 = np.mean(dZ2, axis=0)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * A1 * (1 - A1)
    dW1 = X.T @ dZ1
    db1 = np.mean(dZ1, axis=0)

    # Update
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1
```

---

## 5.9 Learning Curve

```python
plt.plot(losses)
plt.title("Loss During XOR Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.show()
```

✅ Loss drops → learning happens.

---

## 5.10 Final Predictions (SUCCESS 🎉)

```python
_, final_probs = forward(X)
final_preds = (final_probs >= 0.5).astype(int)

print("Final probabilities:\n", final_probs)
print("Final predictions:\n", final_preds)
```

✅ The network **solves XOR**.

---

## 5.11 What Just Happened?

- Hidden neurons learned **intermediate patterns**
- Network combined them to solve XOR
- This is the foundation of **deep learning**

🧠 **Key insight:**  
Depth creates **new feature spaces**.

---

## 5.12 Visualizing Decision Regions (Optional)

```python
xx, yy = np.meshgrid(np.linspace(-0.5,1.5,200), np.linspace(-0.5,1.5,200))
grid = np.c_[xx.ravel(), yy.ravel()]

_, probs = forward(grid)
Z = probs.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=20, alpha=0.7)
plt.scatter(X[:,0], X[:,1], c=y.flatten(), s=100, edgecolors="k")
plt.title("XOR Decision Regions (Deep Network)")
plt.show()
```

---

## 5.13 Why Step 5 Is Critical

✅ Explains **why deep learning exists**  
✅ Shows limits of shallow models  
✅ Makes hidden layers intuitive  
✅ Creates a lasting “Aha!” moment

---

## 5.14 Mini Exercises

### Exercise 1
Change number of hidden neurons:
- Try 2 neurons
- Try 8 neurons

### Exercise 2
Replace sigmoid with ReLU in hidden layer.

### Exercise 3 (Thinking)
Why does adding depth help but adding width sometimes doesn’t?

---

## 5.15 Checklist (Before Moving On)

Students should understand:
- Why XOR fails for perceptron
- What hidden layers do
- Basic backprop idea
- Why this is called *deep* learning

If YES → move to **Step 6: PyTorch (Real AI Frameworks)**

---

## Next Step Preview
Now that we understand **deep learning from scratch**,  
we will use a professional tool that automates everything.

➡️ **Step 6 — PyTorch**
