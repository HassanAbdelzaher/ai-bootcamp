# Step 3 — Logistic Regression (Smart Decisions with Probability)

> **Goal:** Upgrade the perceptron from hard YES/NO decisions to **probabilistic decisions**.  
> **Tools:** Python + NumPy + Matplotlib

---

## 3.1 Big Idea
The perceptron says:
- YES (1) or NO (0)

But real AI should say:
> **"I am 85% confident"**

That’s exactly what **Logistic Regression** does.

---

## 3.2 From Perceptron to Logistic Regression

Same core equation:
z = x · w + b

But instead of a step function, we use **Sigmoid**.

---

## 3.3 Sigmoid Function (Probability Maker)

The sigmoid function:

sigmoid(z) = 1 / (1 + e⁻ᶻ)

It converts any number into a value between **0 and 1**.

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

### Visualize Sigmoid

```python
z_vals = np.linspace(-10, 10, 200)
s_vals = sigmoid(z_vals)

plt.plot(z_vals, s_vals)
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("Probability")
plt.grid(True, alpha=0.3)
plt.show()
```

🧠 **Key idea:** Output is now a **probability**, not a decision.

---

## 3.4 Dataset Example (Pass Probability)

| Study Hours | Pass |
|------------|------|
| 1 | 0 |
| 2 | 0 |
| 3 | 1 |
| 4 | 1 |

```python
X = np.array([1, 2, 3, 4], dtype=float)
y = np.array([0, 0, 1, 1], dtype=float)
```

---

## 3.5 Forward Pass (Prediction)

```python
w = 0.0
b = 0.0

z = w * X + b
y_pred = sigmoid(z)

print("Probabilities:", y_pred)
```

---

## 3.6 Loss Function (How Wrong Are We?)

We use **Binary Cross-Entropy Loss**.

Intuition:
- Confident & correct → small loss
- Confident & wrong → BIG loss

```python
def binary_cross_entropy(y, y_pred):
    return -np.mean(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))

loss = binary_cross_entropy(y, y_pred)
print("Loss:", loss)
```

---

## 3.7 Gradient Descent (Learning Probabilities)

```python
lr = 0.1
w = 0.0
b = 0.0
losses = []

for epoch in range(1000):
    z = w * X + b
    y_pred = sigmoid(z)

    dw = np.mean((y_pred - y) * X)
    db = np.mean(y_pred - y)

    w -= lr * dw
    b -= lr * db

    loss = binary_cross_entropy(y, y_pred)
    losses.append(loss)
```

---

## 3.8 Learning Curve

```python
plt.plot(losses)
plt.title("Loss Decreasing Over Time")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.grid(True, alpha=0.3)
plt.show()
```

✅ Loss goes down → AI is learning probabilities.

---

## 3.9 Final Probabilities

```python
print("Final weights:", w)
print("Final bias:", b)

final_probs = sigmoid(w * X + b)
print("Final probabilities:", final_probs)
```

---

## 3.10 From Probability to Decision

```python
threshold = 0.5
decisions = (final_probs >= threshold).astype(int)

print("Decisions:", decisions)
```

🧠 **Important:** Decision comes **after** probability.

---

## 3.11 Visualizing the Probability Curve

```python
x_vals = np.linspace(0, 5, 200)
probs = sigmoid(w * x_vals + b)

plt.plot(x_vals, probs)
plt.axhline(0.5, linestyle="--", label="Decision Threshold")
plt.xlabel("Study Hours")
plt.ylabel("Pass Probability")
plt.title("Logistic Regression Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 3.12 Why Logistic Regression is Better Than Perceptron

✅ Smooth learning  
✅ Probability output  
✅ Stable training  
❌ Still only straight-line separation

---

## 3.13 Mini Exercises

### Exercise 1
Change the threshold to:
- 0.7
- 0.3

What changes?

### Exercise 2
Add more data points and retrain.

### Exercise 3 (Thinking)
Why is sigmoid better than step function for learning?

---

## 3.14 Checklist (Before Moving On)

Students should understand:
- Difference between decision and probability
- What sigmoid does
- Why loss is different
- How gradient descent updates weights

If YES → move to **Step 4: Multiple Neurons**

---

## Next Step Preview
Now we have **smart single neurons**.

Next, we will:
> Combine many neurons using **matrices**

➡️ **Step 4 – Neural Networks (Multiple Neurons)**
