# Step 0 — Math Foundations for AI (Teen-Friendly, Full Lesson)
> **Goal:** Build the math “language” behind AI so students can understand perceptrons, neural networks, and training later.  
> **Tools:** Python + NumPy + Matplotlib (graphs)

---

## 0.1 What AI Really Does (Big Idea)
AI does **not** think like humans.  
AI repeatedly does **math operations** to turn inputs into outputs.

A simple “AI brain” often looks like this:

z = x · w + b

- **x** = inputs (features)
- **w** = weights (importance)
- **b** = bias (a starting push)
- **z** = score (before making a decision)

---

## 0.2 Setup

```bash
pip install numpy matplotlib
```

```python
import numpy as np
import matplotlib.pyplot as plt
```

---

## 0.3 Numbers & Features

```python
study_hours = 4
math_score = 80
temperature = 28.5
pixel_brightness = 0.92

print(study_hours, math_score, temperature, pixel_brightness)
```

---

## 0.4 Vectors

```python
import numpy as np
x = np.array([80, 70, 75])
print("Student vector:", x)
```

---

## 0.5 Weights

```python
w = np.array([0.6, 0.3, 0.1])
print("Weights:", w)
```

---

## 0.6 Dot Product

```python
x = np.array([80, 70, 75])
w = np.array([0.6, 0.3, 0.1])
z = np.dot(x, w)
print("Dot product score:", z)
```

```python
features = np.array([80, 70, 75])
weights  = np.array([0.6, 0.3, 0.1])

contrib = features * weights

plt.bar(["Math", "Science", "English"], contrib)
plt.title("Feature Contributions")
plt.show()
```

---

## 0.7 Bias

```python
b = -10
z_with_bias = z + b
print("Score with bias:", z_with_bias)
```

---

## 0.8 Mini Neuron

```python
def neuron(x, w, b):
    return np.dot(x, w) + b

print(neuron(x, w, b))
```

---

## 0.9 Decision Boundary

```python
threshold = 60
decision = 1 if z_with_bias >= threshold else 0
print("Decision:", decision)
```

---

## 0.10 Multiple Students

```python
X = np.array([
    [90, 85, 70],
    [40, 50, 60],
    [75, 70, 80],
    [55, 60, 58],
])

scores = X @ w + b
print(scores)
```

---

## 0.11 Exercises
- Change weights and observe impact
- Change bias values
- Implement pass/fail function

---

## Ready for Step 1
Next: **Linear Regression & Gradient Descent**
