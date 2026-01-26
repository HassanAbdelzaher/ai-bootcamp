# Step 2 — Perceptron (First Decision-Making AI)

> **Goal:** Teach how AI makes YES / NO decisions using a simple artificial neuron.  
> **Tools:** Python + NumPy + Matplotlib

---

## 2.1 Big Idea
Linear Regression predicts **numbers**.

Now we ask a new question:

> **Should we decide YES or NO?**

Examples:
- Pass or Fail?
- Spam or Not Spam?
- Buy or Not Buy?

This is where the **Perceptron** appears.

---

## 2.2 The Perceptron Model

The perceptron equation:

z = x · w + b

Decision rule:

- If z ≥ 0 → output = 1 (YES)
- If z < 0 → output = 0 (NO)

🧠 Same math as before, but with a **decision step**.

---

## 2.3 Dataset Example (Pass / Fail)

| Study Hours | Pass |
|------------|------|
| 1 | 0 |
| 2 | 0 |
| 3 | 1 |
| 4 | 1 |

```python
import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4])
y = np.array([0, 0, 1, 1])
```

---

## 2.4 Step Function (Decision Maker)

```python
def step_function(z):
    return 1 if z >= 0 else 0
```

This function turns a **number** into a **decision**.

---

## 2.5 A Single Perceptron (No Learning Yet)

```python
w = 1.0
b = -2.5

predictions = []

for x in X:
    z = w * x + b
    predictions.append(step_function(z))

print("Predictions:", predictions)
```

---

## 2.6 Visualizing the Decision Boundary

The decision boundary is where:

w·x + b = 0

```python
x_vals = np.linspace(0, 5, 100)
boundary = (-b / w)

plt.scatter(X, y, label="Data")
plt.axvline(boundary, linestyle="--", color="red", label="Decision Boundary")
plt.xlabel("Study Hours")
plt.ylabel("Pass (0/1)")
plt.title("Perceptron Decision Boundary")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

🧠 **Key idea:** The perceptron splits data using a straight line.

---

## 2.7 Training the Perceptron (Learning Rule)

Perceptron update rule:

w = w + learning_rate × (y − y_pred) × x  
b = b + learning_rate × (y − y_pred)

---

## 2.8 Training Loop

```python
w = 0.0
b = 0.0
lr = 0.1

for epoch in range(10):
    for i in range(len(X)):
        z = w * X[i] + b
        y_pred = step_function(z)

        error = y[i] - y_pred

        w += lr * error * X[i]
        b += lr * error

    print(f"Epoch {epoch}: w={w:.2f}, b={b:.2f}")
```

---

## 2.9 Final Predictions

```python
final_preds = []

for x in X:
    z = w * x + b
    final_preds.append(step_function(z))

print("Final predictions:", final_preds)
```

---

## 2.10 Visualize Final Decision Boundary

```python
boundary = (-b / w)

plt.scatter(X, y, label="Data")
plt.axvline(boundary, linestyle="--", color="green", label="Learned Boundary")
plt.xlabel("Study Hours")
plt.ylabel("Pass (0/1)")
plt.title("Learned Perceptron Boundary")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

🎉 The perceptron learned a decision rule.

---

## 2.11 Limitations of the Perceptron

❌ Can only draw **straight lines**  
❌ Cannot solve problems like XOR  
❌ Hard YES/NO decisions only

This leads to the next upgrade.

---

## 2.12 Mini Exercises

### Exercise 1
Change the learning rate:
- lr = 0.01
- lr = 1.0

What happens?

### Exercise 2
Change the bias start value.

### Exercise 3 (Thinking)
Why can’t one straight line solve XOR?

---

## 2.13 Checklist (Before Moving On)

Students should understand:
- What a perceptron is
- How it makes decisions
- How weights and bias change
- What its limitations are

If YES → move to **Step 3: Logistic Regression**

---

## Next Step Preview
The perceptron makes **hard decisions**.

Next, we will:
> Make decisions with **confidence (probabilities)**

➡️ **Step 3 – Logistic Regression**
