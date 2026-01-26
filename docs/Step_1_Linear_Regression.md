# Step 1 — Linear Regression (Learning to Predict)

> **Goal:** Teach how AI learns from mistakes by adjusting weights automatically.  
> **Tools:** Python + NumPy + Matplotlib

---

## 1.1 Big Idea
Linear Regression answers one question:

> **How can we predict a number as accurately as possible?**

Example:
- Study hours → Exam score
- Experience → Salary
- Time → Distance

The AI tries to draw the **best possible line**.

---

## 1.2 The Model (Math First, Simple)

The linear model:

y = w·x + b

- **x** = input (study hours)
- **w** = weight (importance)
- **b** = bias (starting value)
- **y** = predicted output

---

## 1.3 Dataset Example

| Study Hours | Score |
|------------|-------|
| 1 | 50 |
| 2 | 60 |
| 3 | 70 |
| 4 | 80 |

```python
import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4], dtype=float)
y = np.array([50, 60, 70, 80], dtype=float)
```

---

## 1.4 Visualize the Data

```python
plt.scatter(X, y)
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Score")
plt.grid(True, alpha=0.3)
plt.show()
```

🧠 **Observation:** More study → higher score.

---

## 1.5 First Bad Guess (No Learning Yet)

```python
w = 0.0
b = 0.0

y_pred = w * X + b
print(y_pred)
```

```python
plt.scatter(X, y, label="Real Data")
plt.plot(X, y_pred, label="Bad Prediction", color="red")
plt.legend()
plt.show()
```

❌ The line is wrong — AI must learn.

---

## 1.6 Error (How Wrong Are We?)

We use **Mean Squared Error (MSE)**:

MSE = average((prediction − real)²)

```python
error = np.mean((y_pred - y) ** 2)
print("Error:", error)
```

🧠 **Key idea:** Learning = reducing error.

---

## 1.7 Gradient Descent (How AI Learns)

Think of:
- Error as a **hill**
- AI wants the **lowest point**
- Gradient = direction to move

---

## 1.8 Training the Model

```python
w = 0.0
b = 0.0
lr = 0.01

errors = []

for epoch in range(1000):
    y_pred = w * X + b

    dw = np.mean((y_pred - y) * X)
    db = np.mean(y_pred - y)

    w -= lr * dw
    b -= lr * db

    error = np.mean((y_pred - y) ** 2)
    errors.append(error)

print("Final w:", w)
print("Final b:", b)
```

---

## 1.9 Learning Curve (Very Important Graph)

```python
plt.plot(errors)
plt.xlabel("Epoch")
plt.ylabel("Error (MSE)")
plt.title("Learning Curve")
plt.grid(True, alpha=0.3)
plt.show()
```

✅ Error goes down → AI is learning.

---

## 1.10 Final Prediction Line

```python
y_pred = w * X + b

plt.scatter(X, y, label="Real Data")
plt.plot(X, y_pred, label="Learned Line", color="green")
plt.legend()
plt.show()
```

🎉 The AI found the best line.

---

## 1.11 Make Predictions

```python
study_hours = 5
predicted_score = w * study_hours + b
print("Predicted score for 5 hours:", predicted_score)
```

---

## 1.12 Mini Exercises

### Exercise 1
Change the learning rate:
- lr = 0.1
- lr = 0.001

What happens?

### Exercise 2
Add more data points.

### Exercise 3 (Challenge)
Predict scores for:
- 6 hours
- 8 hours

---

## 1.13 Checklist (Before Moving On)

Students should understand:
- What linear regression does
- What error means
- How gradient descent works
- Why the line improves

If YES → move to **Step 2: Perceptron**

---

## Next Step Preview
Now we can predict numbers.

Next, we will:
> Turn predictions into **decisions (YES / NO)**

➡️ **Step 2 – Perceptron**
