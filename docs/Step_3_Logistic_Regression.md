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

**Code Explanation:**
- `import numpy as np`: NumPy for numerical operations
- `import matplotlib.pyplot as plt`: For plotting visualizations
- `def sigmoid(z):`: Defines the sigmoid function
  - Takes a score `z` (can be any number, positive or negative)
  - Returns a probability (always between 0 and 1)
- `1 / (1 + np.exp(-z))`: The sigmoid formula
  - `np.exp(-z)`: Exponential function (e raised to -z power)
  - `1 + np.exp(-z)`: Add 1 to the exponential
  - `1 / ...`: Divide 1 by the result
  - **Why this works:**
    - When z is large positive: exp(-z) ≈ 0, so result ≈ 1
    - When z is large negative: exp(-z) ≈ ∞, so result ≈ 0
    - When z = 0: exp(0) = 1, so result = 0.5
  - Output is always between 0 and 1 (perfect for probabilities!)

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

**Code Explanation:**
- `def binary_cross_entropy(y, y_pred):`: Defines loss function
  - `y`: Actual labels (0 or 1)
  - `y_pred`: Predicted probabilities (0 to 1)
- `y * np.log(y_pred + 1e-9)`: Loss when actual = 1
  - If actual = 1: This term contributes to loss
  - If actual = 0: This term is 0 (multiplied by 0)
  - `+ 1e-9`: Small number to avoid log(0) = -infinity
  - `np.log(...)`: Natural logarithm
- `(1 - y) * np.log(1 - y_pred + 1e-9)`: Loss when actual = 0
  - If actual = 0: This term contributes to loss
  - If actual = 1: This term is 0
- `+`: Add both terms together
- `np.mean(...)`: Average across all data points
- `-`: Negate (because we want to maximize log-likelihood, minimize negative)
- **Intuition:**
  - Confident & correct → small loss (log of number close to 1)
  - Confident & wrong → BIG loss (log of number close to 0)
  - Uncertain (0.5) → medium loss

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

**Code Explanation:**
- **Initialization:**
  - `lr = 0.1`: Learning rate (step size)
  - `w = 0.0` and `b = 0.0`: Start with zero weights
  - `losses = []`: Track loss over time

- **Training Loop:**
  - `for epoch in range(1000):`: Train for 1000 iterations

- **Forward Pass:**
  - `z = w * X + b`: Calculate scores (before sigmoid)
  - `y_pred = sigmoid(z)`: Convert scores to probabilities
    - Now outputs are between 0 and 1 (not just 0 or 1!)

- **Calculate Gradients:**
  - `dw = np.mean((y_pred - y) * X)`: Gradient for weight
    - `(y_pred - y)`: Difference between probability and actual
    - `* X`: Multiply by input (chain rule)
    - Same formula as linear regression, but y_pred is now a probability!
  - `db = np.mean(y_pred - y)`: Gradient for bias
    - Average of probability differences

- **Update Weights:**
  - `w -= lr * dw`: Move weight in direction that reduces loss
  - `b -= lr * db`: Move bias in direction that reduces loss
  - Same gradient descent as before!

- **Track Loss:**
  - `loss = binary_cross_entropy(y, y_pred)`: Calculate current loss
  - `losses.append(loss)`: Save for visualization

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

## 3.12 ROC Curve (Receiver Operating Characteristic)

### What is ROC Curve?

The **ROC curve** shows model performance across different classification thresholds. It's a powerful tool for evaluating binary classifiers.

**Key components:**
- **X-axis**: False Positive Rate (1 - Specificity)
  - FPR = FP / (FP + TN)
  - How often we incorrectly predict positive
- **Y-axis**: True Positive Rate (Recall/Sensitivity)
  - TPR = TP / (TP + FN)
  - How often we correctly predict positive
- **AUC (Area Under Curve)**: Single number summarizing performance
  - AUC = 1.0: Perfect classifier
  - AUC = 0.5: Random classifier (no better than guessing)
  - AUC > 0.7: Good classifier

### Calculating ROC Curve

```python
def calculate_roc_curve(y_true, y_scores):
    """Calculate ROC curve points"""
    # Sort by scores (descending)
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]
    
    # Calculate TPR and FPR for each threshold
    thresholds = np.unique(y_scores_sorted)
    thresholds = np.append(thresholds, [1.0, 0.0])  # Add endpoints
    thresholds = np.sort(thresholds)[::-1]
    
    tpr = []
    fpr = []
    
    for threshold in thresholds:
        y_pred = (y_scores_sorted >= threshold).astype(int)
        
        tp = np.sum((y_pred == 1) & (y_true_sorted == 1))
        fp = np.sum((y_pred == 1) & (y_true_sorted == 0))
        tn = np.sum((y_pred == 0) & (y_true_sorted == 0))
        fn = np.sum((y_pred == 0) & (y_true_sorted == 1))
        
        tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr.append(tpr_val)
        fpr.append(fpr_val)
    
    # Calculate AUC using trapezoidal rule
    auc = np.trapz(tpr, fpr)
    
    return np.array(fpr), np.array(tpr), auc
```

**Code Explanation:**
- `y_true`: Actual labels (0 or 1)
- `y_scores`: Predicted probabilities (0 to 1)
- `np.argsort(y_scores)[::-1]`: Sort indices by score (highest first)
- For each threshold:
  - Convert probabilities to predictions (≥ threshold = 1)
  - Calculate True Positives (TP), False Positives (FP), etc.
  - Compute TPR and FPR
- `np.trapz(tpr, fpr)`: Calculate area under curve using trapezoidal rule

### Using ROC Curve

```python
from plotting import plot_roc_curve

# Get probabilities for all data points
final_probs_all = sigmoid(w * X + b)
fpr, tpr, auc_score = calculate_roc_curve(y, final_probs_all)

print(f"AUC Score: {auc_score:.3f}")
print("  AUC = 1.0: Perfect classifier")
print("  AUC = 0.5: Random classifier (no better than guessing)")
print("  AUC > 0.7: Good classifier")

# Plot ROC curve
plot_roc_curve(fpr, tpr, auc_score, title="ROC Curve for Logistic Regression Model")
```

**Code Explanation:**
- `final_probs_all`: All predicted probabilities
- `calculate_roc_curve()`: Computes FPR, TPR, and AUC
- `plot_roc_curve()`: Visualizes the ROC curve

**Expected Output:**
```
AUC Score: 1.000
  AUC = 1.0: Perfect classifier
  AUC = 0.5: Random classifier (no better than guessing)
  AUC > 0.7: Good classifier
```

### Interpreting ROC Curve

**What to look for:**
1. **Curve shape**: 
   - Closer to top-left corner = better model
   - Diagonal line = random classifier (AUC = 0.5)
2. **AUC value**:
   - 0.9-1.0: Excellent
   - 0.8-0.9: Good
   - 0.7-0.8: Fair
   - 0.5-0.7: Poor
   - 0.5: Random

**Example interpretations:**
- AUC = 0.95: Model correctly ranks 95% of positive examples higher than negative examples
- AUC = 0.80: Model correctly ranks 80% of positive examples higher than negative examples
- AUC = 0.50: Model is no better than random guessing

### When to Use ROC Curve

✅ **Use ROC curve when:**
- Balanced dataset (similar number of positive and negative examples)
- You care about both false positives and false negatives
- You want to compare multiple models

❌ **Don't use ROC curve when:**
- Highly imbalanced dataset (use Precision-Recall curve instead)
- You care more about one type of error (e.g., false positives)

---

## 3.13 Why Logistic Regression is Better Than Perceptron

✅ Smooth learning  
✅ Probability output  
✅ Stable training  
❌ Still only straight-line separation

---

## 3.14 Mini Exercises

### Exercise 1
Change the threshold to:
- 0.7
- 0.3

What changes? Observe how the ROC curve changes.

### Exercise 2
Add more data points and retrain. Compare AUC scores before and after.

### Exercise 3 (Thinking)
Why is sigmoid better than step function for learning?

### Exercise 4
What does an AUC of 0.8 mean in practical terms?
- How often does the model correctly rank positive examples?
- Is this a good classifier?

---

## 3.15 Checklist (Before Moving On)

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
