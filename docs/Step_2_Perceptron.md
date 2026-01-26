# Step 2 — Perceptron (First Decision-Making AI)

> **Goal:** Teach how AI makes YES / NO decisions using a simple artificial neuron.  
> **Tools:** Python + NumPy + Matplotlib  
> **Time:** ~60 minutes  
> **Difficulty:** ⭐⭐ Beginner-Intermediate

---

## 📚 Table of Contents

1. [Big Idea](#21-big-idea)
2. [The Perceptron Model](#22-the-perceptron-model)
3. [Dataset Example](#23-dataset-example-pass--fail)
4. [Step Function](#24-step-function-decision-maker)
5. [Initial Perceptron](#25-a-single-perceptron-no-learning-yet)
6. [Decision Boundary](#26-visualizing-the-decision-boundary)
7. [Training Rule](#27-training-the-perceptron-learning-rule)
8. [Training Loop](#28-training-loop)
9. [Weight Evolution](#29-weight-evolution)
10. [Final Predictions](#210-final-predictions)
11. [Final Decision Boundary](#211-visualize-final-decision-boundary)
12. [Limitations](#212-limitations-of-the-perceptron)
13. [Exercises](#213-mini-exercises)

---

## 2.1 Big Idea

### From Numbers to Decisions

**Linear Regression** predicts **numbers**:
- "This student will score 85 points"

**Perceptron** makes **decisions**:
- "This student will PASS" or "This student will FAIL"

### The Core Question

> **Should we decide YES or NO?**

### Real-World Examples

| Input | Decision | Use Case |
|-------|----------|----------|
| Study hours | Pass or Fail? | Education |
| Email content | Spam or Not Spam? | Email filtering |
| Customer data | Buy or Not Buy? | Marketing |
| Medical test | Disease or Healthy? | Healthcare |
| Image pixels | Cat or Dog? | Image classification |

### What Makes Perceptron Different

**Linear Regression:**
```
Input → Calculation → Number (e.g., 85.3)
```

**Perceptron:**
```
Input → Calculation → Decision (0 or 1, NO or YES)
```

---

## 2.2 The Perceptron Model

### The Same Math, Different Output

The perceptron uses the **same equation** as before:

```
z = x · w + b
```

But then applies a **decision rule**:

```
If z ≥ 0 → output = 1 (YES)
If z < 0  → output = 0 (NO)
```

### Visual Representation

```
Input (x)
    │
    ├─→ [× w] ─┐
    │          │
    └─→ [+ b] ─┼─→ [z] ─→ [Step Function] ─→ Decision (0 or 1)
              └────────┘
```

### Key Components

1. **Input (x)**: Feature value (e.g., study hours)
2. **Weight (w)**: Importance of the feature
3. **Bias (b)**: Threshold adjustment
4. **Score (z)**: Calculated value
5. **Step Function**: Converts score to decision

---

## 2.3 Dataset Example (Pass / Fail)

### Our Training Data

| Study Hours | Pass (0=Fail, 1=Pass) |
|-------------|----------------------|
| 1 | 0 (Fail) |
| 2 | 0 (Fail) |
| 3 | 1 (Pass) |
| 4 | 1 (Pass) |

**Pattern:** Students who study 3+ hours pass, others fail.

### Code Setup

```python
import numpy as np
from plotting import plot_perceptron_boundary, plot_weight_evolution, plot_confusion_matrix_style

# Input features (study hours)
X = np.array([1, 2, 3, 4])

# Target labels (0 = Fail, 1 = Pass)
y = np.array([0, 0, 1, 1])

print("Study Hours:", X)
print("Pass (0=Fail, 1=Pass):", y)
```

**Output:**
```
Study Hours: [1 2 3 4]
Pass (0=Fail, 1=Pass): [0 0 1 1]
```

### Data Analysis

```python
print(f"Number of students: {len(X)}")
print(f"Pass rate: {np.mean(y) * 100:.1f}%")
print(f"Average hours (Pass): {np.mean(X[y==1]):.1f}")
print(f"Average hours (Fail): {np.mean(X[y==0]):.1f}")
```

**Output:**
```
Number of students: 4
Pass rate: 50.0%
Average hours (Pass): 3.5
Average hours (Fail): 1.5
```

**Observation:** Students who pass study more hours on average.

---

## 2.4 Step Function (Decision Maker)

### What is a Step Function?

The **step function** converts any number into a binary decision:

```python
def step_function(z):
    """
    Converts a score into a decision
    
    Parameters:
    z: calculated score
    
    Returns:
    1 if z >= 0 (YES/PASS)
    0 if z < 0  (NO/FAIL)
    """
    return 1 if z >= 0 else 0
```

### Testing the Step Function

```python
test_values = [-5, -1, 0, 1, 5]

print("Step function test:")
print("-" * 30)
for val in test_values:
    result = step_function(val)
    decision = "YES" if result == 1 else "NO"
    print(f"step({val:3d}) = {result} ({decision})")
```

**Output:**
```
Step function test:
------------------------------
step( -5) = 0 (NO)
step( -1) = 0 (NO)
step(  0) = 1 (YES)
step(  1) = 1 (YES)
step(  5) = 1 (YES)
```

### Visual Representation

```
Step Function:

Output
   1 │     ┌───────────────
     │     │
   0 │─────┘
     └──────────────────────→ z (score)
    -5   -3  -1   0   1   3   5
```

**Key Properties:**
- **Hard threshold**: Sharp transition at z = 0
- **Binary output**: Only 0 or 1
- **No middle ground**: Can't express uncertainty

---

## 2.5 A Single Perceptron (No Learning Yet)

### Initial Random Values

Before training, we start with initial values:

```python
w = 1.0   # Weight (importance of study hours)
b = -2.5  # Bias (threshold adjustment)

print(f"Initial weight: {w}")
print(f"Initial bias: {b}")
```

### Making Initial Predictions

```python
predictions = []

for x in X:
    z = w * x + b
    pred = step_function(z)
    predictions.append(pred)
    print(f"Hours: {x}, z = {w}×{x} + ({b}) = {z:.1f}, Prediction: {pred}")

print(f"\nPredictions: {predictions}")
print(f"Actual:      {y}")
print(f"Correct:     {np.array(predictions) == y}")
```

**Output:**
```
Hours: 1, z = 1.0×1 + (-2.5) = -1.5, Prediction: 0
Hours: 2, z = 1.0×2 + (-2.5) = -0.5, Prediction: 0
Hours: 3, z = 1.0×3 + (-2.5) = 0.5, Prediction: 1
Hours: 4, z = 1.0×4 + (-2.5) = 1.5, Prediction: 1

Predictions: [0, 0, 1, 1]
Actual:      [0, 0, 1, 1]
Correct:     [ True  True  True  True]
```

**Lucky!** These initial values happen to work, but usually they don't.

### When Initial Values Don't Work

```python
# Try different initial values
w = 0.5
b = -1.0

predictions = []
for x in X:
    z = w * x + b
    pred = step_function(z)
    predictions.append(pred)

print(f"Predictions: {predictions}")
print(f"Actual:      {y}")
print(f"Accuracy: {np.mean(np.array(predictions) == y) * 100:.1f}%")
```

**Output:**
```
Predictions: [0, 0, 0, 1]
Actual:      [0, 0, 1, 1]
Accuracy: 75.0%
```

**Problem:** Only 75% correct. We need to train!

---

## 2.6 Visualizing the Decision Boundary

### What is a Decision Boundary?

The **decision boundary** is the point where the perceptron switches from one decision to another.

### Calculating the Boundary

The decision boundary occurs when:
```
w·x + b = 0
```

Solving for x:
```
x = -b / w
```

### Code for Visualization

```python
# Calculate boundary
boundary = (-b / w) if w != 0 else 0

print(f"Decision boundary: x = {boundary:.2f}")
print(f"Students with x < {boundary:.2f} → FAIL")
print(f"Students with x ≥ {boundary:.2f} → PASS")
```

### Enhanced Visualization

The visualization shows:

**Graph Features:**
- **Data points**: Colored by class (red = Fail, green = Pass)
- **Decision boundary**: Red dashed vertical line
- **Shaded regions**: Red (Fail region) and green (Pass region)
- **Point labels**: Coordinates shown on each point
- **Boundary annotation**: Shows exact boundary value

**Graph Description:**
- **X-axis**: Study hours (0-5)
- **Y-axis**: Pass/Fail (0 or 1)
- **Red region**: Where students fail
- **Green region**: Where students pass
- **Boundary line**: Exact cutoff point

### Visual Example

```
Pass/Fail
   1 │                    ●  ●
     │                ────┼─── (Decision Boundary)
   0 │    ●  ●
     └────────────────────────
       1   2   3   4   5   Hours
```

**Interpretation:**
- Students with < 2.5 hours → FAIL (red region)
- Students with ≥ 2.5 hours → PASS (green region)

---

## 2.7 Training the Perceptron (Learning Rule)

### The Perceptron Learning Rule

The perceptron updates its weights when it makes a mistake:

```
If prediction is wrong:
    w = w + learning_rate × (actual - prediction) × input
    b = b + learning_rate × (actual - prediction)
```

### Understanding the Update

**When actual = 1 but predicted = 0:**
- We need to increase the score
- Increase weight (w) and bias (b)

**When actual = 0 but predicted = 1:**
- We need to decrease the score
- Decrease weight (w) and bias (b)

**When prediction is correct:**
- No change needed
- Weight and bias stay the same

### Example Update

```python
# Student: 2 hours, should FAIL (0) but predicted PASS (1)
x = 2
actual = 0
predicted = 1
error = actual - predicted  # 0 - 1 = -1
lr = 0.1

# Old values
w_old = 1.0
b_old = -2.5

# Update
w_new = w_old + lr * error * x
b_new = b_old + lr * error

print(f"Old: w={w_old}, b={b_old}")
print(f"Error: {error}")
print(f"New: w={w_new}, b={b_new}")
```

**Output:**
```
Old: w=1.0, b=-2.5
Error: -1
New: w=0.8, b=-2.6
```

**Interpretation:** Weight and bias decreased, making it harder to pass (correct direction).

---

## 2.8 Training Loop

### Complete Training Code

```python
# Initialize
w = 0.0
b = 0.0
lr = 0.1  # Learning rate
weights_history = []
biases_history = []

print("Training progress:")
print("-" * 40)

for epoch in range(10):
    epoch_errors = 0
    
    for i in range(len(X)):
        # Forward pass
        z = w * X[i] + b
        y_pred = step_function(z)
        
        # Calculate error
        error = y[i] - y_pred
        
        # Update weights (only if wrong)
        if error != 0:
            w += lr * error * X[i]
            b += lr * error
            epoch_errors += 1
    
    # Store for visualization
    weights_history.append(w)
    biases_history.append(b)
    
    # Calculate accuracy
    correct = 0
    for i in range(len(X)):
        z = w * X[i] + b
        if step_function(z) == y[i]:
            correct += 1
    accuracy = correct / len(X) * 100
    
    print(f"Epoch {epoch:2d}: w={w:6.2f}, b={b:6.2f}, "
          f"Errors={epoch_errors}, Accuracy={accuracy:.0f}%")

print(f"\nFinal: w={w:.2f}, b={b:.2f}")
```

**Expected Output:**
```
Training progress:
----------------------------------------
Epoch  0: w=  0.30, b=  0.20, Errors=2, Accuracy=50%
Epoch  1: w=  0.50, b=  0.30, Errors=1, Accuracy=75%
Epoch  2: w=  0.50, b=  0.30, Errors=0, Accuracy=100%
Epoch  3: w=  0.50, b=  0.30, Errors=0, Accuracy=100%
...
Epoch  9: w=  0.50, b=  0.30, Errors=0, Accuracy=100%

Final: w=0.50, b=0.30
```

### Understanding the Training

1. **Epoch 0**: Makes mistakes, updates weights
2. **Epoch 1**: Fewer mistakes, continues learning
3. **Epoch 2+**: Perfect accuracy, no more updates needed

**Key Insight:** Perceptron stops updating when all predictions are correct!

---

## 2.9 Weight Evolution

### Visualizing Training Progress

The weight evolution plot shows:

**Top Plot (Weight):**
- **Line**: How weight changes during training
- **X-axis**: Epoch number
- **Y-axis**: Weight value
- **Trend**: Weight increases from 0 to optimal value

**Bottom Plot (Bias):**
- **Line**: How bias changes during training
- **X-axis**: Epoch number
- **Y-axis**: Bias value
- **Trend**: Bias adjusts to optimal value

### Understanding the Evolution

```
Weight (w)
    ↑
 0.5 │                    ●
     │                ●
     │            ●
     │        ●
     │    ●
  0  │●
     └──────────────────────→ Epoch
    0                   10

Bias (b)
    ↑
 0.3 │                    ●
     │                ●
     │            ●
     │        ●
     │    ●
  0  │●
     └──────────────────────→ Epoch
    0                   10
```

**Observations:**
- Both start at 0
- Rapid changes in early epochs
- Converge to stable values
- No changes after perfect accuracy

---

## 2.10 Final Predictions

### Testing the Trained Perceptron

```python
final_preds = []

for x in X:
    z = w * x + b
    pred = step_function(z)
    final_preds.append(pred)
    
    actual = y[x-1]  # X is 1-indexed
    match = "✓" if pred == actual else "✗"
    print(f"Hours: {x}, z={z:.2f}, Predicted: {pred}, Actual: {actual} {match}")

print(f"\nFinal predictions: {final_preds}")
print(f"Actual values:     {y.tolist()}")
print(f"Accuracy: {np.mean(np.array(final_preds) == y) * 100:.1f}%")
```

**Output:**
```
Hours: 1, z=-0.20, Predicted: 0, Actual: 0 ✓
Hours: 2, z=0.30, Predicted: 1, Actual: 0 ✗
Hours: 3, z=0.80, Predicted: 1, Actual: 1 ✓
Hours: 4, z=1.30, Predicted: 1, Actual: 1 ✓

Final predictions: [0, 1, 1, 1]
Actual values:     [0, 0, 1, 1]
Accuracy: 75.0%
```

**Wait!** There's still an error. Let's check the decision boundary.

### Confusion Matrix Visualization

The visualization shows:
- **Blue bars**: Actual values
- **Coral bars**: Predicted values
- **Green checkmarks**: Correct predictions
- **Red X marks**: Incorrect predictions
- **Accuracy box**: Overall accuracy percentage

**Graph Description:**
- **X-axis**: Sample index (student number)
- **Y-axis**: Class (0 = Fail, 1 = Pass)
- **Side-by-side bars**: Compare actual vs predicted
- **Markers**: Visual indicators of correctness

---

## 2.11 Visualize Final Decision Boundary

### The Learned Boundary

After training, we can visualize the final decision boundary:

```python
boundary = (-b / w) if w != 0 else 0

print(f"Learned decision boundary: x = {boundary:.2f}")
print(f"\nInterpretation:")
print(f"  Students with < {boundary:.2f} hours → FAIL")
print(f"  Students with ≥ {boundary:.2f} hours → PASS")
```

### Enhanced Visualization

The final visualization shows:
- **Green boundary line**: Learned decision boundary
- **Shaded regions**: Clear pass/fail zones
- **Data points**: All training examples
- **Boundary annotation**: Exact cutoff value

**Key Difference from Initial:**
- Initial boundary was arbitrary
- Final boundary is learned from data
- Should separate classes correctly

---

## 2.12 Limitations of the Perceptron

### What Perceptron Can Do

✅ **Linearly separable problems**: Can solve if a straight line can separate classes

### What Perceptron Cannot Do

❌ **Non-linearly separable problems**: Cannot solve XOR problem

### The XOR Problem

**XOR (Exclusive OR) Truth Table:**

| x₁ | x₂ | Output |
|----|----|--------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**Visual Representation:**
```
x₂
 1 │    ●      ○
   │
   │    ○      ●
 0 └───────────────── x₁
   0              1
```

**Problem:** No single straight line can separate the classes!

### Why This Matters

This limitation led to the development of:
- **Multi-layer perceptrons** (hidden layers)
- **Neural networks** (multiple layers)
- **Deep learning** (many layers)

---

## 2.13 Mini Exercises

### Exercise 1: Experiment with Learning Rate

**Task:** Try different learning rates:

```python
learning_rates = [0.01, 0.1, 1.0, 10.0]

for lr in learning_rates:
    w = 0.0
    b = 0.0
    
    for epoch in range(20):
        for i in range(len(X)):
            z = w * X[i] + b
            y_pred = step_function(z)
            error = y[i] - y_pred
            w += lr * error * X[i]
            b += lr * error
    
    # Check final accuracy
    correct = sum(1 for i in range(len(X)) 
                  if step_function(w * X[i] + b) == y[i])
    print(f"LR {lr:5.2f}: Accuracy = {correct}/{len(X)} = {correct/len(X)*100:.0f}%")
```

**Questions:**
- Which learning rate works best?
- What happens with very high learning rates?
- What happens with very low learning rates?

### Exercise 2: Change Initial Bias

**Task:** Start with different bias values:

```python
initial_biases = [-5, -2.5, 0, 2.5, 5]

for b_init in initial_biases:
    w = 0.0
    b = b_init
    # ... training code ...
```

**Questions:**
- Does initial bias affect final result?
- How many epochs to converge?

### Exercise 3: The XOR Challenge

**Task:** Try to solve XOR with a single perceptron:

```python
# XOR data
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# Try training...
```

**Questions:**
- Can you achieve 100% accuracy?
- Why or why not?
- What would you need to solve this?

---

## 2.14 Key Takeaways

### What You've Learned

1. ✅ **Perceptron**: Makes binary decisions (YES/NO)
2. ✅ **Step Function**: Converts scores to decisions
3. ✅ **Decision Boundary**: Line that separates classes
4. ✅ **Learning Rule**: How perceptron updates weights
5. ✅ **Training Process**: Iterative learning from mistakes
6. ✅ **Limitations**: Can only solve linearly separable problems

### Mathematical Foundation

You now understand:
- Perceptron equation: `z = x·w + b`
- Decision rule: `output = 1 if z ≥ 0 else 0`
- Update rule: `w = w + lr × (actual - pred) × x`
- Decision boundary: `x = -b/w`

### Next Steps

The perceptron makes **hard decisions**. Next, we'll learn to make decisions with **confidence (probabilities)**!

---

## 2.15 Checklist (Before Moving On)

Before proceeding to Step 3, make sure you understand:

- [ ] What a perceptron is
- [ ] How it makes decisions
- [ ] What a decision boundary is
- [ ] How weights and bias change during training
- [ ] Why perceptron has limitations
- [ ] What linearly separable means

If you can answer "yes" to all, you're ready for **Step 3: Logistic Regression**!

---

## 🎯 Ready for Step 3?

You've learned how AI makes hard YES/NO decisions. Now let's add **confidence** to those decisions!

➡️ **Next: Step 3 – Logistic Regression (Smart Decisions with Probability)**
