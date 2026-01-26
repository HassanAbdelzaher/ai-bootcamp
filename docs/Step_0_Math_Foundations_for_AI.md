# Step 0 — Math Foundations for AI (Complete Guide)

> **Goal:** Build the math "language" behind AI so students can understand perceptrons, neural networks, and training later.  
> **Tools:** Python + NumPy + Matplotlib  
> **Time:** ~45 minutes  
> **Difficulty:** ⭐ Beginner

---

## 📚 Table of Contents

1. [What AI Really Does](#01-what-ai-really-does-big-idea)
2. [Setup and Installation](#02-setup)
3. [Numbers & Features](#03-numbers--features)
4. [Vectors](#04-vectors)
5. [Weights](#05-weights)
6. [Dot Product](#06-dot-product)
7. [Bias](#07-bias)
8. [Building a Mini Neuron](#08-mini-neuron)
9. [Decision Boundary](#09-decision-boundary)
10. [Multiple Students](#010-multiple-students)
11. [Visualizations](#011-visualizations)
12. [Exercises](#012-exercises)

---

## 0.1 What AI Really Does (Big Idea)

### The Core Concept

AI does **not** think like humans. Instead, AI repeatedly performs **mathematical operations** to transform inputs into outputs.

Think of AI as a calculator that:
- Takes numbers as input
- Performs calculations
- Produces a result

### The Fundamental Equation

A simple "AI brain" (neuron) follows this equation:

```
z = x · w + b
```

**Breaking it down:**
- **x** = inputs (features) - What you feed into the AI
- **w** = weights (importance) - How much each feature matters
- **b** = bias (starting push) - A constant adjustment
- **z** = score (before making a decision) - The final calculation

### Real-World Analogy

Imagine you're deciding if a student passes a class:

```python
# Student's scores
math_score = 80      # x₁
science_score = 70   # x₂
english_score = 75   # x₃

# How important each subject is
math_weight = 0.6    # w₁ (most important)
science_weight = 0.3 # w₂
english_weight = 0.1 # w₃

# Starting adjustment
bias = -50           # b

# Calculate final score
final_score = (80 × 0.6) + (70 × 0.3) + (75 × 0.1) - 50
            = 48 + 21 + 7.5 - 50
            = 26.5
```

If `final_score ≥ 60`, the student **passes**. Otherwise, they **fail**.

---

## 0.2 Setup

### Installing Required Packages

```bash
pip install numpy matplotlib
```

Or if using the project's virtual environment:

```bash
make install
```

### Import Statements

```python
import numpy as np
import matplotlib.pyplot as plt
```

**Why NumPy?**
- Fast mathematical operations
- Handles arrays and matrices efficiently
- Essential for AI/ML work

**Why Matplotlib?**
- Creates visualizations
- Helps understand what's happening
- Makes learning visual and intuitive

---

## 0.3 Numbers & Features

### Understanding Features

In AI, **features** are the characteristics or measurements we use to make predictions.

### Examples of Features

```python
# Example 1: Student grades
study_hours = 4
math_score = 80
science_score = 70

# Example 2: Weather data
temperature = 28.5  # Celsius
humidity = 0.65    # 0 to 1
wind_speed = 15    # km/h

# Example 3: Image pixels
pixel_brightness = 0.92  # 0 (black) to 1 (white)
pixel_red = 0.85
pixel_green = 0.23
pixel_blue = 0.67

print(f"Study hours: {study_hours}")
print(f"Math score: {math_score}")
print(f"Temperature: {temperature}°C")
print(f"Pixel brightness: {pixel_brightness}")
```

**Output:**
```
Study hours: 4
Math score: 80
Temperature: 28.5°C
Pixel brightness: 0.92
```

### Key Points

- Features can be **integers** (whole numbers) or **floats** (decimals)
- Features represent **real-world measurements**
- More features = more information = (usually) better predictions

---

## 0.4 Vectors

### What is a Vector?

A **vector** is a collection of numbers arranged in a specific order. Think of it as a list of features.

### Creating Vectors with NumPy

```python
import numpy as np

# Single student's scores
x = np.array([80, 70, 75])
print("Student vector:", x)
print("Type:", type(x))
print("Shape:", x.shape)
```

**Output:**
```
Student vector: [80 70 75]
Type: <class 'numpy.ndarray'>
Shape: (3,)
```

### Visual Representation

```
Student Vector: [80, 70, 75]
                │   │   │
                │   │   └─ English score
                │   └───── Science score
                └───────── Math score
```

### Why Vectors Matter

- **Efficiency**: Process multiple features at once
- **Speed**: NumPy operations are optimized
- **Clarity**: Represents one data point clearly

### Multiple Students

```python
# Three students' scores
students = np.array([
    [90, 85, 80],  # Student 1: Math, Science, English
    [40, 50, 60],  # Student 2
    [75, 70, 80],  # Student 3
])

print("Students matrix:")
print(students)
print(f"\nShape: {students.shape}")  # (3 students, 3 subjects)
```

**Output:**
```
Students matrix:
[[90 85 80]
 [40 50 60]
 [75 70 80]]

Shape: (3, 3)
```

---

## 0.5 Weights

### What are Weights?

**Weights** determine how important each feature is in the final decision.

### Example: Weighted Grades

```python
# Student's scores
scores = np.array([80, 70, 75])  # Math, Science, English

# Weights (how important each subject is)
weights = np.array([0.6, 0.3, 0.1])

print("Scores:", scores)
print("Weights:", weights)
print("\nInterpretation:")
print(f"  Math (80) × {weights[0]} = {scores[0] * weights[0]}")
print(f"  Science (70) × {weights[1]} = {scores[1] * weights[1]}")
print(f"  English (75) × {weights[2]} = {scores[2] * weights[2]}")
```

**Output:**
```
Scores: [80 70 75]
Weights: [0.6 0.3 0.1]

Interpretation:
  Math (80) × 0.6 = 48.0
  Science (70) × 0.3 = 21.0
  English (75) × 0.1 = 7.5
```

### Key Properties

1. **Weights sum to 1.0** (in this example): `0.6 + 0.3 + 0.1 = 1.0`
2. **Higher weight = more important**
3. **Weights are learned** during training (we'll see this later)

### Visualizing Weights

```
Feature Importance:
Math:    ████████████████████ 60%
Science: ██████████ 30%
English: ████ 10%
```

---

## 0.6 Dot Product

### What is Dot Product?

The **dot product** multiplies corresponding elements and sums them up.

### Manual Calculation

```python
# Student scores
x = np.array([80, 70, 75])

# Weights
w = np.array([0.6, 0.3, 0.1])

# Manual calculation
result = (80 × 0.6) + (70 × 0.3) + (75 × 0.1)
       = 48 + 21 + 7.5
       = 76.5

print("Manual calculation:", result)
```

### Using NumPy

```python
# Using NumPy's dot product
z = np.dot(x, w)
print("NumPy dot product:", z)
```

**Output:**
```
Manual calculation: 76.5
NumPy dot product: 76.5
```

### Visual Breakdown

```
Dot Product Calculation:

[80, 70, 75] · [0.6, 0.3, 0.1]
    │    │    │    │    │    │
    │    │    │    │    │    └─ 75 × 0.1 = 7.5
    │    │    │    │    └────── 70 × 0.3 = 21.0
    │    │    │    └──────────── 80 × 0.6 = 48.0
    │    │    └─────────────────────────────
    │    └─────────────────────────────────────
    └───────────────────────────────────────────
    
    Sum: 48.0 + 21.0 + 7.5 = 76.5
```

### Why Dot Product is Important

- **Single operation** instead of loops
- **Fast** (optimized in NumPy)
- **Foundation** for all neural network calculations

### Feature Contributions Visualization

When you run the code, you'll see a bar chart showing:
- **Bar heights**: Contribution of each feature
- **Colors**: Different colors for each feature
- **Pie chart**: Relative importance

**Graph Description:**
- Left plot: Bar chart with contribution values labeled on each bar
- Right plot: Pie chart showing percentage contribution of each feature

---

## 0.7 Bias

### What is Bias?

**Bias** is a constant value added to adjust the final score. It's like a starting point or threshold adjustment.

### Without Bias

```python
x = np.array([80, 70, 75])
w = np.array([0.6, 0.3, 0.1])
z = np.dot(x, w)

print("Score without bias:", z)
```

**Output:**
```
Score without bias: 76.5
```

### With Bias

```python
b = -50  # Negative bias (makes it harder to pass)
z_with_bias = z + b

print("Score with bias:", z_with_bias)
print(f"Pass? {z_with_bias >= 60}")
```

**Output:**
```
Score with bias: 26.5
Pass? False
```

### Understanding Bias

- **Positive bias**: Makes it easier to get a high score
- **Negative bias**: Makes it harder to get a high score
- **Zero bias**: No adjustment

### Example Scenarios

```python
# Scenario 1: Easy grading (positive bias)
bias_easy = 30
score_easy = z + bias_easy
print(f"Easy grading: {score_easy} → Pass: {score_easy >= 60}")

# Scenario 2: Hard grading (negative bias)
bias_hard = -30
score_hard = z + bias_hard
print(f"Hard grading: {score_hard} → Pass: {score_hard >= 60}")

# Scenario 3: Neutral (zero bias)
bias_neutral = 0
score_neutral = z + bias_neutral
print(f"Neutral: {score_neutral} → Pass: {score_neutral >= 60}")
```

**Output:**
```
Easy grading: 106.5 → Pass: True
Hard grading: 46.5 → Pass: False
Neutral: 76.5 → Pass: True
```

---

## 0.8 Mini Neuron

### Building Our First Neuron

A **neuron** is the basic building block of AI. It takes inputs, applies weights and bias, and produces an output.

### Neuron Function

```python
def neuron(x, w, b):
    """
    Simple neuron function
    
    Parameters:
    x: input features (vector)
    w: weights (vector)
    b: bias (scalar)
    
    Returns:
    z: output score
    """
    return np.dot(x, w) + b

# Test the neuron
x = np.array([80, 70, 75])
w = np.array([0.6, 0.3, 0.1])
b = -50

result = neuron(x, w, b)
print(f"Neuron output: {result}")
```

**Output:**
```
Neuron output: 26.5
```

### Neuron Diagram

```
Input Features (x)
    │
    ├─→ [× w₁] ─┐
    ├─→ [× w₂] ─┤
    └─→ [× w₃] ─┼─→ [Sum] ─→ [+ b] ─→ Output (z)
                └──────────┘
```

### Testing Multiple Students

```python
# Multiple students
students = np.array([
    [90, 85, 80],
    [40, 50, 60],
    [75, 70, 80],
])

# Same weights and bias for all
w = np.array([0.6, 0.3, 0.1])
b = -50

# Calculate scores for all students
scores = []
for student in students:
    score = neuron(student, w, b)
    scores.append(score)
    print(f"Student {student}: Score = {score:.1f}")

print(f"\nAll scores: {scores}")
```

**Output:**
```
Student [90 85 80]: Score = 79.0
Student [40 50 60]: Score = 9.0
Student [75 70 80]: Score = 76.0

All scores: [79.0, 9.0, 76.0]
```

---

## 0.9 Decision Boundary

### Making Decisions

After calculating the score, we need to make a decision based on a **threshold**.

### Decision Function

```python
def make_decision(score, threshold=60):
    """
    Make a decision based on score
    
    Parameters:
    score: calculated score
    threshold: minimum score to pass
    
    Returns:
    1 if pass, 0 if fail
    """
    return 1 if score >= threshold else 0

# Test
scores = [79.0, 9.0, 76.0]
threshold = 60

for i, score in enumerate(scores):
    decision = make_decision(score, threshold)
    status = "PASS" if decision == 1 else "FAIL"
    print(f"Student {i+1}: Score = {score:.1f} → {status}")
```

**Output:**
```
Student 1: Score = 79.0 → PASS
Student 2: Score = 9.0 → FAIL
Student 3: Score = 76.0 → PASS
```

### Visualizing Decision Boundary

The decision boundary is the line that separates:
- **Pass** (score ≥ threshold)
- **Fail** (score < threshold)

```
Score Scale:
0 ──────────────────── 60 ──────────────────── 100
│                        │                        │
FAIL                    THRESHOLD                PASS
```

---

## 0.10 Multiple Students

### Matrix Operations

When dealing with multiple students, we can use **matrix multiplication** for efficiency.

### Using Matrix Multiplication

```python
# Multiple students (each row is a student)
X = np.array([
    [90, 85, 70],  # Student 1
    [40, 50, 60],  # Student 2
    [75, 70, 80],  # Student 3
    [55, 60, 58],  # Student 4
])

# Weights (column vector)
w = np.array([0.6, 0.3, 0.1])
b = -50

# Matrix multiplication: X @ w
# This calculates dot product for each row
scores = X @ w + b

print("Students matrix:")
print(X)
print(f"\nWeights: {w}")
print(f"Bias: {b}")
print(f"\nScores for all students:")
for i, score in enumerate(scores):
    print(f"  Student {i+1}: {score:.1f}")
```

**Output:**
```
Students matrix:
[[90 85 70]
 [40 50 60]
 [75 70 80]
 [55 60 58]]

Weights: [0.6 0.3 0.1]
Bias: -50

Scores for all students:
  Student 1: 79.0
  Student 2: 9.0
  Student 3: 76.0
  Student 4: 26.5
```

### Why Matrix Operations?

- **Faster**: One operation instead of loops
- **Cleaner**: Less code, easier to read
- **Scalable**: Works with any number of students

### Visual Representation

```
Input Matrix (X)    Weights (w)    Bias (b)    Output (scores)
┌─────────┐         ┌─────┐                    ┌──────┐
│ 90 85 70│    @   │ 0.6 │    +    -50   =   │ 79.0 │
│ 40 50 60│         │ 0.3 │                    │  9.0 │
│ 75 70 80│         │ 0.1 │                    │ 76.0 │
│ 55 60 58│         └─────┘                    │ 26.5 │
└─────────┘                                    └──────┘
```

---

## 0.11 Visualizations

### What You'll See

When you run `step_0_math_foundations.py`, you'll see several visualizations:

#### 1. Feature Contributions Bar Chart

**Description:**
- **X-axis**: Feature names (Math, Science, English)
- **Y-axis**: Contribution value
- **Bars**: Height shows how much each feature contributes
- **Colors**: Different color for each feature
- **Labels**: Exact values on top of each bar

**What it shows:**
- Which features contribute most
- Relative importance of each feature

#### 2. Feature Contributions Pie Chart

**Description:**
- **Slices**: Each slice represents one feature
- **Size**: Proportional to contribution
- **Percentages**: Shows exact percentage contribution
- **Colors**: Matches bar chart colors

**What it shows:**
- Visual representation of relative importance
- Easy to see which feature dominates

### Interpreting the Graphs

1. **Bar Chart**: Use for exact values
2. **Pie Chart**: Use for relative proportions
3. **Together**: Give complete picture of feature importance

---

## 0.12 Exercises

### Exercise 1: Experiment with Weights

**Task:** Change the weights and observe the impact.

```python
# Original weights
w1 = np.array([0.6, 0.3, 0.1])

# Try these variations:
w2 = np.array([0.1, 0.1, 0.8])  # English is most important
w3 = np.array([0.33, 0.33, 0.34])  # Equal importance
w4 = np.array([0.8, 0.15, 0.05])  # Math is very important

# Test with same student
student = np.array([80, 70, 75])
b = -50

for i, w in enumerate([w1, w2, w3, w4], 1):
    score = np.dot(student, w) + b
    print(f"Weight set {i}: Score = {score:.1f}")
```

**Questions to think about:**
- How do different weights change the final score?
- Which weight set gives the highest score? Why?

### Exercise 2: Experiment with Bias

**Task:** Change the bias value and see how it affects decisions.

```python
student = np.array([80, 70, 75])
w = np.array([0.6, 0.3, 0.1])
threshold = 60

biases = [-70, -50, -30, 0, 30]

for b in biases:
    score = np.dot(student, w) + b
    decision = "PASS" if score >= threshold else "FAIL"
    print(f"Bias {b:3d}: Score = {score:5.1f} → {decision}")
```

**Questions to think about:**
- How does bias affect the decision?
- What bias value makes everyone pass? Fail?

### Exercise 3: Implement Pass/Fail Function

**Task:** Create a function that takes student scores and returns pass/fail.

```python
def check_pass_fail(student_scores, weights, bias, threshold=60):
    """
    Check if a student passes
    
    Parameters:
    student_scores: array of [math, science, english]
    weights: array of [math_weight, science_weight, english_weight]
    bias: bias value
    threshold: minimum score to pass
    
    Returns:
    (score, decision) tuple
    """
    score = np.dot(student_scores, weights) + bias
    decision = "PASS" if score >= threshold else "FAIL"
    return score, decision

# Test
student = np.array([80, 70, 75])
w = np.array([0.6, 0.3, 0.1])
b = -50

score, decision = check_pass_fail(student, w, b)
print(f"Score: {score:.1f}, Decision: {decision}")
```

### Exercise 4: Multiple Students Analysis

**Task:** Process multiple students and find statistics.

```python
students = np.array([
    [90, 85, 80],
    [40, 50, 60],
    [75, 70, 80],
    [55, 60, 58],
    [95, 90, 88],
])

w = np.array([0.6, 0.3, 0.1])
b = -50
threshold = 60

scores = students @ w + b
decisions = (scores >= threshold).astype(int)

print("Results:")
for i, (score, decision) in enumerate(zip(scores, decisions)):
    status = "PASS" if decision == 1 else "FAIL"
    print(f"Student {i+1}: {score:.1f} → {status}")

print(f"\nStatistics:")
print(f"  Average score: {np.mean(scores):.1f}")
print(f"  Pass rate: {np.mean(decisions) * 100:.1f}%")
print(f"  Highest score: {np.max(scores):.1f}")
print(f"  Lowest score: {np.min(scores):.1f}")
```

---

## 0.13 Key Takeaways

### What You've Learned

1. ✅ **Features**: Real-world measurements used as inputs
2. ✅ **Vectors**: Collections of features
3. ✅ **Weights**: Importance of each feature
4. ✅ **Dot Product**: Efficient way to combine features and weights
5. ✅ **Bias**: Constant adjustment to the score
6. ✅ **Neuron**: Basic building block that calculates: `z = x·w + b`
7. ✅ **Decision**: Using threshold to make pass/fail decisions
8. ✅ **Matrix Operations**: Processing multiple examples efficiently

### Mathematical Foundation

You now understand:
- How AI represents data (vectors)
- How AI combines information (dot product)
- How AI makes adjustments (bias)
- How AI makes decisions (threshold)

### Next Steps

You're ready to learn:
- **Step 1**: How AI learns from mistakes (Linear Regression)
- **Step 2**: How AI makes decisions (Perceptron)
- **Step 3**: How AI expresses confidence (Logistic Regression)

---

## 0.14 Checklist (Before Moving On)

Before proceeding to Step 1, make sure you understand:

- [ ] What a feature is
- [ ] How to create vectors with NumPy
- [ ] What weights represent
- [ ] How to calculate dot product
- [ ] What bias does
- [ ] How a neuron calculates its output
- [ ] How to make a decision based on a score

If you can answer "yes" to all, you're ready for **Step 1: Linear Regression**!

---

## 🎯 Ready for Step 1?

You've built the mathematical foundation. Now let's see how AI **learns** from data!

➡️ **Next: Step 1 – Linear Regression (Learning to Predict)**
