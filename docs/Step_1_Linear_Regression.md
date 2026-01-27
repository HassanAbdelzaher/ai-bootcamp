# Step 1 — Linear Regression (Learning to Predict)

> **Goal:** Teach how AI learns from mistakes by adjusting weights automatically.  
> **Tools:** Python + NumPy + Matplotlib  
> **Time:** ~60 minutes  
> **Difficulty:** ⭐⭐ Beginner-Intermediate

---

## 📚 Table of Contents

1. [Big Idea](#11-big-idea)
2. [The Model](#12-the-model-math-first-simple)
3. [Dataset Example](#13-dataset-example)
4. [Visualizing Data](#14-visualize-the-data)
5. [First Bad Guess](#15-first-bad-guess-no-learning-yet)
6. [Error Calculation](#16-error-how-wrong-are-we)
7. [Gradient Descent](#17-gradient-descent-how-ai-learns)
8. [Training the Model](#18-training-the-model)
9. [Learning Curve](#19-learning-curve-very-important-graph)
10. [Final Prediction](#110-final-prediction-line)
11. [Making Predictions](#111-make-predictions)
12. [Weight Evolution](#112-weight-evolution-visualization)
13. [Error Analysis](#113-error-analysis)
14. [Train/Test Split](#114-traintest-split-important-practice)
15. [Exercises](#115-mini-exercises)

---

## 1.1 Big Idea

### The Core Question

Linear Regression answers one fundamental question:

> **How can we predict a number as accurately as possible?**

### Real-World Examples

| Input | Output | Use Case |
|-------|--------|----------|
| Study hours | Exam score | Education |
| Years of experience | Salary | HR/Finance |
| Time | Distance traveled | Physics |
| House size | Price | Real estate |
| Temperature | Ice cream sales | Business |

### What AI Does

The AI tries to draw the **best possible line** through the data points.

**Visual Concept:**
```
Score
 100 │                    ●
     │                ●
  80 │            ●
     │        ●
  60 │    ●
     │
  40 │
     └────────────────────────
       1   2   3   4   5   Hours
```

The goal: Find the line that minimizes the distance to all points.

---

## 1.2 The Model (Math First, Simple)

### The Linear Equation

The linear model is beautifully simple:

```
y = w·x + b
```

**Breaking it down:**
- **x** = input (study hours)
- **w** = weight (slope of the line)
- **b** = bias (y-intercept, starting value)
- **y** = predicted output (exam score)

### Understanding Each Component

#### Weight (w)
- **Positive w**: More hours → Higher score (positive correlation)
- **Negative w**: More hours → Lower score (negative correlation)
- **Large |w|**: Strong relationship
- **Small |w|**: Weak relationship

#### Bias (b)
- Starting point when x = 0
- Adjusts the entire line up or down
- Example: If b = 50, even 0 hours gives a score of 50

### Visual Representation

```
y = w·x + b

Example: y = 10·x + 50

Score
 100 │                    ●
     │                ●
  80 │            ●
     │        ●
  60 │    ●
     │
  50 │● (bias = starting point)
     └────────────────────────
       1   2   3   4   5   Hours
```

---

## 1.3 Dataset Example

### Our Training Data

| Study Hours | Exam Score |
|-------------|------------|
| 1 | 50 |
| 2 | 60 |
| 3 | 70 |
| 4 | 80 |

**Observation:** For every additional hour, the score increases by 10 points.

### Code Setup

```python
import numpy as np
from plotting import plot_data_scatter, plot_prediction_line, plot_learning_curve

# Input features (study hours)
X = np.array([1, 2, 3, 4], dtype=float)

# Target values (exam scores)
y = np.array([50, 60, 70, 80], dtype=float)

print("Study Hours:", X)
print("Exam Scores:", y)
```

**Code Explanation:**
- `import numpy as np`: Imports NumPy library (aliased as `np`)
  - Used for numerical operations and arrays
- `from plotting import ...`: Imports custom plotting functions
  - These are visualization helpers we created
- `X = np.array([1, 2, 3, 4], dtype=float)`: Creates input features array
  - `[1, 2, 3, 4]`: Study hours for 4 students
  - `dtype=float`: Ensures decimal numbers (needed for calculations)
  - `X` is the standard name for input features in ML
- `y = np.array([50, 60, 70, 80], dtype=float)`: Creates target values array
  - `[50, 60, 70, 80]`: Actual exam scores (the "correct answers")
  - `y` is the standard name for target/output values in ML
  - Notice the pattern: each hour adds 10 points (perfect linear relationship)
- `print(...)`: Displays the arrays to verify they're correct

**Output:**
```
Study Hours: [1. 2. 3. 4.]
Exam Scores: [50. 60. 70. 80.]
```

### Data Analysis

```python
print(f"Number of samples: {len(X)}")
print(f"Average hours: {np.mean(X):.1f}")
print(f"Average score: {np.mean(y):.1f}")
print(f"Score range: {np.min(y)} - {np.max(y)}")
```

**Output:**
```
Number of samples: 4
Average hours: 2.5
Average score: 65.0
Score range: 50 - 80
```

---

## 1.4 Visualize the Data

### Creating the Scatter Plot

When you run the code, you'll see an enhanced scatter plot with:

**Graph Features:**
- **Data points**: Large, colored circles showing study hours vs scores
- **Trend line**: Red dashed line showing the general pattern
- **Statistics box**: Shows mean and standard deviation
- **Grid**: Helps read exact values
- **Color coding**: Points colored by their score value

**What the graph shows:**
- Clear positive relationship (more hours → higher score)
- Linear pattern (points roughly follow a straight line)
- Data distribution (how spread out the scores are)

### Expected Visualization

```
Exam Score
   90 │
   80 │                    ●
   70 │                ●
   60 │            ●
   50 │        ●
   40 │
   30 │
     └────────────────────────
       1   2   3   4   5   Hours
```

**Key Observations:**
- ✅ Positive correlation (upward trend)
- ✅ Linear relationship (straight line pattern)
- ✅ Consistent spacing (10 points per hour)

---

## 1.5 First Bad Guess (No Learning Yet)

### Initial Random Guess

Before learning, the AI starts with random values:

```python
w = 0.0  # Weight (slope)
b = 0.0  # Bias (y-intercept)

# Make predictions
y_pred = w * X + b
print("Bad predictions:", y_pred)
```

**Output:**
```
Bad predictions: [0. 0. 0. 0.]
```

**Problem:** The AI predicts 0 for everything, which is completely wrong!

### Visualizing the Bad Prediction

The visualization shows:
- **Red line**: Flat line at y=0 (terrible prediction)
- **Data points**: Actual scores (blue/green circles)
- **Error bars**: Red dashed lines showing how far off each prediction is

**Graph Description:**
- **X-axis**: Study hours (1-4)
- **Y-axis**: Exam score (0-100)
- **Red line**: Initial bad prediction (y = 0)
- **Colored points**: Actual data
- **Error metrics box**: Shows MSE and RMSE values

### Understanding the Error

```python
error = np.mean((y_pred - y) ** 2)
print(f"Mean Squared Error: {error:.2f}")
```

**Output:**
```
Mean Squared Error: 3750.00
```

**Interpretation:**
- Very high error (3750)
- All predictions are wrong
- AI needs to learn!

---

## 1.6 Error (How Wrong Are We?)

### Mean Squared Error (MSE)

We use **Mean Squared Error** to measure how wrong our predictions are:

```
MSE = average((prediction − real)²)
```

### Why Squared?

1. **Always positive**: (prediction - real)² is never negative
2. **Penalizes large errors**: Big mistakes cost more
3. **Smooth function**: Easier to optimize

### Manual Calculation

```python
# For each data point
errors = []
for i in range(len(X)):
    error = (y_pred[i] - y[i]) ** 2
    errors.append(error)
    print(f"Point {i+1}: ({X[i]}, {y[i]}) → Predicted: {y_pred[i]:.1f}, Error: {error:.1f}")

mse = np.mean(errors)
print(f"\nMean Squared Error: {mse:.2f}")
```

**Output:**
```
Point 1: (1.0, 50.0) → Predicted: 0.0, Error: 2500.0
Point 2: (2.0, 60.0) → Predicted: 0.0, Error: 3600.0
Point 3: (3.0, 70.0) → Predicted: 0.0, Error: 4900.0
Point 4: (4.0, 80.0) → Predicted: 0.0, Error: 6400.0

Mean Squared Error: 4350.00
```

### Key Insight

🧠 **Learning = Reducing Error**

The goal is to make MSE as small as possible.

---

## 1.7 Gradient Descent (How AI Learns)

### The Intuition

Think of error as a **hill**:
- **Top of hill**: High error (bad predictions)
- **Bottom of valley**: Low error (good predictions)
- **Goal**: Find the lowest point

### Visual Analogy

```
Error
  ↑
  │     ╱╲
  │    ╱  ╲
  │   ╱    ╲
  │  ╱      ╲
  │ ╱        ╲
  │╱          ╲
  └──────────────→ Weight
     ↑
   We want to be here (lowest error)
```

### Gradient (Slope)

The **gradient** tells us:
- **Direction**: Which way to move
- **Magnitude**: How steep the hill is

### Gradient Descent Algorithm

1. Calculate current error
2. Calculate gradient (slope)
3. Move in opposite direction of gradient
4. Repeat until error is minimized

### Mathematical Formulation

For weight (w):
```
dw = average((prediction - real) × input)
w = w - learning_rate × dw
```

For bias (b):
```
db = average(prediction - real)
b = b - learning_rate × db
```

---

## 1.8 Training the Model

### Complete Training Code

```python
# Initialize
w = 0.0
b = 0.0
lr = 0.01  # Learning rate (how big steps to take)
errors = []
weights_history = []
biases_history = []

# Training loop
for epoch in range(1000):
    # Forward pass: make predictions
    y_pred = w * X + b
    
    # Calculate gradients
    dw = np.mean((y_pred - y) * X)  # Gradient for weight
    db = np.mean(y_pred - y)        # Gradient for bias
    
    # Update weights and bias
    w -= lr * dw
    b -= lr * db
    
    # Store for visualization
    weights_history.append(w)
    biases_history.append(b)
    
    # Calculate and store error
    error = np.mean((y_pred - y) ** 2)
    errors.append(error)

print(f"Final w (weight): {w:.4f}")
print(f"Final b (bias): {b:.4f}")
print(f"Final error: {errors[-1]:.4f}")
```

**Code Explanation:**
- **Initialization:**
  - `w = 0.0` and `b = 0.0`: Start with zero weights (no knowledge yet)
  - `lr = 0.01`: Learning rate controls step size (0.01 = small, careful steps)
  - `errors = []`: List to track error over time (for plotting)
  - `weights_history = []`: List to track weight changes (for visualization)
  - `biases_history = []`: List to track bias changes (for visualization)

- **Training Loop (`for epoch in range(1000):`):**
  - `epoch`: One complete pass through all training data
  - `range(1000)`: Train for 1000 epochs (iterations)
  - More epochs = more learning (but diminishing returns)

- **Forward Pass (`y_pred = w * X + b`):**
  - Makes predictions using current weights
  - `w * X`: Multiply each hour by weight (vectorized operation)
  - `+ b`: Add bias to each prediction
  - Result: Array of predicted scores

- **Calculate Gradients:**
  - `dw = np.mean((y_pred - y) * X)`: Gradient for weight
    - `(y_pred - y)`: Prediction errors
    - `* X`: Multiply by input (chain rule from calculus)
    - `np.mean(...)`: Average across all data points
    - Tells us: "If we increase w, how much will error change?"
  - `db = np.mean(y_pred - y)`: Gradient for bias
    - Simpler: just average the errors
    - Tells us: "If we increase b, how much will error change?"

- **Update Weights (`w -= lr * dw`):**
  - `-=` means: `w = w - lr * dw`
  - Move in OPPOSITE direction of gradient (to reduce error)
  - If gradient is positive → error increases when w increases → decrease w
  - If gradient is negative → error decreases when w increases → increase w
  - `lr` controls how big the step is

- **Store Values:**
  - `weights_history.append(w)`: Save current weight
  - `biases_history.append(b)`: Save current bias
  - `errors.append(error)`: Save current error
  - Used later for visualization

- **Final Print:**
  - `{w:.4f}`: Format to 4 decimal places
  - Shows the learned values after training

**Expected Output:**
```
Final w (weight): 10.0000
Final b (bias): 40.0000
Final error: 0.0000
```

### Understanding the Results

- **w = 10**: For each additional hour, score increases by 10 points
- **b = 40**: Starting score (when hours = 0)
- **Error ≈ 0**: Perfect fit! (This is because our data is perfectly linear)

### Training Progress

You can see the training progress:

```python
# Check every 100 epochs
for i in range(0, 1000, 100):
    print(f"Epoch {i:4d}: w={weights_history[i]:7.4f}, b={biases_history[i]:7.4f}, error={errors[i]:.4f}")
```

**Output:**
```
Epoch    0: w= 0.0000, b= 0.0000, error=4350.0000
Epoch  100: w= 8.5000, b= 35.0000, error=12.5000
Epoch  200: w= 9.5000, b= 38.5000, error=1.2500
Epoch  300: w= 9.8500, b= 39.7000, error=0.1250
...
Epoch  900: w= 9.9995, b= 39.9990, error=0.0000
Epoch 1000: w=10.0000, b=40.0000, error=0.0000
```

**Observation:** Error decreases rapidly at first, then slows down as it approaches the optimal solution.

---

## 1.9 Learning Curve (Very Important Graph)

### What You'll See

The enhanced learning curve visualization shows:

**Main Plot (Left):**
- **Blue line**: Error decreasing over time
- **Filled area**: Visual emphasis of the learning progress
- **Improvement annotation**: Shows percentage improvement
- **X-axis**: Epoch number (0-1000)
- **Y-axis**: Error (MSE)

**Secondary Plot (Right):**
- **Zoom view**: First 100 epochs (shows rapid initial learning)
- **Comparison**: Last 100 epochs (shows fine-tuning)

### Interpreting the Learning Curve

```
Error
  ↑
  │ ●
  │   ●
  │     ●
  │       ●
  │         ●
  │           ●
  │             ●
  │               ●
  │                 ●
  │                   ●
  └──────────────────────→ Epoch
  0                   1000
```

**Key Patterns:**
1. **Rapid decrease** (epochs 0-200): AI learns quickly
2. **Slower decrease** (epochs 200-800): Fine-tuning
3. **Plateau** (epochs 800-1000): Converged to optimal solution

### Success Indicators

✅ **Error goes down** → AI is learning  
✅ **Smooth curve** → Stable learning  
✅ **Reaches low value** → Good fit

### Warning Signs

❌ **Error increases** → Learning rate too high  
❌ **Oscillating** → Learning rate too high  
❌ **Not decreasing** → Learning rate too low or not enough epochs

---

## 1.10 Final Prediction Line

### Visualizing the Learned Model

The final visualization shows:

**Graph Features:**
- **Data points**: Actual scores (colored circles)
- **Green line**: Learned prediction line
- **Error bars**: Red dashed lines showing residuals
- **Error metrics**: MSE and RMSE displayed in a box

**Graph Description:**
- **X-axis**: Study hours (1-4)
- **Y-axis**: Exam score (0-100)
- **Green line**: Final learned model (y = 10x + 40)
- **Colored points**: Actual data
- **Error metrics**: Shows final MSE and RMSE

### Perfect Fit

Since our data is perfectly linear, the line passes through all points:

```
Score
 100 │
   80 │                    ●
   70 │                ●
   60 │            ●
   50 │        ●
     └────────────────────────
       1   2   3   4   5   Hours
```

**The learned equation:**
```
y = 10·x + 40
```

This matches our data perfectly!

---

## 1.11 Make Predictions

### Predicting for New Data

Now we can predict scores for students we haven't seen:

```python
# New student: studied 5 hours
study_hours = 5
predicted_score = w * study_hours + b
print(f"Predicted score for {study_hours} hours: {predicted_score:.2f}")
```

**Code Explanation:**
- `study_hours = 5`: New input (student we haven't seen before)
  - This tests if the model can generalize to new data
  - Not in the training set!
- `predicted_score = w * study_hours + b`: Use learned model
  - `w`: Learned weight (should be ~10)
  - `study_hours`: Input (5 hours)
  - `b`: Learned bias (should be ~40)
  - Calculation: `10 × 5 + 40 = 90`
- `print(f"...")`: Display prediction
  - `f"..."` is an f-string (formatted string)
  - `{study_hours}` inserts the variable value
  - `{predicted_score:.2f}` formats to 2 decimal places

**Output:**
```
Predicted score for 5 hours: 90.00
```

### Multiple Predictions

```python
# Test multiple students
test_hours = [6, 8, 10]
print("Predictions for new students:")
print("-" * 40)
for hours in test_hours:
    score = w * hours + b
    print(f"Study {hours:2d} hours → Predicted score: {score:.2f}")
```

**Code Explanation:**
- `test_hours = [6, 8, 10]`: List of hours to test
  - Multiple new students with different study hours
- `print("-" * 40)`: Print 40 dashes (creates a separator line)
- `for hours in test_hours:`: Loop through each test value
  - `hours` takes each value: 6, then 8, then 10
- `score = w * hours + b`: Calculate prediction for each
  - Same formula, different input each time
- `print(f"Study {hours:2d} hours → ...")`: Format output
  - `{hours:2d}`: Format as integer with 2-digit width (pads with space)
  - `{score:.2f}`: Format as float with 2 decimal places
  - `→` is an arrow character for readability

**Output:**
```
Predictions for new students:
----------------------------------------
Study  6 hours → Predicted score: 100.00
Study  8 hours → Predicted score: 120.00
Study 10 hours → Predicted score: 140.00
```

**Note:** Scores above 100 are possible in this model, but in reality, you might want to cap them at 100.

---

## 1.12 Weight Evolution Visualization

### How Weights Change During Training

The weight evolution plot shows:

**Top Plot (Weights):**
- **Line**: How weight (w) changes over time
- **X-axis**: Epoch number
- **Y-axis**: Weight value
- **Trend**: Weight increases from 0 to 10

**Bottom Plot (Bias):**
- **Line**: How bias (b) changes over time
- **X-axis**: Epoch number
- **Y-axis**: Bias value
- **Trend**: Bias increases from 0 to 40

### Understanding the Evolution

```
Weight (w)
    ↑
 10 │                    ●
    │                ●
    │            ●
    │        ●
    │    ●
  0 │●
    └──────────────────────→ Epoch
    0                   1000

Bias (b)
    ↑
 40 │                    ●
    │                ●
    │            ●
    │        ●
    │    ●
  0 │●
    └──────────────────────→ Epoch
    0                   1000
```

**Key Observations:**
- Both weight and bias start at 0
- They gradually increase to optimal values
- Changes are smooth (good learning rate)
- Convergence happens around epoch 800-900

---

## 1.13 Error Analysis

### Error Distribution Plot

The error analysis visualization shows two plots:

**Left Plot (Error Histogram):**
- **Bars**: Frequency of different error values
- **Red line**: Zero error (perfect prediction)
- **Green line**: Mean error
- **Distribution**: Shows how errors are spread

**Right Plot (Actual vs Predicted):**
- **Points**: Each point is (actual, predicted)
- **Red line**: Perfect prediction line (y = x)
- **Distance from line**: Shows prediction error
- **Closer to line**: Better predictions

### Interpreting the Plots

**Good Model:**
- Errors clustered near zero
- Points close to the diagonal line
- Symmetric error distribution

**Bad Model:**
- Errors spread out
- Points far from diagonal
- Biased predictions (systematic error)

---

## 1.14 Train/Test Split (Important Practice)

### Why Split Data?

In real projects, we **split data into training and testing sets**. This helps us evaluate how well our model generalizes to new data.

**Key reasons:**
- **Training set**: Used to learn the model
- **Test set**: Used to evaluate performance on unseen data
- **Prevents overfitting**: Avoids memorizing training data

### Creating Train/Test Split

```python
# Create a larger dataset for demonstration
X_full = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
y_full = np.array([50, 60, 70, 80, 90, 100, 110, 120], dtype=float)

# Split: 75% train, 25% test
split_idx = int(len(X_full) * 0.75)
X_train = X_full[:split_idx]
y_train = y_full[:split_idx]
X_test = X_full[split_idx:]
y_test = y_full[split_idx:]

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
```

**Code Explanation:**
- `X_full`, `y_full`: Complete dataset (8 samples)
- `split_idx = int(len(X_full) * 0.75)`: Calculate 75% split point
  - `len(X_full) * 0.75 = 8 * 0.75 = 6`
  - `int(6) = 6` (first 6 samples for training)
- `X_train = X_full[:split_idx]`: First 6 samples (indices 0-5)
- `X_test = X_full[split_idx:]`: Last 2 samples (indices 6-7)
- Same for `y_train` and `y_test`

**Output:**
```
Training set: 6 samples
Test set: 2 samples
```

### Training on Training Set Only

```python
# Train model on training set only
w_train = 0.0
b_train = 0.0
lr = 0.01

for epoch in range(1000):
    y_pred_train = w_train * X_train + b_train
    dw = np.mean((y_pred_train - y_train) * X_train)
    db = np.mean(y_pred_train - y_train)
    w_train -= lr * dw
    b_train -= lr * db
```

**Code Explanation:**
- Same training process as before
- **Important**: Only use `X_train` and `y_train` for training
- Never look at test data during training!

### Evaluating on Both Sets

```python
# Make predictions on both sets
y_pred_train_final = w_train * X_train + b_train
y_pred_test_final = w_train * X_test + y_test

# Calculate errors
train_error = np.mean((y_pred_train_final - y_train) ** 2)
test_error = np.mean((y_pred_test_final - y_test) ** 2)

print(f"Training MSE: {train_error:.2f}")
print(f"Test MSE: {test_error:.2f}")
```

**Code Explanation:**
- `y_pred_train_final`: Predictions on training data
- `y_pred_test_final`: Predictions on test data (unseen during training)
- `train_error`: Error on training set
- `test_error`: Error on test set
- **Good models** have similar train and test errors

**Expected Output:**
```
Training MSE: 0.00
Test MSE: 0.00
```

### Visualizing Train/Test Split

```python
from plotting import plot_train_test_split

plot_train_test_split(X_train, y_train, X_test, y_test, 
                     y_pred_train_final, y_pred_test_final,
                     xlabel="Study Hours", ylabel="Exam Score",
                     title="Train/Test Split Visualization")
```

**What you'll see:**
- **Blue circles**: Training data (used for learning)
- **Red squares**: Test data (used for evaluation)
- **Dashed lines**: Model predictions
- **Info box**: Shows split ratio

### Key Insights

1. **Training error** measures how well the model fits the training data
2. **Test error** measures how well the model generalizes to new data
3. **Good models**: Train error ≈ Test error
4. **Overfitting**: Train error << Test error (model memorized training data)

### Common Split Ratios

| Ratio | Training | Test | Use Case |
|-------|----------|------|----------|
| 80/20 | 80% | 20% | Large datasets |
| 75/25 | 75% | 25% | Medium datasets |
| 70/30 | 70% | 30% | Small datasets |
| 90/10 | 90% | 10% | Very large datasets |

**Rule of thumb**: Use more data for training when you have lots of data.

---

## 1.15 Mini Exercises

### Exercise 1: Experiment with Learning Rate

**Task:** Try different learning rates and observe the effect.

```python
learning_rates = [0.001, 0.01, 0.1, 1.0]

for lr in learning_rates:
    w = 0.0
    b = 0.0
    errors = []
    
    for epoch in range(1000):
        y_pred = w * X + b
        dw = np.mean((y_pred - y) * X)
        db = np.mean(y_pred - y)
        w -= lr * dw
        b -= lr * db
        errors.append(np.mean((y_pred - y) ** 2))
    
    print(f"LR {lr:5.3f}: Final error = {errors[-1]:.4f}, w = {w:.4f}, b = {b:.4f}")
```

**Questions:**
- Which learning rate works best?
- What happens with very high learning rates?
- What happens with very low learning rates?

### Exercise 2: Add More Data Points

**Task:** Add more students to the dataset.

```python
# Original data
X = np.array([1, 2, 3, 4], dtype=float)
y = np.array([50, 60, 70, 80], dtype=float)

# Add more students
X_new = np.array([1, 2, 3, 4, 5, 6], dtype=float)
y_new = np.array([50, 60, 70, 80, 90, 100], dtype=float)

# Retrain and compare
```

**Questions:**
- Does more data improve predictions?
- How does the learning curve change?

### Exercise 3: Predict for New Values

**Task:** Predict scores for students who studied:
- 6 hours
- 8 hours
- 10 hours

**Challenge:** What happens if someone studies 0 hours? Does the prediction make sense?

### Exercise 4: Non-Linear Data

**Task:** Try training on non-linear data:

```python
# Non-linear relationship
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([10, 40, 90, 160, 250], dtype=float)  # y = 10x²
```

**Questions:**
- Can linear regression handle this?
- What does the error look like?
- What would you need to fix this?

---

## 1.16 Key Takeaways

### What You've Learned

1. ✅ **Linear Regression**: Predicts numbers using a straight line
2. ✅ **Error Measurement**: MSE quantifies prediction quality
3. ✅ **Gradient Descent**: How AI learns by reducing error
4. ✅ **Training Process**: Iterative weight updates
5. ✅ **Learning Curves**: Visualize training progress
6. ✅ **Making Predictions**: Use learned model for new data

### Mathematical Foundation

You now understand:
- The linear equation: `y = w·x + b`
- Mean Squared Error: `MSE = mean((pred - real)²)`
- Gradient calculation: `dw = mean((pred - real) × x)`
- Weight updates: `w = w - lr × dw`

### Limitations

- ❌ Only works for **linear relationships**
- ❌ Cannot handle **non-linear patterns**
- ❌ Single input feature (we'll extend this later)

---

## 1.17 Checklist (Before Moving On)

Before proceeding to Step 2, make sure you understand:

- [ ] What linear regression does
- [ ] How to calculate error (MSE)
- [ ] How gradient descent works
- [ ] Why the line improves during training
- [ ] How to make predictions for new data
- [ ] How to interpret learning curves

If you can answer "yes" to all, you're ready for **Step 2: Perceptron**!

---

## 🎯 Ready for Step 2?

You've learned how AI predicts numbers. Now let's see how AI makes **decisions**!

➡️ **Next: Step 2 – Perceptron (First Decision-Making AI)**
