# Step 4 — Multiple Neurons & Neural Network Layer (Matrix Thinking)

> **Goal:** Move from a single neuron to **many neurons working together** using matrices.  
> **Tools:** Python + NumPy + Matplotlib

---

## 4.1 Big Idea
So far, we used **one neuron**.

But real AI works like a **team of neurons**:
- Each neuron looks at the data differently
- Together they make better decisions

> **A Neural Network = Many neurons + Math + Repetition**

---

## 4.2 From One Neuron to Many

Single neuron:
z = x · w + b

Multiple neurons (layer):
Z = X · W + b

Where:
- **X** = input matrix (many samples)
- **W** = weight matrix (many neurons)
- **b** = bias vector

---

## 4.3 Dataset Example (Student Features)

Features:
- Math score
- Science score

Target:
- Pass (1) or Fail (0)

```python
import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [80, 70],
    [60, 65],
    [90, 95],
    [50, 45]
], dtype=float)

y = np.array([[1], [0], [1], [0]])
```

**Code Explanation:**
- `X = np.array([[...]])`: Creates a 2D matrix (4 students × 2 features)
  - Each row is one student: `[math_score, science_score]`
  - Shape: `(4, 2)` = 4 samples, 2 features per sample
  - `dtype=float`: Ensures decimal numbers for calculations
- `y = np.array([[1], [0], [1], [0]])`: Target labels (2D array)
  - Each row is one label: `[1]` = Pass, `[0]` = Fail
  - Shape: `(4, 1)` = 4 samples, 1 output per sample
  - Using 2D array `[[1], [0]]` instead of 1D `[1, 0]` for matrix compatibility

---

## 4.4 Understanding Shapes (VERY IMPORTANT)

```python
print("X shape:", X.shape)   # (samples, features)
```

Example:
- 4 students
- 2 features

X shape = (4, 2)

---

## 4.5 Weight Matrix (Many Neurons)

Let’s use **3 neurons** in one layer.

```python
W = np.random.randn(2, 3)  # 2 features → 3 neurons
b = np.zeros((1, 3))

print("W shape:", W.shape)
print("b shape:", b.shape)
```

**Code Explanation:**
- `W = np.random.randn(2, 3)`: Creates weight matrix
  - `np.random.randn(...)`: Random values from normal distribution (mean=0, std=1)
  - Shape `(2, 3)`: 2 input features → 3 neurons
  - Each column represents weights for one neuron
  - Column 1: weights connecting inputs to neuron 1
  - Column 2: weights connecting inputs to neuron 2
  - Column 3: weights connecting inputs to neuron 3
- `b = np.zeros((1, 3))`: Creates bias vector
  - `np.zeros(...)`: Creates array filled with zeros
  - Shape `(1, 3)`: One bias per neuron (3 biases total)
  - Starts at zero (will be learned during training)
- `print(...)`: Display shapes to verify dimensions
  - `W.shape` should be `(2, 3)`
  - `b.shape` should be `(1, 3)`

---

## 4.6 Forward Pass (Matrix Multiplication)

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

Z = X @ W + b
A = sigmoid(Z)

print("Z shape:", Z.shape)
print("A (outputs):\n", A)
```

**Code Explanation:**
- `def sigmoid(z):`: Activation function
  - Converts any number to value between 0 and 1
  - Used to introduce non-linearity (essential for neural networks!)
- `Z = X @ W + b`: Matrix multiplication (forward pass)
  - `X @ W`: Matrix multiply (4×2) @ (2×3) = (4×3)
    - Each row of X (one student) × weight matrix = scores for 3 neurons
    - Result: 4 students × 3 neuron scores
  - `+ b`: Broadcasting adds bias to each neuron
    - NumPy automatically adds (1×3) bias to each of 4 rows
  - `Z` shape: `(4, 3)` = 4 students, 3 neuron scores each
- `A = sigmoid(Z)`: Apply activation function
  - Converts scores to activations (probabilities)
  - Applied element-wise to entire matrix
  - `A` shape: `(4, 3)` = 4 students, 3 neuron activations
- **Key insight:** Each column in A = output of one neuron across all students

🧠 **Key idea:**  
Each column in **A** = output of one neuron.

---

## 4.7 Visualizing Neuron Outputs

```python
plt.figure(figsize=(7,4))

for i in range(A.shape[1]):
    plt.plot(A[:, i], label=f"Neuron {i+1}")

plt.title("Outputs of Neurons in One Layer")
plt.xlabel("Student Index")
plt.ylabel("Activation")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 4.8 Single Output Neuron (Combining the Layer)

Now we combine the neuron layer into **one output neuron**.

```python
W_out = np.random.randn(3, 1)
b_out = np.zeros((1, 1))

Z_out = A @ W_out + b_out
y_pred = sigmoid(Z_out)

print("Final output probabilities:\n", y_pred)
```

**Code Explanation:**
- `W_out = np.random.randn(3, 1)`: Output layer weights
  - Shape `(3, 1)`: 3 hidden neurons → 1 output neuron
  - Combines the 3 hidden neuron outputs into 1 final output
- `b_out = np.zeros((1, 1))`: Output layer bias
  - Shape `(1, 1)`: Single bias value for output neuron
- `Z_out = A @ W_out + b_out`: Calculate final scores
  - `A @ W_out`: (4×3) @ (3×1) = (4×1)
    - Each student's 3 neuron activations → 1 final score
  - `+ b_out`: Add output bias
  - `Z_out` shape: `(4, 1)` = 4 students, 1 score each
- `y_pred = sigmoid(Z_out)`: Convert to probabilities
  - Final predictions as probabilities (0 to 1)
  - `y_pred` shape: `(4, 1)`
- **Network flow:** Input (4×2) → Hidden (4×3) → Output (4×1)

---

## 4.9 Loss Function

```python
def binary_cross_entropy(y, y_pred):
    return -np.mean(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))

loss = binary_cross_entropy(y, y_pred)
print("Loss:", loss)
```

---

## 4.10 Training the Network (One Hidden Layer)

```python
lr = 0.1
losses = []

for epoch in range(1000):
    # Forward
    Z = X @ W + b
    A = sigmoid(Z)

    Z_out = A @ W_out + b_out
    y_pred = sigmoid(Z_out)

    # Loss
    loss = binary_cross_entropy(y, y_pred)
    losses.append(loss)

    # Backprop (simplified)
    dZ_out = y_pred - y
    dW_out = A.T @ dZ_out
    db_out = np.mean(dZ_out, axis=0)

    dA = dZ_out @ W_out.T
    dZ = dA * A * (1 - A)
    dW = X.T @ dZ
    db = np.mean(dZ, axis=0)

    # Update
    W_out -= lr * dW_out
    b_out -= lr * db_out
    W -= lr * dW
    b -= lr * db
```

---

## 4.11 Learning Curve

```python
plt.plot(losses)
plt.title("Loss During Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.show()
```

✅ Loss goes down → network is learning.

---

## 4.12 Final Predictions

```python
final_probs = sigmoid((sigmoid(X @ W + b)) @ W_out + b_out)
predictions = (final_probs >= 0.5).astype(int)

print("Final probabilities:\n", final_probs)
print("Predictions:\n", predictions)
```

---

## 4.13 Why This Matters

✅ First **real neural network**  
✅ Multiple neurons learn different patterns  
✅ Matrix math = speed + power  

❌ Still limited to simple patterns

---

## 4.14 Mini Exercises

### Exercise 1
Change number of neurons:
- Try 2 neurons
- Try 5 neurons

### Exercise 2
Change learning rate and observe training.

### Exercise 3 (Thinking)
Why is matrix multiplication faster than loops?

---

## 4.15 Checklist (Before Moving On)

Students should understand:
- What a layer is
- Why we use matrices
- How neurons work together
- Forward + backward pass idea

If YES → move to **Step 5: Hidden Layers & XOR**

---

## Next Step Preview
Now we know how to build a neural network layer.

Next, we will:
> Solve a problem that **single layers cannot**

➡️ **Step 5 – XOR & Hidden Layers**
