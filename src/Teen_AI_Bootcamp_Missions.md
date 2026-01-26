# 🤖 Teen AI Bootcamp — Missions Book
> **Hands-on missions for every step (0 → 6)**  
> Designed for teens • Learn by doing • No magic

---

## 🧩 Mission 0 — Math Foundations (Think Like AI)

### 🎯 Objective
Understand how AI turns numbers into decisions.

### 🧠 What You Use
- Vectors
- Weights
- Dot product
- Bias

### 🛠 Tasks
1. Create a vector for a student: `[math, science, english]`
2. Create weights for each subject.
3. Calculate the dot product.
4. Add a bias.
5. Print the final score.

### 🧪 Starter Code
```python
import numpy as np

student = np.array([80, 70, 75])
weights = np.array([0.5, 0.3, 0.2])
bias = -10

score = np.dot(student, weights) + bias
print("AI Score:", score)
```

### 🏁 Mission Complete When
- You can explain what each number means.

---

## 📈 Mission 1 — Linear Regression (Predict the Future)

### 🎯 Objective
Teach AI to predict a number.

### 🧠 What You Use
- y = wx + b
- Error (MSE)
- Gradient Descent

### 🛠 Tasks
1. Create a dataset (hours → score).
2. Initialize w and b.
3. Train using gradient descent.
4. Plot the prediction line.

### 🧪 Starter Code
```python
import numpy as np

X = np.array([1,2,3,4], dtype=float)
y = np.array([50,60,70,80], dtype=float)

w, b = 0.0, 0.0
lr = 0.01
```

### 🏁 Mission Complete When
- Error decreases over time.

---

## ⚡ Mission 2 — Perceptron (YES or NO AI)

### 🎯 Objective
Build an AI that makes decisions.

### 🧠 What You Use
- Step function
- Decision boundary

### 🛠 Tasks
1. Create a step function.
2. Predict pass/fail from study hours.
3. Train the perceptron.
4. Plot the decision boundary.

### 🧪 Starter Code
```python
def step(z):
    return 1 if z >= 0 else 0
```

### 🏁 Mission Complete When
- Predictions match the dataset.

---

## 🎯 Mission 3 — Logistic Regression (AI with Confidence)

### 🎯 Objective
Make AI output probabilities.

### 🧠 What You Use
- Sigmoid
- Binary Cross-Entropy
- Threshold

### 🛠 Tasks
1. Implement sigmoid.
2. Train logistic regression.
3. Plot probability curve.
4. Convert probability to decision.

### 🧪 Starter Code
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

### 🏁 Mission Complete When
- AI outputs probabilities (0 → 1).

---

## 🧠 Mission 4 — Neural Network Layer (Team of Neurons)

### 🎯 Objective
Build your first real neural network layer.

### 🧠 What You Use
- Matrices
- Multiple neurons
- Forward pass

### 🛠 Tasks
1. Create a weight matrix.
2. Compute X @ W + b.
3. Apply activation.
4. Add output neuron.

### 🧪 Starter Code
```python
W = np.random.randn(2, 3)
b = np.zeros((1,3))
```

### 🏁 Mission Complete When
- Network produces valid outputs.

---

## 🔥 Mission 5 — XOR Challenge (Beat the Impossible)

### 🎯 Objective
Solve a problem a single neuron cannot.

### 🧠 What You Use
- Hidden layer
- Backpropagation
- Non-linearity

### 🛠 Tasks
1. Plot XOR data.
2. Try single-layer (fail).
3. Add hidden layer.
4. Train until success.

### 🧪 Starter Code
```python
W1 = np.random.randn(2,4)
W2 = np.random.randn(4,1)
```

### 🏁 Mission Complete When
- XOR predictions are correct.

---

## 🚀 Mission 6 — PyTorch Engineer (Real AI)

### 🎯 Objective
Build AI using a professional framework.

### 🧠 What You Use
- PyTorch tensors
- nn.Linear
- Optimizer
- Loss function

### 🛠 Tasks
1. Build model with `nn.Sequential`.
2. Train using PyTorch loop.
3. Plot loss.
4. Make predictions.

### 🧪 Starter Code
```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2,4),
    nn.ReLU(),
    nn.Linear(4,1),
    nn.Sigmoid()
)
```

### 🏁 Mission Complete When
- Model solves XOR using PyTorch.

---

## 🏆 Final Boss Mission — Build Your Own AI

### 🎯 Choose One
- Pass/Fail Predictor
- Grade Classifier
- Student Performance AI

### 🧠 Requirements
- Use neural network
- Show training curve
- Explain result

### 🎉 Graduation
You are now a **Junior AI Engineer** 🧠🤖
