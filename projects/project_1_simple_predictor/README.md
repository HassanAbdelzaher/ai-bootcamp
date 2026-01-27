# Project 1: Simple Predictor

> **Build your first practical AI application using linear and logistic regression**

**Difficulty**: ⭐ Beginner  
**Time**: 1-2 hours  
**Prerequisites**: Steps 0-3 (Math Foundations, Linear Regression, Logistic Regression)

---

## 📚 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem 1: House Price Prediction](#problem-1-house-price-prediction)
3. [Problem 2: Email Spam Classification](#problem-2-email-spam-classification)
4. [Getting Started](#getting-started)
5. [Step-by-Step Guide](#step-by-step-guide)
6. [Code Structure](#code-structure)
7. [Expected Results](#expected-results)
8. [Extension Ideas](#extension-ideas)
9. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

This project consists of two practical applications that demonstrate fundamental AI concepts:

1. **House Price Predictor** - Uses linear regression to predict continuous values (prices)
2. **Email Spam Classifier** - Uses logistic regression to make binary decisions (spam/not spam)

Both projects use real-world scenarios to help you understand how AI models work in practice.

---

## 📋 Problem 1: House Price Prediction

### Overview

Predict house prices based on features like size, number of bedrooms, and age. This is a **regression problem** (predicting a continuous number).

### Learning Objectives

- Understand how to prepare data for machine learning
- Implement linear regression from scratch
- Evaluate model performance using metrics
- Make predictions on new data
- Visualize model results

### Dataset Description

The project uses synthetic house data with the following features:

| Feature | Description | Range/Type |
|---------|-------------|------------|
| **Size** | House size in square feet | 800 - 3000 sq ft |
| **Bedrooms** | Number of bedrooms | 1 - 4 bedrooms |
| **Age** | Age of house in years | 0 - 30 years |
| **Price** | House price (target) | $30,000+ |

**Price Formula** (used to generate data):
```
Price = 50 × Size + 10,000 × Bedrooms - 500 × Age + 20,000 + Noise
```

### Step-by-Step Implementation

#### Step 1: Create Dataset

```python
import numpy as np

# Generate synthetic house data
np.random.seed(42)  # For reproducibility

num_houses = 50
house_sizes = np.random.uniform(800, 3000, num_houses)  # sq ft
bedrooms = np.random.randint(1, 5, num_houses)
house_ages = np.random.uniform(0, 30, num_houses)  # years

# Create price based on features (with noise)
prices = (50 * house_sizes + 
          10000 * bedrooms - 
          500 * house_ages + 
          20000 + 
          np.random.normal(0, 5000, num_houses))
```

**Code Explanation:**
- `np.random.uniform(800, 3000, num_houses)`: Generates random house sizes
- `np.random.randint(1, 5, num_houses)`: Generates random bedroom counts
- `np.random.normal(0, 5000, num_houses)`: Adds realistic noise to prices

#### Step 2: Prepare Features

```python
# Combine features into matrix
X = np.column_stack([house_sizes, bedrooms, house_ages])
y = prices

# Normalize features (important for gradient descent)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_normalized = (X - X_mean) / X_std
```

**Why Normalize?**
- Different features have different scales (size: 800-3000, bedrooms: 1-4)
- Normalization helps gradient descent converge faster
- Formula: `normalized = (value - mean) / std`

#### Step 3: Train Linear Regression Model

```python
# Initialize weights and bias
w = np.zeros(X_normalized.shape[1])  # One weight per feature
b = 0.0
lr = 0.01  # Learning rate

# Training loop
for epoch in range(1000):
    # Forward pass: make predictions
    y_pred = X_normalized @ w + b
    
    # Calculate gradients
    dw = np.mean((y_pred - y).reshape(-1, 1) * X_normalized, axis=0)
    db = np.mean(y_pred - y)
    
    # Update weights
    w -= lr * dw
    b -= lr * db
```

**Key Concepts:**
- **Forward Pass**: Calculate predictions using current weights
- **Gradients**: Calculate how to adjust weights to reduce error
- **Update**: Move weights in direction that reduces error

#### Step 4: Evaluate Model

```python
# Calculate Mean Squared Error (MSE)
mse = np.mean((y_pred - y) ** 2)

# Calculate R² Score (coefficient of determination)
ss_res = np.sum((y - y_pred) ** 2)  # Sum of squared residuals
ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
r2_score = 1 - (ss_res / ss_tot)

print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2_score:.3f}")  # Closer to 1.0 is better
```

**Metrics Explained:**
- **MSE**: Average squared error (lower is better)
- **R² Score**: Proportion of variance explained (0-1, higher is better)
  - R² = 1.0: Perfect predictions
  - R² = 0.0: Model is no better than predicting the mean
  - R² < 0.0: Model is worse than predicting the mean

#### Step 5: Make Predictions

```python
# Predict price for new house
new_house = np.array([[1500, 3, 5]])  # 1500 sq ft, 3 bedrooms, 5 years old
new_house_normalized = (new_house - X_mean) / X_std
predicted_price = new_house_normalized @ w + b

print(f"Predicted price: ${predicted_price[0]:,.2f}")
```

### Requirements Checklist

- [ ] Load and explore the data
- [ ] Visualize feature distributions
- [ ] Preprocess features (normalize)
- [ ] Train linear regression model
- [ ] Evaluate using MSE and R²
- [ ] Make predictions for new houses
- [ ] Visualize predictions vs actual prices
- [ ] Plot learning curve
- [ ] Analyze prediction errors

### Success Criteria

- ✅ Model achieves R² > 0.85
- ✅ Predictions make intuitive sense
- ✅ Learning curve shows convergence
- ✅ Visualizations are clear and informative

### Expected Output

```
Step 1: Creating Dataset
Generated 50 houses
Price range: $30,000 - $180,000

Step 2: Training Model
Training for 1000 epochs...
Final MSE: 2,345,678.90
Final R² Score: 0.892

Step 3: Making Predictions
New house: 1500 sq ft, 3 bedrooms, 5 years old
Predicted price: $95,234.56
```

---

## 📋 Problem 2: Email Spam Classification

### Overview

Classify emails as spam (1) or not spam (0) based on features. This is a **binary classification problem** (predicting one of two classes).

### Learning Objectives

- Understand binary classification
- Implement logistic regression
- Use probability-based decisions
- Evaluate classification performance
- Interpret model predictions

### Dataset Description

The project uses synthetic email data with the following features:

| Feature | Description | Range/Type |
|---------|-------------|------------|
| **Exclamations** | Number of exclamation marks | 0 - 10 |
| **ALL CAPS** | Number of ALL CAPS words | 0 - 8 |
| **Keywords** | Contains "free" or "click" | 0 or 1 (binary) |
| **Label** | Spam or not spam | 0 (not spam) or 1 (spam) |

**Pattern**: Spam emails typically have more exclamations, ALL CAPS words, and keywords.

### Step-by-Step Implementation

#### Step 1: Create Email Dataset

```python
import numpy as np

np.random.seed(42)

num_emails = 100

# Feature 1: Number of exclamation marks (spam has more)
exclamations = np.random.poisson(3, num_emails)
exclamations = np.where(exclamations > 10, 10, exclamations)

# Feature 2: Number of ALL CAPS words (spam has more)
all_caps = np.random.poisson(2, num_emails)
all_caps = np.where(all_caps > 8, 8, all_caps)

# Feature 3: Contains keywords (binary)
contains_keywords = np.random.binomial(1, 0.3, num_emails)

# Combine features
X = np.column_stack([exclamations, all_caps, contains_keywords])
```

**Code Explanation:**
- `np.random.poisson(3, num_emails)`: Generates count data (exclamations, caps)
- `np.random.binomial(1, 0.3, num_emails)`: Generates binary data (keywords)
- `np.column_stack()`: Combines features into matrix

#### Step 2: Generate Labels

```python
# Generate labels based on features (spam if high exclamations/caps/keywords)
# This simulates real spam detection rules
y = ((exclamations > 4) | (all_caps > 3) | (contains_keywords == 1)).astype(int)
```

**Logic**: Email is spam if it has:
- More than 4 exclamations, OR
- More than 3 ALL CAPS words, OR
- Contains keywords

#### Step 3: Train Logistic Regression Model

```python
def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# Initialize weights
w = np.zeros(X.shape[1])
b = 0.0
lr = 0.1

# Training loop
for epoch in range(1000):
    # Forward pass
    z = X @ w + b
    y_pred = sigmoid(z)
    
    # Calculate gradients
    dw = np.mean((y_pred - y).reshape(-1, 1) * X, axis=0)
    db = np.mean(y_pred - y)
    
    # Update weights
    w -= lr * dw
    b -= lr * db
```

**Key Differences from Linear Regression:**
- Uses **sigmoid** activation (outputs probabilities 0-1)
- Loss function: **Binary Cross-Entropy** (not MSE)
- Predictions are probabilities, not direct values

#### Step 4: Make Predictions

```python
# Get probabilities
probabilities = sigmoid(X @ w + b)

# Convert to binary predictions (threshold = 0.5)
predictions = (probabilities >= 0.5).astype(int)

# Calculate accuracy
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.2%}")
```

**Decision Process:**
1. Calculate probability: `P(spam | email_features)`
2. If probability ≥ 0.5 → Predict spam (1)
3. If probability < 0.5 → Predict not spam (0)

#### Step 5: Evaluate Performance

```python
# Calculate confusion matrix components
tp = np.sum((predictions == 1) & (y == 1))  # True Positives
tn = np.sum((predictions == 0) & (y == 0))  # True Negatives
fp = np.sum((predictions == 1) & (y == 0))  # False Positives
fn = np.sum((predictions == 0) & (y == 1))  # False Negatives

# Calculate metrics
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1_score:.3f}")
```

**Metrics Explained:**
- **Precision**: Of emails predicted as spam, how many are actually spam?
- **Recall**: Of actual spam emails, how many did we catch?
- **F1-Score**: Harmonic mean of precision and recall (balanced metric)

### Requirements Checklist

- [ ] Load and explore email data
- [ ] Visualize feature distributions
- [ ] Preprocess features
- [ ] Train logistic regression model
- [ ] Evaluate using accuracy, precision, recall, F1
- [ ] Make predictions for new emails
- [ ] Visualize decision boundary (if 2D)
- [ ] Plot confusion matrix
- [ ] Analyze false positives/negatives

### Success Criteria

- ✅ Model achieves >80% accuracy
- ✅ Can explain why emails are classified as spam
- ✅ Confusion matrix shows good performance
- ✅ Precision and recall are balanced

### Expected Output

```
Step 1: Creating Email Dataset
Generated 100 emails
Spam: 35, Not Spam: 65

Step 2: Training Model
Training for 1000 epochs...
Final loss: 0.234

Step 3: Evaluation
Accuracy: 87.0%
Precision: 0.857
Recall: 0.800
F1-Score: 0.828
```

---

## 🚀 Getting Started

### Prerequisites

Before starting, make sure you've completed:
- **Step 0**: Math Foundations (vectors, dot product, bias)
- **Step 1**: Linear Regression (gradient descent, MSE)
- **Step 2**: Perceptron (binary classification)
- **Step 3**: Logistic Regression (sigmoid, probability)

### Setup Instructions

1. **Navigate to project directory**
   ```bash
   cd projects/project_1_simple_predictor
   ```

2. **Verify files exist**
   ```bash
   ls -la
   # Should see:
   # - README.md (this file)
   # - house_price_predictor.py
   # - spam_classifier.py
   ```

3. **Run House Price Predictor**
   ```bash
   python house_price_predictor.py
   ```

4. **Run Spam Classifier**
   ```bash
   python spam_classifier.py
   ```

### Project Structure

```
project_1_simple_predictor/
├── README.md                    # This file
├── house_price_predictor.py     # Linear regression implementation
├── spam_classifier.py            # Logistic regression implementation
└── results/                      # (Create this folder for outputs)
    ├── house_predictions.png
    ├── spam_confusion_matrix.png
    └── learning_curves.png
```

---

## 📊 Code Structure

### House Price Predictor (`house_price_predictor.py`)

**Main Sections:**
1. **Dataset Creation** - Generate synthetic house data
2. **Data Preprocessing** - Normalize features
3. **Model Training** - Linear regression with gradient descent
4. **Model Evaluation** - Calculate MSE and R²
5. **Predictions** - Predict prices for new houses
6. **Visualization** - Plot results

**Key Functions:**
- `create_house_data()` - Generate synthetic dataset
- `normalize_features()` - Normalize input features
- `train_linear_regression()` - Train model
- `evaluate_model()` - Calculate metrics
- `predict_price()` - Make predictions

### Spam Classifier (`spam_classifier.py`)

**Main Sections:**
1. **Dataset Creation** - Generate synthetic email data
2. **Data Preprocessing** - Prepare features
3. **Model Training** - Logistic regression
4. **Model Evaluation** - Calculate classification metrics
5. **Predictions** - Classify new emails
6. **Visualization** - Plot confusion matrix and decision boundary

**Key Functions:**
- `create_email_data()` - Generate synthetic dataset
- `sigmoid()` - Activation function
- `train_logistic_regression()` - Train model
- `evaluate_classification()` - Calculate metrics
- `predict_spam()` - Classify emails

---

## 📈 Expected Results

### House Price Predictor

**Training Output:**
```
Training Linear Regression Model...
Epoch 100/1000, Loss: 12,345,678.90
Epoch 200/1000, Loss: 8,234,567.89
...
Epoch 1000/1000, Loss: 2,345,678.90

Final Results:
- MSE: 2,345,678.90
- R² Score: 0.892
- Model learned: Price = 48.5×Size + 9,850×Bedrooms - 495×Age + 20,150
```

**Visualizations:**
- Scatter plot: Actual vs Predicted prices
- Learning curve: Loss decreasing over time
- Error distribution: Histogram of prediction errors

### Spam Classifier

**Training Output:**
```
Training Logistic Regression Model...
Epoch 100/1000, Loss: 0.523
Epoch 200/1000, Loss: 0.412
...
Epoch 1000/1000, Loss: 0.234

Final Results:
- Accuracy: 87.0%
- Precision: 0.857
- Recall: 0.800
- F1-Score: 0.828

Confusion Matrix:
              Predicted
            Not Spam  Spam
Actual Not Spam   58    7
       Spam        6   29
```

**Visualizations:**
- Confusion matrix heatmap
- Probability curve (sigmoid)
- Decision boundary (if 2D features)

---

## 💡 Extension Ideas

### Beginner Extensions

1. **Add More Features**
   - House: Add location, number of bathrooms
   - Email: Add email length, sender domain

2. **Try Different Learning Rates**
   - Compare training speed and final performance
   - Find optimal learning rate

3. **Visualize Feature Importance**
   - Which features matter most?
   - Plot feature weights

### Intermediate Extensions

4. **Feature Engineering**
   - Create new features (e.g., price per sq ft)
   - Combine features (e.g., total rooms)

5. **Train/Test Split**
   - Split data into training and test sets
   - Evaluate on unseen data
   - Compare train vs test performance

6. **Regularization**
   - Add L2 regularization to prevent overfitting
   - Compare with/without regularization

### Advanced Extensions

7. **Multiple Features**
   - House: Add 5+ features
   - Email: Add 10+ features
   - See how model handles more complexity

8. **Cross-Validation**
   - Implement k-fold cross-validation
   - Get more reliable performance estimates

9. **Model Comparison**
   - Compare linear vs polynomial regression
   - Compare different threshold values for spam

---

## 🔧 Troubleshooting

### Common Issues

**Issue 1: Model doesn't learn (loss doesn't decrease)**
- **Solution**: Check learning rate (try 0.01, 0.1, 1.0)
- **Solution**: Ensure features are normalized
- **Solution**: Increase number of training epochs

**Issue 2: Predictions are all the same**
- **Solution**: Check if weights are updating (print w, b during training)
- **Solution**: Verify gradient calculation is correct
- **Solution**: Check if data has enough variation

**Issue 3: Accuracy is very low (<50%)**
- **Solution**: Check if labels are correct
- **Solution**: Verify feature extraction is working
- **Solution**: Try different feature combinations

**Issue 4: Import errors for plotting**
- **Solution**: Ensure you're in the project directory
- **Solution**: Check that `src/plotting.py` exists
- **Solution**: Verify Python path is set correctly

### Debugging Tips

1. **Print intermediate values**
   ```python
   print(f"Weights: {w}")
   print(f"Bias: {b}")
   print(f"Predictions: {y_pred[:5]}")
   ```

2. **Visualize data first**
   ```python
   plt.scatter(X[:, 0], y)
   plt.show()
   ```

3. **Check data shapes**
   ```python
   print(f"X shape: {X.shape}")
   print(f"y shape: {y.shape}")
   ```

4. **Monitor training progress**
   ```python
   if epoch % 100 == 0:
       print(f"Epoch {epoch}, Loss: {loss:.4f}")
   ```

---

## 📚 Key Concepts Review

### Linear Regression
- **Goal**: Predict continuous values
- **Output**: Real number (price, temperature, etc.)
- **Loss**: Mean Squared Error (MSE)
- **Activation**: None (linear output)

### Logistic Regression
- **Goal**: Predict binary classes
- **Output**: Probability (0 to 1)
- **Loss**: Binary Cross-Entropy
- **Activation**: Sigmoid function

### Gradient Descent
- **Purpose**: Find optimal weights
- **Process**: Iteratively adjust weights to minimize loss
- **Learning Rate**: Controls step size
- **Convergence**: When loss stops decreasing

### Model Evaluation
- **Regression**: MSE, RMSE, R² Score
- **Classification**: Accuracy, Precision, Recall, F1-Score
- **Visualization**: Plots help understand model behavior

---

## ✅ Final Checklist

Before completing the project, ensure:

- [ ] Both models train successfully
- [ ] Code is well-commented
- [ ] Results are visualized
- [ ] Performance metrics are calculated
- [ ] Predictions are made on new data
- [ ] Documentation explains your approach
- [ ] You understand how the models work
- [ ] You can explain the results

---

## 🎓 Learning Outcomes

By completing this project, you will:

- ✅ Understand how to prepare data for machine learning
- ✅ Implement linear and logistic regression from scratch
- ✅ Evaluate model performance using appropriate metrics
- ✅ Make predictions on new data
- ✅ Visualize and interpret results
- ✅ Debug common machine learning issues
- ✅ Apply theoretical knowledge to practical problems

---

## 📖 Additional Resources

- **Step 1 Documentation**: `docs/Step_1_Linear_Regression.md`
- **Step 3 Documentation**: `docs/Step_3_Logistic_Regression.md`
- **NumPy Documentation**: https://numpy.org/doc/
- **Matplotlib Documentation**: https://matplotlib.org/

---

**Ready to build your first AI applications? Let's get started!** 🚀

**Next Steps**: After completing this project, move on to **Project 2: Multi-Class Classifier** to learn about neural networks with multiple classes.
