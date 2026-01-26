# Project 1: Simple Predictor

> **Build your first practical AI application using linear and logistic regression**

**Difficulty**: ⭐ Beginner  
**Time**: 1-2 hours  
**Prerequisites**: Steps 0-3 (Math Foundations, Linear Regression, Logistic Regression)

---

## 🎯 Project Goals

Build two prediction systems:
1. **House Price Predictor** (Linear Regression)
2. **Email Spam Classifier** (Logistic Regression)

---

## 📋 Problem 1: House Price Prediction

### Task
Predict house prices based on features like size, number of bedrooms, location, etc.

### Dataset
Create synthetic data or use a simple dataset with:
- House size (sq ft)
- Number of bedrooms
- Age of house
- Price (target)

### Requirements
- [ ] Load and explore the data
- [ ] Preprocess features (normalize if needed)
- [ ] Train a linear regression model
- [ ] Evaluate using MSE and R²
- [ ] Make predictions for new houses
- [ ] Visualize predictions vs actual prices

### Success Criteria
- Model achieves reasonable accuracy
- Predictions make intuitive sense
- Visualizations are clear

---

## 📋 Problem 2: Email Spam Classification

### Task
Classify emails as spam (1) or not spam (0) based on features.

### Dataset
Create synthetic data with features like:
- Number of exclamation marks
- Number of ALL CAPS words
- Contains "free" or "click"
- Email length
- Label (spam/not spam)

### Requirements
- [ ] Load and explore the data
- [ ] Preprocess features
- [ ] Train a logistic regression model
- [ ] Evaluate using accuracy, precision, recall
- [ ] Make predictions for new emails
- [ ] Visualize decision boundary (if 2D)

### Success Criteria
- Model achieves >80% accuracy
- Can explain why emails are classified as spam
- Confusion matrix shows good performance

---

## 🚀 Getting Started

### Step 1: Create Project Structure

```bash
cd projects/project_1_simple_predictor
mkdir -p data results
```

### Step 2: Implement House Price Predictor

See `house_price_predictor.py` for starter code.

### Step 3: Implement Spam Classifier

See `spam_classifier.py` for starter code.

---

## 📊 Expected Deliverables

1. **Code Files**:
   - `house_price_predictor.py`
   - `spam_classifier.py`
   - `utils.py` (helper functions)

2. **Results**:
   - Model performance metrics
   - Visualization plots
   - Predictions on test data

3. **Documentation**:
   - Brief report explaining your approach
   - Model performance summary
   - Challenges faced and solutions

---

## 💡 Extension Ideas

- Add more features to improve accuracy
- Try different learning rates
- Implement feature engineering
- Add data validation
- Create a simple web interface

---

## 📚 Learning Resources

- Review Step 1 (Linear Regression) documentation
- Review Step 3 (Logistic Regression) documentation
- NumPy documentation for data manipulation
- Matplotlib documentation for visualizations

---

## ✅ Checklist

Before submitting, ensure:
- [ ] Code is well-commented
- [ ] Models train successfully
- [ ] Results are visualized
- [ ] Performance metrics are calculated
- [ ] Documentation is complete

---

**Good luck! This is your first real AI project!** 🎉
