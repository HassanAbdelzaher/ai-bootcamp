# Project 2: Multi-Class Classifier

> **Build a neural network that classifies data into multiple categories**

**Difficulty**: ⭐⭐ Intermediate  
**Time**: 2-3 hours  
**Prerequisites**: Steps 0-5 (All foundational steps, including hidden layers)

---

## 🎯 Project Goals

Build a neural network that can classify data into multiple categories (not just 2). Choose one of these problems:

1. **Handwritten Digit Recognition** (0-9)
2. **Animal Classification** (Cat, Dog, Bird, etc.)
3. **Sentiment Analysis** (Positive, Neutral, Negative)

---

## 📋 Problem Options

### Option A: Handwritten Digit Recognition

**Task**: Classify handwritten digits (0-9) based on pixel values.

**Dataset**: Create synthetic 8×8 pixel images or use a simple dataset.

**Features**: 64 features (8×8 pixels), each pixel value 0-1.

**Classes**: 10 classes (digits 0-9).

**Requirements**:
- [ ] Create or load digit dataset
- [ ] Preprocess images (normalize pixel values)
- [ ] Build multi-layer neural network
- [ ] Train with multi-class cross-entropy loss
- [ ] Evaluate using accuracy and confusion matrix
- [ ] Visualize predictions

---

### Option B: Animal Classification

**Task**: Classify animals into categories based on features.

**Dataset**: Create synthetic data with features like:
- Size (small/medium/large)
- Number of legs
- Has fur (0/1)
- Can fly (0/1)
- Habitat type

**Classes**: 3-5 animal categories.

**Requirements**:
- [ ] Create feature-based dataset
- [ ] Encode categorical features
- [ ] Build and train neural network
- [ ] Evaluate classification performance
- [ ] Interpret learned features

---

### Option C: Sentiment Analysis

**Task**: Classify text sentiment into 3 categories.

**Dataset**: Create synthetic text data with:
- Word counts for positive/negative words
- Text length
- Exclamation marks
- Capital letters

**Classes**: Positive, Neutral, Negative.

**Requirements**:
- [ ] Create text feature dataset
- [ ] Extract features from text
- [ ] Train multi-class classifier
- [ ] Evaluate on test set
- [ ] Show example classifications

---

## 🚀 Getting Started

### Step 1: Choose Your Problem

Pick one of the three options above based on your interest.

### Step 2: Create Dataset

Generate or load your dataset. See `create_dataset.py` for examples.

### Step 3: Build Network

Implement a multi-layer neural network:
- Input layer: Number of features
- Hidden layer(s): 8-16 neurons
- Output layer: Number of classes

### Step 4: Train Model

Use softmax activation for output layer and categorical cross-entropy loss.

### Step 5: Evaluate

Calculate accuracy, create confusion matrix, visualize results.

---

## 📊 Expected Deliverables

1. **Code**:
   - `main.py` - Main training script
   - `model.py` - Neural network implementation
   - `utils.py` - Helper functions

2. **Results**:
   - Training/validation accuracy
   - Confusion matrix
   - Sample predictions

3. **Documentation**:
   - Approach explanation
   - Architecture diagram
   - Performance analysis

---

## 💡 Extension Ideas

- Add more hidden layers
- Try different activation functions
- Implement dropout for regularization
- Add batch normalization
- Visualize learned features
- Create a simple web interface

---

## 📚 Key Concepts

- **Multi-class classification**: More than 2 classes
- **Softmax**: Converts scores to probabilities that sum to 1
- **Categorical Cross-Entropy**: Loss function for multi-class
- **One-hot encoding**: Represent classes as vectors
- **Confusion Matrix**: Shows per-class performance

---

## ✅ Success Criteria

- Model achieves >70% accuracy
- All classes are reasonably predicted
- Training converges smoothly
- Code is well-organized and documented

---

**Ready to build your first multi-class classifier?** 🚀
