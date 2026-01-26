# Step 8b — Image Classifiers

> **Goal:** Build and improve image classification models.  
> **Tools:** Python + PyTorch

---

## 8b.1 Classification Pipeline

1. Load and preprocess images
2. Build CNN architecture
3. Train on labeled data
4. Evaluate on test set
5. Make predictions on new images

---

## 8b.2 Key Improvements

### Batch Normalization
- Normalizes activations
- Faster training
- Better convergence

### Dropout
- Prevents overfitting
- Randomly sets neurons to zero
- Forces network to be robust

### Deeper Architecture
- More layers = more features
- Hierarchical feature learning

---

## 8b.3 Classification Metrics

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True + False positives)
- **Recall**: True positives / (True + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Shows per-class performance

---

## 8b.4 Tips for Better Classification

✅ **Data:**
- More training data
- Balanced classes
- Data augmentation

✅ **Architecture:**
- Deeper networks
- Batch normalization
- Residual connections

✅ **Training:**
- Learning rate scheduling
- Early stopping
- Ensemble methods
