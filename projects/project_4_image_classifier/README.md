# Project 4: Image Classifier

> **Build a CNN-based image classification system**

**Difficulty**: ⭐⭐⭐ Advanced  
**Time**: 4-5 hours  
**Prerequisites**: Steps 0-8 (Including CNNs, especially Step 8a: Real Datasets, Step 8c: Transfer Learning)

---

## 📚 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem: Multi-Class Image Classification](#problem-multi-class-image-classification)
3. [Key Concepts](#key-concepts)
4. [Step-by-Step Guide](#step-by-step-guide)
5. [Expected Results](#expected-results)
6. [Extension Ideas](#extension-ideas)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

This project teaches you to build **Convolutional Neural Networks (CNNs)** for image classification. You'll learn to:

- Design CNN architectures
- Train models on image data
- Apply transfer learning
- Visualize learned features

### Why Image Classification?

- **Real-world applications**: Medical imaging, autonomous vehicles, security
- **Foundation for vision**: Prepares you for object detection, segmentation
- **Transferable skills**: CNN concepts apply to many vision tasks

---

## 📋 Problem: Multi-Class Image Classification

### Task

Build a CNN that classifies images into multiple categories (e.g., animals, objects, scenes).

### Learning Objectives

- Understand CNN architecture
- Preprocess image data
- Design convolutional layers
- Apply transfer learning
- Evaluate classification performance

### Dataset Description

Create or use image dataset with multiple classes:

| Class | Examples | Images |
|-------|----------|--------|
| **Cats** | Cat images | 100 |
| **Dogs** | Dog images | 100 |
| **Birds** | Bird images | 100 |
| **Cars** | Car images | 100 |

**Image Format**:
- Size: 64×64 or 128×128 pixels
- Channels: RGB (3 channels) or Grayscale (1 channel)
- Format: NumPy arrays or image files

---

## 🧠 Key Concepts

### 1. CNN Architecture

**Typical CNN Structure**:
```
Input Image (H×W×C)
  ↓
Convolutional Layers (feature extraction)
  ↓
Pooling Layers (downsampling)
  ↓
Fully Connected Layers (classification)
  ↓
Output (class probabilities)
```

**Key Components**:
- **Convolution**: Detect features (edges, shapes, patterns)
- **Pooling**: Reduce size, increase robustness
- **Activation**: ReLU for non-linearity
- **Dropout**: Prevent overfitting

### 2. Convolution Operation

**Purpose**: Detect local patterns in images

**Example**:
```python
# 3×3 convolution filter (edge detector)
filter = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
])

# Apply to image
convolved = convolve(image, filter)
```

**Parameters**:
- **Kernel size**: 3×3, 5×5 (filter size)
- **Stride**: How much filter moves (1, 2)
- **Padding**: Add zeros around image

### 3. Transfer Learning

**Concept**: Use pre-trained model, adapt to your task

**Process**:
1. Load pre-trained model (trained on ImageNet)
2. Freeze early layers (keep learned features)
3. Train only final layers on your data
4. Fine-tune if needed

**Benefits**:
- Faster training
- Better performance
- Less data needed

---

## 🚀 Step-by-Step Guide

### Step 1: Create/Load Image Dataset

```python
import numpy as np
from PIL import Image
import os

def load_images_from_folder(folder_path, img_size=64):
    """Load images from folder"""
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(['cats', 'dogs', 'birds', 'cars']):
        class_path = os.path.join(folder_path, class_name)
        
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            
            # Load and resize image
            img = Image.open(img_path).convert('RGB')
            img = img.resize((img_size, img_size))
            
            # Convert to array
            img_array = np.array(img) / 255.0  # Normalize to [0, 1]
            img_array = img_array.transpose(2, 0, 1)  # (H, W, C) → (C, H, W)
            
            images.append(img_array)
            labels.append(class_idx)
    
    return np.array(images), np.array(labels)

# Load dataset
X, y = load_images_from_folder('data/images', img_size=64)
print(f"Loaded {len(X)} images")
print(f"Image shape: {X[0].shape}")  # (3, 64, 64)
print(f"Classes: {len(np.unique(y))}")
```

### Step 2: Preprocess Data

```python
from sklearn.model_selection import train_test_split

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
import torch
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

print(f"Training: {len(X_train)} images")
print(f"Test: {len(X_test)} images")
```

### Step 3: Build CNN from Scratch

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 3→32 channels
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 32→64 channels
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # 64→128 channels
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)  # Halves size
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # After 3 pools: 64→32→16→8
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # x shape: (batch, 3, 64, 64)
        
        # Conv block 1
        x = F.relu(self.conv1(x))  # (batch, 32, 64, 64)
        x = self.pool(x)  # (batch, 32, 32, 32)
        
        # Conv block 2
        x = F.relu(self.conv2(x))  # (batch, 64, 32, 32)
        x = self.pool(x)  # (batch, 64, 16, 16)
        
        # Conv block 3
        x = F.relu(self.conv3(x))  # (batch, 128, 16, 16)
        x = self.pool(x)  # (batch, 128, 8, 8)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (batch, 128*8*8)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
```

**Architecture Explanation**:
- **Conv layers**: Extract features (edges → shapes → objects)
- **Pool layers**: Reduce size, increase receptive field
- **FC layers**: Classify based on features
- **Dropout**: Regularization to prevent overfitting

### Step 4: Train Model

```python
model = SimpleCNN(num_classes=4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
batch_size = 32

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    
    # Train in batches
    for i in range(0, len(X_train_tensor), batch_size):
        batch_X = X_train_tensor[i:i+batch_size]
        batch_y = y_train_tensor[i:i+batch_size]
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(X_train_tensor)*batch_size:.4f}")
```

### Step 5: Evaluate Model

```python
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predictions = torch.max(outputs, 1)
    accuracy = (predictions == y_test_tensor).float().mean()
    
    print(f"Test Accuracy: {accuracy:.2%}")

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_tensor.numpy(), predictions.numpy())
print("\nConfusion Matrix:")
print(cm)
```

### Step 6: Transfer Learning (Advanced)

```python
import torchvision.models as models

# Load pre-trained ResNet
pretrained_model = models.resnet18(pretrained=True)

# Freeze early layers
for param in pretrained_model.parameters():
    param.requires_grad = False

# Replace final layer
pretrained_model.fc = nn.Linear(512, 4)  # 4 classes

# Train only final layer
optimizer = torch.optim.Adam(pretrained_model.fc.parameters(), lr=0.001)

# Training (same as before, but faster!)
```

---

## 📊 Expected Results

### Training Output

```
Building CNN from Scratch...
Epoch 10/50, Loss: 1.2345
Epoch 20/50, Loss: 0.8765
Epoch 30/50, Loss: 0.5432
Epoch 40/50, Loss: 0.3210
Epoch 50/50, Loss: 0.1987

Evaluation:
Test Accuracy: 87.5%

Confusion Matrix:
        Cats  Dogs  Birds  Cars
Cats     18    1     0     1
Dogs      2   17     1     0
Birds     0    1    19     0
Cars      1    0     0    19
```

### Transfer Learning Output

```
Using Pre-trained ResNet...
Training final layer only...
Epoch 10/20, Loss: 0.5432
Epoch 20/20, Loss: 0.1234

Test Accuracy: 92.5%
(Transfer learning achieved better results with less training!)
```

---

## 💡 Extension Ideas

### Beginner Extensions

1. **Try Different Architectures**
   - More/fewer convolutional layers
   - Different filter sizes
   - Compare performance

2. **Experiment with Hyperparameters**
   - Learning rates: 0.0001, 0.001, 0.01
   - Batch sizes: 16, 32, 64
   - Dropout rates: 0.3, 0.5, 0.7

3. **Data Augmentation**
   - Random rotations
   - Random flips
   - Color jittering

### Intermediate Extensions

4. **Batch Normalization**
   - Add BatchNorm layers
   - Compare with/without
   - Faster convergence

5. **Different Optimizers**
   - SGD with momentum
   - Adam
   - RMSprop
   - Compare performance

6. **Visualize Learned Features**
   - Plot convolutional filters
   - Visualize activations
   - Understand what model sees

### Advanced Extensions

7. **Use Real Datasets**
   - CIFAR-10
   - ImageNet subset
   - Custom dataset

8. **Advanced Architectures**
   - ResNet
   - VGG
   - EfficientNet

9. **Object Detection**
   - Extend to detect objects
   - Draw bounding boxes
   - Multi-object detection

---

## 🔧 Troubleshooting

### Common Issues

**Issue 1: Model doesn't learn (loss doesn't decrease)**
- **Solution**: Check learning rate
- **Solution**: Verify data preprocessing
- **Solution**: Ensure images are normalized

**Issue 2: Overfitting (train accuracy >> test accuracy)**
- **Solution**: Add dropout
- **Solution**: Use data augmentation
- **Solution**: Reduce model capacity

**Issue 3: Out of memory**
- **Solution**: Reduce batch size
- **Solution**: Use smaller images
- **Solution**: Process in smaller batches

**Issue 4: Slow training**
- **Solution**: Use GPU if available
- **Solution**: Reduce image size
- **Solution**: Use transfer learning

---

## ✅ Success Criteria

- ✅ Model achieves >80% accuracy
- ✅ Training converges smoothly
- ✅ Test accuracy close to train accuracy
- ✅ Code is well-organized
- ✅ Can explain CNN architecture

---

## 🎓 Learning Outcomes

By completing this project, you will:

- ✅ Understand CNN architecture
- ✅ Design convolutional layers
- ✅ Preprocess image data
- ✅ Apply transfer learning
- ✅ Evaluate classification performance
- ✅ Visualize learned features

---

## 📖 Additional Resources

- **Step 8 Documentation**: `docs/Step_8_CNNs.md`
- **Step 8a Documentation**: `docs/Step_8a_Real_Datasets.md`
- **Step 8c Documentation**: `docs/Step_8c_Transfer_Learning.md`

---

**Ready to classify images? Let's build a CNN!** 🚀

**Next Steps**: After completing this project, move on to **Project 5: Time Series Predictor** to learn about RNNs for sequences.
