# Project 8d: Object Detection

> **Detect and locate objects in images using bounding boxes**

**Difficulty**: ⭐⭐⭐⭐ Expert  
**Time**: 5-6 hours  
**Prerequisites**: Steps 0-8 (Especially Step 8: CNNs, Step 8a: Real Datasets)

---

## 📚 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem: Object Detection](#problem-object-detection)
3. [Key Concepts](#key-concepts)
4. [Step-by-Step Guide](#step-by-step-guide)
5. [Expected Results](#expected-results)
6. [Extension Ideas](#extension-ideas)

---

## 🎯 Project Overview

This project teaches you to detect and locate objects in images. You'll learn to:

- Understand object detection vs classification
- Create bounding box annotations
- Build object detection models
- Implement YOLO-style detection (simplified)
- Evaluate detection performance

### Why Object Detection?

- **Real-world applications**: Autonomous vehicles, security, medical imaging
- **Foundation for vision**: Prepares for advanced tasks
- **Industry demand**: High-value skill
- **Portfolio project**: Impressive demonstration

---

## 📋 Problem: Object Detection

### Task

Build an object detection system that:
1. **Detects objects**: Find objects in images
2. **Draws bounding boxes**: Locate objects with rectangles
3. **Classifies objects**: Identify what each object is
4. **Handles multiple objects**: Detect multiple objects per image

### Learning Objectives

- Understand bounding boxes
- Build detection models
- Handle multi-object scenarios
- Evaluate detection performance

---

## 🧠 Key Concepts

### 1. Object Detection vs Classification

**Classification**: "What is in the image?" (single label)
**Detection**: "What AND where?" (multiple objects with locations)

### 2. Bounding Boxes

**Format**: (x_min, y_min, x_max, y_max) or (center_x, center_y, width, height)

**Example**: Car at (100, 50, 200, 150) means:
- Top-left: (100, 50)
- Bottom-right: (200, 150)
- Width: 100, Height: 100

### 3. Detection Approaches

**Two-stage** (R-CNN): Detect regions, then classify
**One-stage** (YOLO): Detect and classify simultaneously

---

## 🚀 Step-by-Step Guide

### Step 1: Create Detection Dataset

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_detection_data(num_samples=100, img_size=128):
    """Create synthetic object detection data"""
    images = []
    boxes = []
    labels = []
    
    for i in range(num_samples):
        # Create blank image
        img = np.zeros((3, img_size, img_size), dtype=np.float32)
        
        # Add object (square or circle)
        obj_type = i % 2  # 0=square, 1=circle
        center_x = np.random.randint(30, img_size - 30)
        center_y = np.random.randint(30, img_size - 30)
        size = np.random.randint(20, 40)
        
        # Create bounding box
        x_min = max(0, center_x - size // 2)
        y_min = max(0, center_y - size // 2)
        x_max = min(img_size, center_x + size // 2)
        y_max = min(img_size, center_y + size // 2)
        
        # Draw object
        if obj_type == 0:  # Square
            img[:, y_min:y_max, x_min:x_max] = 1.0
        else:  # Circle
            y_coords, x_coords = np.ogrid[:img_size, :img_size]
            mask = ((x_coords - center_x)**2 + (y_coords - center_y)**2) <= (size//2)**2
            img[:, mask] = 1.0
        
        images.append(img)
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(obj_type)
    
    return np.array(images), np.array(boxes), np.array(labels)

X, boxes, labels = create_detection_data(num_samples=200, img_size=128)
print(f"Created {len(X)} images with objects")
```

### Step 2: Build Detection Model

```python
import torch
import torch.nn as nn

class SimpleDetector(nn.Module):
    """Simple object detection model"""
    def __init__(self, num_classes=2):
        super(SimpleDetector, self).__init__()
        
        # Backbone (feature extractor)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Detection head (bounding box + class)
        self.bbox_regressor = nn.Linear(128, 4)  # x_min, y_min, x_max, y_max
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        bbox = self.bbox_regressor(features)
        class_logits = self.classifier(features)
        
        return bbox, class_logits
```

### Step 3: Loss Functions

```python
def detection_loss(bbox_pred, bbox_true, class_pred, class_true, lambda_box=1.0):
    """Combined loss for detection"""
    # Bounding box loss (L1 or smooth L1)
    bbox_loss = nn.SmoothL1Loss()(bbox_pred, bbox_true)
    
    # Classification loss
    class_loss = nn.CrossEntropyLoss()(class_pred, class_true)
    
    # Combined loss
    total_loss = lambda_box * bbox_loss + class_loss
    
    return total_loss, bbox_loss, class_loss
```

### Step 4: Training

```python
model = SimpleDetector(num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Normalize boxes to [0, 1]
boxes_normalized = boxes / 128.0
X_tensor = torch.FloatTensor(X)
boxes_tensor = torch.FloatTensor(boxes_normalized)
labels_tensor = torch.LongTensor(labels)

epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    bbox_pred, class_pred = model(X_tensor)
    loss, bbox_loss, class_loss = detection_loss(
        bbox_pred, boxes_tensor, class_pred, labels_tensor
    )
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: Total={loss.item():.4f}, "
              f"BBox={bbox_loss.item():.4f}, Class={class_loss.item():.4f}")
```

### Step 5: Visualization

```python
def visualize_detection(img, bbox_pred, bbox_true, class_pred, class_true):
    """Visualize detection results"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Denormalize boxes
    img_size = img.shape[1]
    bbox_pred = bbox_pred * img_size
    bbox_true = bbox_true * img_size
    
    # Plot predicted
    axes[0].imshow(img.transpose(1, 2, 0))
    rect_pred = patches.Rectangle(
        (bbox_pred[0], bbox_pred[1]),
        bbox_pred[2] - bbox_pred[0],
        bbox_pred[3] - bbox_pred[1],
        linewidth=2, edgecolor='r', facecolor='none'
    )
    axes[0].add_patch(rect_pred)
    axes[0].set_title(f'Predicted: Class {class_pred}')
    
    # Plot ground truth
    axes[1].imshow(img.transpose(1, 2, 0))
    rect_true = patches.Rectangle(
        (bbox_true[0], bbox_true[1]),
        bbox_true[2] - bbox_true[0],
        bbox_true[3] - bbox_true[1],
        linewidth=2, edgecolor='g', facecolor='none'
    )
    axes[1].add_patch(rect_true)
    axes[1].set_title(f'Ground Truth: Class {class_true}')
    
    plt.tight_layout()
    plt.show()

# Visualize first image
model.eval()
with torch.no_grad():
    bbox_pred, class_pred = model(X_tensor[:1])
    class_pred_idx = torch.argmax(class_pred, dim=1)
    
visualize_detection(
    X[0], bbox_pred[0].numpy(), boxes_normalized[0],
    class_pred_idx[0].item(), labels[0]
)
```

---

## 📊 Expected Results

### Training Output

```
Epoch 20: Total=0.5234, BBox=0.3456, Class=0.1778
Epoch 40: Total=0.2345, BBox=0.1234, Class=0.1111
Epoch 60: Total=0.1234, BBox=0.0567, Class=0.0667
Epoch 80: Total=0.0678, BBox=0.0234, Class=0.0444
Epoch 100: Total=0.0345, BBox=0.0123, Class=0.0222
```

### Detection Results

- Bounding boxes accurately locate objects
- Classification correctly identifies object types
- IoU (Intersection over Union) > 0.7 for good detections

---

## 💡 Extension Ideas

1. **Multiple Objects**
   - Detect multiple objects per image
   - Non-maximum suppression
   - Handle overlapping boxes

2. **Advanced Architectures**
   - YOLO-style detection
   - R-CNN approach
   - Anchor-based detection

3. **Real Datasets**
   - COCO dataset
   - Pascal VOC
   - Custom datasets

---

## ✅ Success Criteria

- ✅ Detect objects in images
- ✅ Draw accurate bounding boxes
- ✅ Classify detected objects
- ✅ Handle multiple objects
- ✅ Evaluate with IoU metric

---

**Ready to detect objects? Let's build a detection system!** 🚀
