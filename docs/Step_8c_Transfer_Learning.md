# Step 8c — Transfer Learning

> **Goal:** Use pre-trained models and fine-tune for your task.  
> **Tools:** Python + PyTorch + torchvision

---

## 8c.1 What is Transfer Learning?

Reuse knowledge from one task to another.

**Process:**
1. Train model on large dataset (ImageNet)
2. Freeze early layers (keep learned features)
3. Train only final layers on your data

**Benefits:**
- Faster training
- Less data needed
- Better performance
- Saves computation

---

## 8c.2 Pre-trained Models

Popular models:
- ResNet
- VGG
- AlexNet
- EfficientNet
- Vision Transformer (ViT)

All trained on ImageNet (1,000 classes).

---

## 8c.3 Transfer Learning Approaches

**Approach 1: Feature Extraction**
- Freeze all pre-trained layers
- Train only new classifier
- Fastest, least flexible

**Approach 2: Fine-tuning**
- Freeze early layers
- Train later layers + classifier
- Better performance

**Approach 3: Full Fine-tuning**
- Train all layers
- Use lower learning rate
- Best performance, slowest

---

## 8c.4 When to Use

✅ **Good for:**
- Small datasets
- Similar tasks to ImageNet
- Limited compute resources
- Quick prototyping

❌ **May not help:**
- Very different domains
- Very large datasets
- Specialized tasks
