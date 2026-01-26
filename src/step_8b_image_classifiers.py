"""
Step 8b — Image Classifiers
Goal: Build and improve image classification models.
Tools: Python + PyTorch + NumPy
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore', message='.*Failed to initialize NumPy.*')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from plotting import plot_learning_curve

print("=== Step 8b: Image Classifiers ===")
print("Building and improving image classification models")
print()

# 8b.1 Classification Pipeline
print("=== 8b.1 Classification Pipeline ===")
print("Image classification steps:")
print("  1. Load and preprocess images")
print("  2. Build CNN architecture")
print("  3. Train on labeled data")
print("  4. Evaluate on test set")
print("  5. Make predictions on new images")
print()

# 8b.2 Create Multi-class Dataset
print("=== 8b.2 Create Multi-class Dataset ===")
def create_classification_data(num_classes=5, samples_per_class=100, img_size=64):
    """Create multi-class image classification data"""
    X = []
    y = []
    
    for class_id in range(num_classes):
        for _ in range(samples_per_class):
            # Create class-specific pattern
            img = np.zeros((1, img_size, img_size), dtype=np.float32)
            
            # Add class-specific features
            center = img_size // 2
            if class_id == 0:  # Circle
                y_coords, x_coords = np.ogrid[:img_size, :img_size]
                mask = (x_coords - center)**2 + (y_coords - center)**2 <= (img_size//4)**2
                img[0][mask] = 1.0
            elif class_id == 1:  # Square
                margin = img_size // 4
                img[0, margin:-margin, margin:-margin] = 1.0
            elif class_id == 2:  # Horizontal lines
                for i in range(0, img_size, 8):
                    img[0, i, :] = 1.0
            elif class_id == 3:  # Vertical lines
                for i in range(0, img_size, 8):
                    img[0, :, i] = 1.0
            else:  # Diagonal
                for i in range(img_size):
                    if i < img_size:
                        img[0, i, i] = 1.0
            
            X.append(img)
            y.append(class_id)
    
    return np.array(X), np.array(y)

X, y = create_classification_data(num_classes=5, samples_per_class=80, img_size=64)
print(f"Created {len(X)} images")
print(f"Classes: {len(np.unique(y))}")
print(f"Image shape: {X[0].shape}")
print()

# 8b.3 Improved CNN Architecture
print("=== 8b.3 Improved CNN Architecture ===")
class ImprovedImageClassifier(nn.Module):
    """Improved CNN for image classification"""
    def __init__(self, num_classes=5):
        super(ImprovedImageClassifier, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = ImprovedImageClassifier(num_classes=5)
print("Improved model architecture:")
print(model)
print()

# 8b.4 Key Improvements
print("=== 8b.4 Key Improvements ===")
print("1. Batch Normalization:")
print("   - Normalizes activations")
print("   - Faster training")
print("   - Better convergence")
print()
print("2. Dropout:")
print("   - Prevents overfitting")
print("   - Randomly sets neurons to zero")
print("   - Forces network to be robust")
print()
print("3. Deeper Architecture:")
print("   - More layers = more features")
print("   - Hierarchical feature learning")
print()
print("4. Better Regularization:")
print("   - Dropout2d for conv layers")
print("   - Dropout for FC layers")
print()

# 8b.5 Training
print("=== 8b.5 Training Improved Classifier ===")
X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

losses = []
num_epochs = 50

print(f"Training for {num_epochs} epochs...")
for epoch in range(num_epochs):
    model.train()
    predictions = model(X_tensor)
    loss = loss_fn(predictions, y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

print("Training complete!")
print()

# 8b.6 Learning Curve
print("=== 8b.6 Learning Curve ===")
plot_learning_curve(losses, title="Image Classifier Training Loss", ylabel="Loss (Cross-Entropy)")

# 8b.7 Evaluation
print("=== 8b.7 Evaluation ===")
model.eval()
with torch.no_grad():
    predictions = model(X_tensor[:50])
    predicted_classes = torch.argmax(predictions, dim=1)
    
    correct = (predicted_classes == y_tensor[:50]).sum().item()
    accuracy = correct / 50 * 100
    
    print(f"Accuracy on 50 samples: {accuracy:.1f}%")
    print(f"Correct: {correct}/50")
print()

# 8b.8 Making Predictions
print("=== 8b.8 Making Predictions ===")
def predict_image(model, image_tensor):
    """Predict class for a single image"""
    model.eval()
    with torch.no_grad():
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        return predicted_class, confidence

# Test on a few images
print("Sample predictions:")
for i in range(5):
    pred_class, confidence = predict_image(model, X_tensor[i])
    actual_class = y_tensor[i].item()
    match = "✓" if pred_class == actual_class else "✗"
    print(f"  Image {i+1}: Actual={actual_class}, Predicted={pred_class}, "
          f"Confidence={confidence:.2f} {match}")
print()

# 8b.9 Classification Metrics
print("=== 8b.9 Classification Metrics ===")
print("Important metrics for classification:")
print("  - Accuracy: Overall correctness")
print("  - Precision: True positives / (True + False positives)")
print("  - Recall: True positives / (True + False negatives)")
print("  - F1-Score: Harmonic mean of precision and recall")
print("  - Confusion Matrix: Shows per-class performance")
print()

# 8b.10 Tips for Better Classification
print("=== 8b.10 Tips for Better Classification ===")
print("✅ Data:")
print("   - More training data")
print("   - Balanced classes")
print("   - Data augmentation")
print()
print("✅ Architecture:")
print("   - Deeper networks")
print("   - Batch normalization")
print("   - Residual connections")
print()
print("✅ Training:")
print("   - Learning rate scheduling")
print("   - Early stopping")
print("   - Ensemble methods")
print()

print("🎉 Image classifier complete!")
print("You've built an improved image classification model!")
