"""
Step 8c — Transfer Learning
Goal: Use pre-trained models and fine-tune for your task.
Tools: Python + PyTorch + torchvision
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore', message='.*Failed to initialize NumPy.*')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from plotting import plot_learning_curve

print("=== Step 8c: Transfer Learning ===")
print("Using pre-trained models for your tasks")
print()

# 8c.1 What is Transfer Learning?
print("=== 8c.1 What is Transfer Learning? ===")
print("Transfer Learning = Reuse knowledge from one task to another")
print()
print("Process:")
print("  1. Train model on large dataset (ImageNet)")
print("  2. Freeze early layers (keep learned features)")
print("  3. Train only final layers on your data")
print()
print("Benefits:")
print("  ✅ Faster training")
print("  ✅ Less data needed")
print("  ✅ Better performance")
print("  ✅ Saves computation")
print()

# 8c.2 Pre-trained Models
print("=== 8c.2 Pre-trained Models ===")
print("Popular pre-trained models:")
print("  - ResNet: Deep residual networks")
print("  - VGG: Very deep networks")
print("  - AlexNet: Early breakthrough")
print("  - EfficientNet: Efficient architectures")
print("  - Vision Transformer (ViT): Transformer for images")
print()
print("All trained on ImageNet (1,000 classes)")
print()

# 8c.3 Create Simple Dataset
print("=== 8c.3 Create Simple Dataset ===")
def create_simple_data(num_samples=200, img_size=224, num_classes=3):
    """Create simple 3-class dataset"""
    X = []
    y = []
    
    for i in range(num_samples):
        img = np.random.rand(3, img_size, img_size).astype(np.float32)
        class_id = i % num_classes
        # Add class-specific pattern
        img[class_id, :, :] += 0.5
        X.append(img)
        y.append(class_id)
    
    return np.array(X), np.array(y)

X, y = create_simple_data(num_samples=200, img_size=224, num_classes=3)
X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)

print(f"Created {len(X)} images")
print(f"Image size: 224x224 (ImageNet standard)")
print(f"Classes: 3")
print()

# 8c.4 Simulate Pre-trained Model
print("=== 8c.4 Simulate Pre-trained Model ===")
class PretrainedFeatureExtractor(nn.Module):
    """Simulates a pre-trained feature extractor"""
    def __init__(self):
        super(PretrainedFeatureExtractor, self).__init__()
        # Simulate pre-trained layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )
    
    def forward(self, x):
        return self.features(x)

# Create pre-trained feature extractor
pretrained_model = PretrainedFeatureExtractor()
print("Pre-trained feature extractor created")
print()

# 8c.5 Transfer Learning Approaches
print("=== 8c.5 Transfer Learning Approaches ===")
print("Approach 1: Feature Extraction (Freeze all)")
print("  - Freeze pre-trained layers")
print("  - Train only new classifier")
print("  - Fastest, least flexible")
print()
print("Approach 2: Fine-tuning (Train some layers)")
print("  - Freeze early layers")
print("  - Train later layers + classifier")
print("  - Better performance, more training")
print()
print("Approach 3: Full fine-tuning")
print("  - Train all layers")
print("  - Use lower learning rate")
print("  - Best performance, slowest")
print()

# 8c.6 Build Transfer Learning Model
print("=== 8c.6 Build Transfer Learning Model ===")
class TransferLearningModel(nn.Module):
    """Model using transfer learning"""
    def __init__(self, pretrained_features, num_classes=3):
        super(TransferLearningModel, self).__init__()
        
        # Use pre-trained features (frozen)
        self.features = pretrained_features
        
        # Freeze pre-trained layers
        for param in self.features.parameters():
            param.requires_grad = False
        
        # New classifier for our task
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Extract features using pre-trained model
        features = self.features(x)
        features = features.view(features.size(0), -1)  # Flatten
        # Classify with new layers
        output = self.classifier(features)
        return output

model = TransferLearningModel(pretrained_model, num_classes=3)
print("Transfer learning model:")
print(model)
print()

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
print("(Most parameters are frozen from pre-trained model)")
print()

# 8c.7 Training with Transfer Learning
print("=== 8c.7 Training with Transfer Learning ===")
loss_fn = nn.CrossEntropyLoss()
# Use lower learning rate for fine-tuning
optimizer = optim.Adam(model.parameters(), lr=0.0001)

losses = []
num_epochs = 30

print(f"Training for {num_epochs} epochs...")
print("(Only new classifier layers are being trained)")
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

# 8c.8 Learning Curve
print("=== 8c.8 Learning Curve ===")
plot_learning_curve(losses, title="Transfer Learning Training Loss", ylabel="Loss (Cross-Entropy)")

# 8c.9 Evaluation
print("=== 8c.9 Evaluation ===")
model.eval()
with torch.no_grad():
    predictions = model(X_tensor[:50])
    predicted_classes = torch.argmax(predictions, dim=1)
    correct = (predicted_classes == y_tensor[:50]).sum().item()
    accuracy = correct / 50 * 100
    print(f"Accuracy: {accuracy:.1f}% ({correct}/50)")
print()

# 8c.10 Using Real Pre-trained Models
print("=== 8c.10 Using Real Pre-trained Models ===")
print("To use real pre-trained models from torchvision:")
print("""
import torchvision.models as models

# Load pre-trained ResNet
resnet = models.resnet18(pretrained=True)

# Freeze all layers
for param in resnet.parameters():
    param.requires_grad = False

# Replace final layer
num_classes = 10
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

# Only new layer is trainable
for param in resnet.fc.parameters():
    param.requires_grad = True
""")
print()

# 8c.11 When to Use Transfer Learning
print("=== 8c.11 When to Use Transfer Learning ===")
print("✅ Good for:")
print("   - Small datasets")
print("   - Similar tasks to ImageNet")
print("   - Limited compute resources")
print("   - Quick prototyping")
print()
print("❌ May not help:")
print("   - Very different domains")
print("   - Very large datasets")
print("   - Specialized tasks")
print()

# 8c.12 Fine-tuning Strategy
print("=== 8c.12 Fine-tuning Strategy ===")
print("Step-by-step fine-tuning:")
print("  1. Start with frozen layers")
print("  2. Train classifier only")
print("  3. Unfreeze last few layers")
print("  4. Train with lower learning rate")
print("  5. Gradually unfreeze more if needed")
print()

print("🎉 Transfer learning complete!")
print("You've learned to leverage pre-trained models!")
