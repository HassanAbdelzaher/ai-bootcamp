"""
Step 8a — Real Datasets (CIFAR-10, ImageNet)
Goal: Train CNNs on real image datasets like CIFAR-10.
Tools: Python + PyTorch + torchvision
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore', message='.*Failed to initialize NumPy.*')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from plotting import plot_learning_curve

print("=== Step 8a: Real Datasets (CIFAR-10) ===")
print("Training CNNs on real-world image datasets")
print()

# 8a.1 About CIFAR-10
print("=== 8a.1 About CIFAR-10 ===")
print("CIFAR-10 dataset:")
print("  - 60,000 color images")
print("  - 32x32 pixels")
print("  - 10 classes: airplane, automobile, bird, cat, deer,")
print("                dog, frog, horse, ship, truck")
print("  - 50,000 training, 10,000 test")
print()

# 8a.2 Create Synthetic CIFAR-like Data
print("=== 8a.2 Create Synthetic CIFAR-like Data ===")
print("Note: For this example, we'll create synthetic data")
print("In practice, use: torchvision.datasets.CIFAR10")
print()

def create_cifar_like_data(num_samples=1000, img_size=32, num_classes=10):
    """Create synthetic CIFAR-like data"""
    # Lists to store images and labels
    X = []  # Images
    y = []  # Class labels
    
    for i in range(num_samples):
        # Create random color image (3 channels: RGB)
        # np.random.rand(3, img_size, img_size) creates random values in [0, 1)
        # Shape: (3, 32, 32) = 3 channels, 32 height, 32 width
        # .astype(np.float32) converts to 32-bit float (PyTorch compatible)
        img = np.random.rand(3, img_size, img_size).astype(np.float32)
        
        # Add some structure based on class (makes classification possible)
        # class_id: Assign class based on sample index (cycles through classes)
        # i % num_classes: Remainder when dividing by num_classes (0-9)
        class_id = i % num_classes
        
        # Add class-specific patterns (simplified)
        # Enhance one channel based on class to create distinguishable patterns
        # class_id % 3: Selects which channel (0, 1, or 2) to enhance
        # += 0.3: Adds 0.3 to all pixels in that channel (makes it brighter)
        img[class_id % 3, :, :] += 0.3
        
        # Store image and label
        X.append(img)
        y.append(class_id)
    
    # Convert lists to NumPy arrays
    return np.array(X), np.array(y)

# Create smaller dataset for faster training
X_train, y_train = create_cifar_like_data(num_samples=500, img_size=32, num_classes=10)
X_test, y_test = create_cifar_like_data(num_samples=100, img_size=32, num_classes=10)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Image shape: {X_train[0].shape} (channels, height, width)")
print(f"Classes: {len(np.unique(y_train))}")
print()

# 8a.3 Prepare Data Loaders
print("=== 8a.3 Prepare Data Loaders ===")
# Convert to tensors
# torch.FloatTensor(): Convert NumPy array to PyTorch float tensor
# X_train shape: (500, 3, 32, 32) - 500 images, 3 channels, 32x32 pixels
X_train_tensor = torch.FloatTensor(X_train)
# torch.LongTensor(): Convert NumPy array to PyTorch long (integer) tensor
# y_train shape: (500,) - 500 class labels
y_train_tensor = torch.LongTensor(y_train)

# Test tensors (same process)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# Create datasets
# TensorDataset: PyTorch dataset that pairs inputs and labels
# train_dataset: Pairs X_train_tensor with y_train_tensor
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# test_dataset: Pairs X_test_tensor with y_test_tensor
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create data loaders
# DataLoader: Provides batches of data for training
# batch_size=32: Process 32 images at a time (faster than one-by-one)
# shuffle=True: Randomize order for training (helps learning)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# shuffle=False: Keep test data in order (for consistent evaluation)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Data loaders created")
print(f"  Batch size: 32")
print(f"  Training batches: {len(train_loader)}")
print(f"  Test batches: {len(test_loader)}")
print()

# 8a.4 Build CIFAR-10 CNN
print("=== 8a.4 Build CIFAR-10 CNN ===")
class CIFAR10CNN(nn.Module):
    """CNN for CIFAR-10 classification"""
    def __init__(self, num_classes=10):
        super(CIFAR10CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # x: (batch, 3, 32, 32)
        x = self.pool(F.relu(self.conv1(x)))  # (batch, 32, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))  # (batch, 64, 8, 8)
        x = self.pool(F.relu(self.conv3(x)))  # (batch, 128, 4, 4)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CIFAR10CNN(num_classes=10)
print("Model architecture:")
print(model)
print()

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print()

# 8a.5 Training
print("=== 8a.5 Training on CIFAR-like Data ===")
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

losses = []
num_epochs = 20

print(f"Training for {num_epochs} epochs...")
for epoch in range(num_epochs):
    epoch_loss = 0
    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        output = model(data)
        loss = loss_fn(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

print("Training complete!")
print()

# 8a.6 Learning Curve
print("=== 8a.6 Learning Curve ===")
plot_learning_curve(losses, title="CIFAR-10 Training Loss", ylabel="Loss (Cross-Entropy)")

# 8a.7 Evaluation
print("=== 8a.7 Evaluation ===")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        predicted = torch.argmax(output, dim=1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{total})")
print()

# 8a.8 About ImageNet
print("=== 8a.8 About ImageNet ===")
print("ImageNet dataset:")
print("  - 14 million images")
print("  - 1,000 classes")
print("  - 224x224 pixels (standard)")
print("  - Used for ImageNet Challenge")
print()
print("Famous models trained on ImageNet:")
print("  - AlexNet (2012)")
print("  - VGG (2014)")
print("  - ResNet (2015)")
print("  - Inception (2015)")
print()

# 8a.9 Using Real CIFAR-10
print("=== 8a.9 Using Real CIFAR-10 ===")
print("To use real CIFAR-10 dataset:")
print("""
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=32, shuffle=True
)
""")
print()

# 8a.10 Data Augmentation
print("=== 8a.10 Data Augmentation ===")
print("Common augmentations for images:")
print("  - Random horizontal flip")
print("  - Random rotation")
print("  - Color jitter")
print("  - Random crop")
print("  - Normalization")
print()
print("Benefits:")
print("  ✅ More training data")
print("  ✅ Better generalization")
print("  ✅ Reduces overfitting")
print()

print("🎉 Real dataset training complete!")
print("You've learned how to train CNNs on real image datasets!")
