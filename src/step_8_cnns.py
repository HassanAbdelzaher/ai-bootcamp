"""
Step 8 — CNNs (Convolutional Neural Networks for Images)
Goal: Learn how to process images using Convolutional Neural Networks.
Tools: Python + PyTorch + NumPy + Matplotlib
"""

# Import numpy first to ensure proper initialization before PyTorch
import numpy as np
import warnings

# Suppress NumPy initialization warnings (common with PyTorch)
warnings.filterwarnings('ignore', message='.*Failed to initialize NumPy.*')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from plotting import plot_learning_curve

# Check if PyTorch is available
try:
    print("PyTorch version:", torch.__version__)
except ImportError:
    print("ERROR: PyTorch is not installed!")
    print("Install with: pip install torch torchvision torchaudio")
    exit(1)
print()

# 8.1 Big Idea: Why CNNs for Images?
print("=== 8.1 Big Idea: Why CNNs for Images? ===")
print("Regular neural networks:")
print("  - Treat images as flat vectors")
print("  - Lose spatial structure")
print("  - Too many parameters")
print("")
print("CNNs (Convolutional Neural Networks):")
print("  - Preserve spatial structure")
print("  - Detect patterns (edges, shapes, objects)")
print("  - Fewer parameters through weight sharing")
print("  - Perfect for: images, videos, medical scans")
print()

# 8.2 Understanding Images as Data
print("=== 8.2 Understanding Images as Data ===")
print("Images are 2D arrays of pixels")
print("  - Grayscale: (height, width)")
print("  - Color: (height, width, channels)")
print("  - Example: 28x28 grayscale image = 784 pixels")
print()

# Create simple synthetic image data
def create_simple_image_data(num_samples=200, img_size=28):
    """Create simple synthetic images: circles vs squares"""
    X = []
    y = []
    
    for i in range(num_samples):
        img = np.zeros((img_size, img_size))
        
        if i % 2 == 0:  # Circle
            center = img_size // 2
            radius = img_size // 4
            y_val, x_val = np.ogrid[:img_size, :img_size]
            mask = (x_val - center)**2 + (y_val - center)**2 <= radius**2
            img[mask] = 1.0
            label = 0  # Circle
        else:  # Square
            margin = img_size // 4
            img[margin:-margin, margin:-margin] = 1.0
            label = 1  # Square
        
        X.append(img)
        y.append(label)
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

X_images, y_labels = create_simple_image_data(num_samples=200, img_size=28)
print(f"Created {len(X_images)} images")
print(f"Image shape: {X_images[0].shape}")
print(f"Labels: {np.unique(y_labels)} (0=Circle, 1=Square)")
print()

# 8.3 Preparing Image Data for CNN
print("=== 8.3 Preparing Image Data for CNN ===")
print("CNNs need data in shape: (batch, channels, height, width)")
print()

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X_images).unsqueeze(1)  # Add channel dimension: (200, 1, 28, 28)
y_tensor = torch.LongTensor(y_labels)  # (200,)

print("Tensor shapes:")
print(f"  X: {X_tensor.shape} (batch, channels, height, width)")
print(f"  y: {y_tensor.shape} (batch,)")
print()

# 8.4 Understanding Convolution
print("=== 8.4 Understanding Convolution ===")
print("Convolution = sliding a filter over the image")
print("  - Detects patterns (edges, corners, textures)")
print("  - Each filter learns different features")
print()

# Simple convolution example
example_img = X_images[0]
print("Example: 3x3 edge detection filter")
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

# Manual convolution (simplified)
print("Filter:")
print(kernel)
print("This filter detects edges in the image")
print()

# 8.5 Building a CNN
print("=== 8.5 Building a CNN ===")
class SimpleCNN(nn.Module):
    """Simple CNN for image classification"""
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        # Conv1: 1 channel → 16 channels, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # Conv2: 16 channels → 32 channels, 3x3 kernel
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Pooling layer (reduces size)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # After conv layers: 32 channels, 7x7 (28/2/2 = 7)
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # x shape: (batch, 1, 28, 28)
        
        # First conv block
        x = self.conv1(x)  # (batch, 16, 28, 28)
        x = F.relu(x)
        x = self.pool(x)   # (batch, 16, 14, 14)
        
        # Second conv block
        x = self.conv2(x)  # (batch, 32, 14, 14)
        x = F.relu(x)
        x = self.pool(x)   # (batch, 32, 7, 7)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # (batch, 32*7*7)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

model = SimpleCNN(num_classes=2)
print("Model architecture:")
print(model)
print()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
print()

# 8.6 Understanding CNN Components
print("=== 8.6 Understanding CNN Components ===")
print("1. Convolutional Layer (Conv2d):")
print("   - Applies filters to detect patterns")
print("   - Preserves spatial relationships")
print("")
print("2. Pooling Layer (MaxPool2d):")
print("   - Reduces image size")
print("   - Makes network translation-invariant")
print("   - Reduces computation")
print("")
print("3. Fully Connected Layer:")
print("   - Final classification")
print("   - Combines learned features")
print()

# 8.7 Training the CNN
print("=== 8.7 Training the CNN ===")
loss_fn = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

losses = []
num_epochs = 50

print(f"Training for {num_epochs} epochs...")
for epoch in range(num_epochs):
    # Forward pass
    predictions = model(X_tensor)
    loss = loss_fn(predictions, y_tensor)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

print("Training complete!")
print()

# 8.8 Learning Curve
print("=== 8.8 Learning Curve ===")
plot_learning_curve(losses, title="CNN Training Loss", ylabel="Loss (Cross-Entropy)")

# 8.9 Testing the CNN
print("=== 8.9 Testing the CNN ===")
model.eval()  # Set to evaluation mode
with torch.no_grad():
    test_predictions = model(X_tensor[:10])  # Test on first 10 samples
    predicted_classes = torch.argmax(test_predictions, dim=1)
    
    print("Sample predictions:")
    correct = 0
    for i in range(10):
        actual = y_tensor[i].item()
        predicted = predicted_classes[i].item()
        shape_name = "Circle" if actual == 0 else "Square"
        is_correct = "✓" if actual == predicted else "✗"
        if actual == predicted:
            correct += 1
        print(f"  Image {i+1}: {shape_name} → Predicted: {predicted} ({is_correct})")
    
    accuracy = correct / 10 * 100
    print(f"\nAccuracy on 10 samples: {accuracy:.1f}%")
print()

# 8.10 Visualizing Learned Features (Optional)
print("=== 8.10 Visualizing Learned Features ===")
print("CNNs learn to detect:")
print("  - Edges and lines (early layers)")
print("  - Shapes and patterns (middle layers)")
print("  - Complex objects (later layers)")
print()

# Get first layer filters
first_conv_weights = model.conv1.weight.data
print(f"First conv layer has {first_conv_weights.shape[0]} filters")
print("Each filter learns to detect different patterns")
print()

# 8.11 Why CNNs Work for Images
print("=== 8.11 Why CNNs Work for Images ===")
print("✅ Translation invariance:")
print("   - Object in different positions → same detection")
print("")
print("✅ Parameter sharing:")
print("   - Same filter used across entire image")
print("   - Much fewer parameters than fully connected")
print("")
print("✅ Hierarchical features:")
print("   - Early layers: edges, corners")
print("   - Middle layers: shapes, textures")
print("   - Late layers: objects, faces")
print()

# 8.12 Real-World Applications
print("=== 8.12 Real-World Applications ===")
print("CNNs are used in:")
print("  📸 Image classification (cats vs dogs)")
print("  🚗 Self-driving cars (object detection)")
print("  🏥 Medical imaging (tumor detection)")
print("  📱 Face recognition")
print("  🎨 Style transfer and image generation")
print("  🔍 Satellite image analysis")
print()

# 8.13 Advanced CNN Architectures
print("=== 8.13 Advanced CNN Architectures ===")
print("Famous CNN architectures:")
print("  - LeNet (1998): First successful CNN")
print("  - AlexNet (2012): Deep learning breakthrough")
print("  - VGG (2014): Very deep networks")
print("  - ResNet (2015): Residual connections")
print("  - Inception (2015): Multiple filter sizes")
print()

# 8.14 Next Steps
print("=== 8.14 Next Steps ===")
print("You've learned:")
print("  ✅ What CNNs are and why they work for images")
print("  ✅ How convolution and pooling work")
print("  ✅ How to build a CNN in PyTorch")
print("  ✅ How to train on image data")
print("")
print("Try these next:")
print("  - Train on real datasets (CIFAR-10, ImageNet)")
print("  - Build image classifiers")
print("  - Explore transfer learning")
print("  - Learn about object detection (YOLO, R-CNN)")
print("  - Try image generation (GANs, VAEs)")
print()

print("🎉 Congratulations on completing Step 8: CNNs!")
print("You now understand how AI processes images!")
