"""
Step 8d — Object Detection (YOLO, R-CNN)
Goal: Introduction to object detection algorithms.
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

print("=== Step 8d: Object Detection (YOLO, R-CNN) ===")
print("Introduction to object detection in images")
print()

# 8d.1 Object Detection vs Classification
print("=== 8d.1 Object Detection vs Classification ===")
print("Image Classification:")
print("  - What is in the image?")
print("  - Single label per image")
print("  - Example: 'This is a cat'")
print()
print("Object Detection:")
print("  - What AND where?")
print("  - Multiple objects per image")
print("  - Bounding boxes + labels")
print("  - Example: 'Cat at (100, 50, 200, 150)'")
print()

# 8d.2 Bounding Boxes
print("=== 8d.2 Bounding Boxes ===")
print("Bounding box format:")
print("  (x_min, y_min, x_max, y_max)")
print("  or")
print("  (center_x, center_y, width, height)")
print()
print("Example:")
print("  Image: 640x480")
print("  Box: (100, 50, 200, 150)")
print("  → Object from (100, 50) to (200, 150)")
print()

# 8d.3 Create Simple Detection Data
print("=== 8d.3 Create Simple Detection Data ===")
def create_detection_data(num_samples=100, img_size=128):
    """Create simple object detection data"""
    X = []
    y_boxes = []
    y_classes = []
    
    for i in range(num_samples):
        img = np.zeros((3, img_size, img_size), dtype=np.float32)
        
        # Add object (square or circle)
        obj_type = i % 2  # 0 = square, 1 = circle
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
        
        X.append(img)
        y_boxes.append([x_min, y_min, x_max, y_max])
        y_classes.append(obj_type)
    
    return np.array(X), np.array(y_boxes), np.array(y_classes)

X, boxes, classes = create_detection_data(num_samples=200, img_size=128)
print(f"Created {len(X)} images with objects")
print(f"Bounding boxes: {len(boxes)}")
print(f"Classes: {len(np.unique(classes))} (0=Square, 1=Circle)")
print()

# 8d.4 Simple Detection Model
print("=== 8d.4 Simple Detection Model ===")
class SimpleDetector(nn.Module):
    """Simple object detection model"""
    def __init__(self, num_classes=2):
        super(SimpleDetector, self).__init__()
        
        # Feature extraction
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),  # Reduce to smaller feature map
        )
        
        # Detection heads
        # Box regression: predict 4 values (x_min, y_min, x_max, y_max)
        self.box_head = nn.Linear(64 * 8 * 8, 4)
        # Classification: predict class
        self.class_head = nn.Linear(64 * 8 * 8, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        boxes = self.box_head(features)
        classes = self.class_head(features)
        
        return boxes, classes

model = SimpleDetector(num_classes=2)
print("Detection model architecture:")
print(model)
print()

# 8d.5 Training Detection Model
print("=== 8d.5 Training Detection Model ===")
X_tensor = torch.FloatTensor(X)
boxes_tensor = torch.FloatTensor(boxes)
classes_tensor = torch.LongTensor(classes)

# Combined loss: box regression + classification
def detection_loss(pred_boxes, pred_classes, target_boxes, target_classes):
    """Combined loss for detection"""
    # Box regression loss (L1)
    box_loss = F.l1_loss(pred_boxes, target_boxes)
    
    # Classification loss (Cross-entropy)
    class_loss = F.cross_entropy(pred_classes, target_classes)
    
    # Combined
    return box_loss + class_loss

optimizer = optim.Adam(model.parameters(), lr=0.001)

losses = []
num_epochs = 50

print(f"Training for {num_epochs} epochs...")
for epoch in range(num_epochs):
    model.train()
    pred_boxes, pred_classes = model(X_tensor)
    loss = detection_loss(pred_boxes, pred_classes, boxes_tensor, classes_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

print("Training complete!")
print()

# 8d.6 Learning Curve
print("=== 8d.6 Learning Curve ===")
plot_learning_curve(losses, title="Object Detection Training Loss", ylabel="Loss")

# 8d.7 R-CNN Overview
print("=== 8d.7 R-CNN Overview ===")
print("R-CNN = Region-based CNN")
print()
print("Process:")
print("  1. Generate region proposals (candidate boxes)")
print("  2. Extract features for each region")
print("  3. Classify each region")
print("  4. Refine bounding boxes")
print()
print("Evolution:")
print("  - R-CNN (2014): Slow, many proposals")
print("  - Fast R-CNN (2015): Faster, shared features")
print("  - Faster R-CNN (2016): End-to-end, learn proposals")
print()

# 8d.8 YOLO Overview
print("=== 8d.8 YOLO Overview ===")
print("YOLO = You Only Look Once")
print()
print("Key idea:")
print("  - Single pass through network")
print("  - Predict boxes and classes directly")
print("  - Much faster than R-CNN")
print()
print("Process:")
print("  1. Divide image into grid")
print("  2. Each cell predicts boxes + classes")
print("  3. Non-maximum suppression (remove duplicates)")
print()
print("Versions:")
print("  - YOLOv1 (2016): First version")
print("  - YOLOv2/YOLO9000 (2017): Better accuracy")
print("  - YOLOv3 (2018): Multi-scale detection")
print("  - YOLOv4/v5/v6/v7/v8: Continued improvements")
print()

# 8d.9 YOLO vs R-CNN
print("=== 8d.9 YOLO vs R-CNN ===")
print("R-CNN:")
print("  ✅ Higher accuracy")
print("  ✅ Better for small objects")
print("  ❌ Slower (multiple passes)")
print("  ❌ More complex")
print()
print("YOLO:")
print("  ✅ Very fast (real-time)")
print("  ✅ Simpler architecture")
print("  ✅ End-to-end training")
print("  ❌ Lower accuracy on small objects")
print()

# 8d.10 Evaluation Metrics
print("=== 8d.10 Evaluation Metrics ===")
print("mAP (mean Average Precision):")
print("  - Primary metric for object detection")
print("  - Measures accuracy of boxes and classes")
print("  - Higher is better (0-1 scale)")
print()
print("IoU (Intersection over Union):")
print("  - Measures box overlap")
print("  - IoU = (Intersection) / (Union)")
print("  - Good detection: IoU > 0.5")
print()

# 8d.11 Real-World Applications
print("=== 8d.11 Real-World Applications ===")
print("Object detection is used in:")
print("  🚗 Self-driving cars (detect cars, pedestrians)")
print("  🏪 Retail (inventory tracking)")
print("  🏥 Medical imaging (tumor detection)")
print("  🛡️  Security (person detection)")
print("  📱 Mobile apps (AR, photo tagging)")
print("  🏭 Manufacturing (quality control)")
print()

# 8d.12 Using Pre-trained Models
print("=== 8d.12 Using Pre-trained Models ===")
print("Popular detection models:")
print("  - YOLOv8 (Ultralytics)")
print("  - Detectron2 (Facebook)")
print("  - MMDetection (OpenMMLab)")
print()
print("Example with YOLOv8:")
print("""
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolov8n.pt')  # nano version

# Detect objects
results = model('image.jpg')

# Results contain boxes, classes, confidence
""")
print()

# 8d.13 Next Steps
print("=== 8d.13 Next Steps ===")
print("You've learned:")
print("  ✅ Difference between classification and detection")
print("  ✅ How bounding boxes work")
print("  ✅ R-CNN and YOLO architectures")
print("  ✅ Detection model training")
print()
print("Try these next:")
print("  - Use pre-trained YOLO model")
print("  - Train on custom dataset")
print("  - Explore instance segmentation")
print("  - Learn about anchor boxes")
print()

print("🎉 Object detection learning complete!")
print("You now understand how AI detects objects in images!")
