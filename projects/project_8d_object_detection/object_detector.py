"""
Project 8d: Object Detection
Detect and locate objects in images using bounding boxes
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os

# Add parent directory to path to import plotting utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from plotting import plot_learning_curve

print("=" * 70)
print("Project 8d: Object Detection")
print("=" * 70)
print()

# ============================================================================
# Step 1: Create Detection Dataset
# ============================================================================
print("=" * 70)
print("Step 1: Creating Object Detection Dataset")
print("=" * 70)
print()

def create_detection_data(num_samples=100, img_size=128):
    """Create simple object detection data"""
    # Lists to store images, bounding boxes, and labels
    images = []   # RGB images
    boxes = []    # Bounding box coordinates
    labels = []   # Object class labels
    
    for i in range(num_samples):
        # Create blank image (black background)
        # np.zeros((3, img_size, img_size)): 3 channels (RGB), img_size×img_size pixels
        # dtype=np.float32: 32-bit float (PyTorch compatible)
        # Shape: (3, 128, 128) - channels first format
        img = np.zeros((3, img_size, img_size), dtype=np.float32)
        
        # Determine object type: Alternate between square and circle
        # i % 2: Remainder when dividing by 2 (0 or 1)
        # 0 = square, 1 = circle
        obj_type = i % 2  # 0=square, 1=circle
        
        # Randomly position object in image
        # center_x, center_y: Center coordinates of object
        # np.random.randint(30, img_size - 30): Random integer in [30, img_size-30)
        # Keeps object away from edges (ensures full object visible)
        center_x = np.random.randint(30, img_size - 30)
        center_y = np.random.randint(30, img_size - 30)
        
        # Random object size
        # np.random.randint(20, 40): Random size between 20 and 40 pixels
        size = np.random.randint(20, 40)
        
        # Create bounding box coordinates
        # Bounding box format: (x_min, y_min, x_max, y_max)
        # center_x - size // 2: Left edge of box
        # max(0, ...): Ensure box doesn't go outside image (left boundary)
        x_min = max(0, center_x - size // 2)
        # center_y - size // 2: Top edge of box
        # max(0, ...): Ensure box doesn't go outside image (top boundary)
        y_min = max(0, center_y - size // 2)
        # center_x + size // 2: Right edge of box
        # min(img_size, ...): Ensure box doesn't go outside image (right boundary)
        x_max = min(img_size, center_x + size // 2)
        # center_y + size // 2: Bottom edge of box
        # min(img_size, ...): Ensure box doesn't go outside image (bottom boundary)
        y_max = min(img_size, center_y + size // 2)
        
        # Draw object on image
        if obj_type == 0:  # Square
            # Draw filled square
            # img[:, y_min:y_max, x_min:x_max]: All channels, rows y_min to y_max, cols x_min to x_max
            # = 1.0: Set all pixels in box to white (1.0)
            img[:, y_min:y_max, x_min:x_max] = 1.0
        else:  # Circle
            # Create coordinate grids for circle
            # np.ogrid: Creates open grids for efficient array operations
            # y_coords: Row indices (0 to img_size-1)
            # x_coords: Column indices (0 to img_size-1)
            y_coords, x_coords = np.ogrid[:img_size, :img_size]
            
            # Create circular mask
            # (x_coords - center_x)**2 + (y_coords - center_y)**2: Distance squared from center
            # <= (size//2)**2: Points within radius size//2
            # mask: Boolean array, True for points inside circle
            mask = ((x_coords - center_x)**2 + (y_coords - center_y)**2) <= (size//2)**2
            
            # Draw filled circle
            # img[:, mask]: All channels, pixels where mask is True
            # = 1.0: Set all pixels in circle to white (1.0)
            img[:, mask] = 1.0
        
        # Store image, bounding box, and label
        images.append(img)
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(obj_type)
    
    # Convert lists to NumPy arrays
    return np.array(images), np.array(boxes), np.array(labels)

X, boxes, labels = create_detection_data(num_samples=200, img_size=128)
print(f"Created {len(X)} images with objects")
print(f"Bounding boxes: {len(boxes)}")
print(f"Classes: {len(np.unique(labels))} (0=Square, 1=Circle)")
print()

# ============================================================================
# Step 2: Build Detection Model
# ============================================================================
print("=" * 70)
print("Step 2: Building Object Detection Model")
print("=" * 70)
print()

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

# ============================================================================
# Step 3: Loss Function
# ============================================================================
print("=" * 70)
print("Step 3: Defining Detection Loss")
print("=" * 70)
print()

def detection_loss(bbox_pred, bbox_true, class_pred, class_true, lambda_box=1.0):
    """Combined loss for detection"""
    # ===== BOUNDING BOX LOSS =====
    # Smooth L1 Loss: Less sensitive to outliers than L2, smoother than L1
    # nn.SmoothL1Loss(): PyTorch smooth L1 loss function
    # Formula: 0.5 × (x)² if |x| < 1, else |x| - 0.5
    # bbox_pred: Predicted bounding boxes (batch, 4) - [x_min, y_min, x_max, y_max]
    # bbox_true: True bounding boxes (batch, 4)
    # Measures how close predicted boxes are to true boxes
    bbox_loss = nn.SmoothL1Loss()(bbox_pred, bbox_true)
    
    # ===== CLASSIFICATION LOSS =====
    # Cross-Entropy Loss: Standard loss for multi-class classification
    # nn.CrossEntropyLoss(): PyTorch cross-entropy loss
    # class_pred: Predicted class logits (batch, num_classes)
    # class_true: True class labels (batch,) - integer class indices
    # Measures how well model classifies objects
    class_loss = nn.CrossEntropyLoss()(class_pred, class_true)
    
    # ===== COMBINED LOSS =====
    # Total loss: Weighted combination of both losses
    # lambda_box: Weight for bounding box loss (controls relative importance)
    # lambda_box=1.0: Equal weight to both losses
    # Higher lambda_box = care more about box accuracy
    # Lower lambda_box = care more about classification accuracy
    total_loss = lambda_box * bbox_loss + class_loss
    
    # Return all losses for monitoring
    return total_loss, bbox_loss, class_loss

# ============================================================================
# Step 4: Training
# ============================================================================
print("=" * 70)
print("Step 4: Training Detection Model")
print("=" * 70)
print()

model = SimpleDetector(num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Normalize boxes to [0, 1]
boxes_normalized = boxes / 128.0
X_tensor = torch.FloatTensor(X)
boxes_tensor = torch.FloatTensor(boxes_normalized)
labels_tensor = torch.LongTensor(labels)

epochs = 100
print("Training...")
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
        print(f"  Epoch {epoch+1}/{epochs}: Total={loss.item():.4f}, "
              f"BBox={bbox_loss.item():.4f}, Class={class_loss.item():.4f}")

print()

# ============================================================================
# Step 5: Visualization
# ============================================================================
print("=" * 70)
print("Step 5: Visualizing Detection Results")
print("=" * 70)
print()

def visualize_detection(img, bbox_pred, bbox_true, class_pred, class_true, img_size=128):
    """Visualize detection results"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Denormalize boxes
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
    axes[0].axis('off')
    
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
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig

# Visualize first few images
model.eval()
with torch.no_grad():
    bbox_pred, class_pred = model(X_tensor[:5])
    class_pred_idx = torch.argmax(class_pred, dim=1)
    
    for i in range(3):
        fig = visualize_detection(
            X[i], 
            bbox_pred[i].numpy(), 
            boxes_normalized[i],
            class_pred_idx[i].item(), 
            labels[i]
        )
        plt.savefig(f'detection_example_{i}.png', dpi=150, bbox_inches='tight')
        plt.close()

print("Saved detection visualizations")
print()

# Calculate IoU (Intersection over Union)
def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou

# Evaluate on test set
model.eval()
with torch.no_grad():
    bbox_pred, class_pred = model(X_tensor)
    class_pred_idx = torch.argmax(class_pred, dim=1)
    
    # Calculate IoU
    ious = []
    for i in range(len(X)):
        pred_box = bbox_pred[i].numpy() * 128
        true_box = boxes[i]
        iou = calculate_iou(pred_box, true_box)
        ious.append(iou)
    
    mean_iou = np.mean(ious)
    accuracy = (class_pred_idx == labels_tensor).float().mean().item()

print(f"Detection Performance:")
print(f"  Mean IoU: {mean_iou:.4f}")
print(f"  Classification Accuracy: {accuracy:.2%}")
print()

print("=" * 70)
print("Project 8d Complete!")
print("=" * 70)
print()
print("Summary:")
print("  ✅ Created object detection dataset")
print("  ✅ Built detection model")
print("  ✅ Trained on bounding boxes and classes")
print("  ✅ Visualized detection results")
print("  ✅ Evaluated with IoU metric")
print()
