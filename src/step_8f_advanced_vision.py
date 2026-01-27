"""
Step 8f — Advanced Vision Tasks
Goal: Learn advanced computer vision tasks beyond classification
Tools: Python + PyTorch + NumPy + Matplotlib
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', message='.*Failed to initialize NumPy.*')

print("=" * 70)
print("Step 8f: Advanced Vision Tasks")
print("=" * 70)
print()
print("Goal: Learn advanced computer vision tasks beyond classification")
print()

# ============================================================================
# 8f.1 Beyond Classification
# ============================================================================
print("=== 8f.1 Beyond Classification ===")
print()
print("Computer vision goes beyond just classifying images:")
print("  • Image Segmentation: Identify objects pixel by pixel")
print("  • Style Transfer: Apply artistic styles to images")
print("  • Super-Resolution: Enhance image quality")
print("  • Depth Estimation: Estimate 3D depth from 2D images")
print("  • Video Processing: Analyze temporal sequences")
print()

# ============================================================================
# 8f.2 Image Segmentation
# ============================================================================
print("=== 8f.2 Image Segmentation ===")
print()
print("Segmentation: Classify each pixel in an image")
print()
print("Types of Segmentation:")
print("  1. Semantic Segmentation:")
print("     • Classify pixels into categories")
print("     • 'All sky pixels are sky'")
print("     • Doesn't distinguish between instances")
print()
print("  2. Instance Segmentation:")
print("     • Identify individual objects")
print("     • 'This car vs that car'")
print("     • Combines detection + segmentation")
print()
print("  3. Panoptic Segmentation:")
print("     • Combines semantic + instance")
print("     • Every pixel belongs to one instance or background")
print()

# Create synthetic image for segmentation demo
np.random.seed(42)
img_size = 32

# Create simple image with different regions
image = np.zeros((img_size, img_size, 3))
# Sky region (top)
image[:img_size//3, :, 0] = 0.5  # Blue
image[:img_size//3, :, 2] = 0.8
# Ground region (bottom)
image[img_size//3:, :, 1] = 0.6  # Green
# Object (circle in middle)
center = img_size // 2
radius = img_size // 6
y, x = np.ogrid[:img_size, :img_size]
mask = (x - center)**2 + (y - center)**2 <= radius**2
image[mask, 0] = 1.0  # Red object
image[mask, 1] = 0.0
image[mask, 2] = 0.0

# Create segmentation mask
segmentation_mask = np.zeros((img_size, img_size))
segmentation_mask[:img_size//3, :] = 0  # Sky = 0
segmentation_mask[img_size//3:, :] = 1  # Ground = 1
segmentation_mask[mask] = 2  # Object = 2

print("Semantic Segmentation Example:")
print("  Classes: 0=Sky, 1=Ground, 2=Object")
print()

# Visualize segmentation
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image)
axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(segmentation_mask, cmap='viridis')
axes[1].set_title('Segmentation Mask', fontsize=12, fontweight='bold')
axes[1].axis('off')

# Overlay
overlay = image.copy()
overlay[:, :, 0] = np.where(segmentation_mask == 0, 0.3, overlay[:, :, 0])
overlay[:, :, 1] = np.where(segmentation_mask == 1, 0.3, overlay[:, :, 1])
overlay[:, :, 2] = np.where(segmentation_mask == 2, 0.3, overlay[:, :, 2])
axes[2].imshow(overlay)
axes[2].set_title('Segmentation Overlay', fontsize=12, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.show()

# Simple segmentation model (U-Net inspired)
class SimpleSegmentationModel(nn.Module):
    """Simple segmentation model using encoder-decoder"""
    def __init__(self, num_classes=3):
        super().__init__()
        # Encoder (downsampling)
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)
        
        # Decoder (upsampling)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU()
        )
        
        self.up2 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, num_classes, 3, padding=1)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # Decoder
        u1 = self.up1(p2)
        # Skip connection (concatenate)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)
        
        u2 = self.up2(d1)
        d2 = self.dec2(u2)
        
        return d2

print("Segmentation Model Architecture:")
print("  Encoder: Extract features (downsample)")
print("  Decoder: Reconstruct segmentation (upsample)")
print("  Skip Connections: Preserve spatial information")
print()

# ============================================================================
# 8f.3 Style Transfer
# ============================================================================
print("=== 8f.3 Style Transfer ===")
print()
print("Style Transfer: Apply artistic style from one image to another")
print()
print("Key Idea:")
print("  • Content image: What to show")
print("  • Style image: How to show it")
print("  • Generated image: Content with style")
print()
print("How it works:")
print("  1. Use pre-trained CNN (VGG) to extract features")
print("  2. Content loss: Match content image features")
print("  3. Style loss: Match style image statistics")
print("  4. Optimize generated image to minimize both losses")
print()

# Create simple content and style images
content_img = np.random.rand(32, 32, 3) * 0.5 + 0.25  # Gray content
style_img = np.random.rand(32, 32, 3)  # Colorful style

# Simple style transfer visualization (conceptual)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(content_img)
axes[0].set_title('Content Image', fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(style_img)
axes[1].set_title('Style Image', fontsize=12, fontweight='bold')
axes[1].axis('off')

# Simple style transfer (just for visualization)
generated = content_img * 0.6 + style_img * 0.4
axes[2].imshow(generated)
axes[2].set_title('Generated (Style Transfer)', fontsize=12, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.show()

print("Real Style Transfer:")
print("  • Uses Gram matrix for style representation")
print("  • Iterative optimization")
print("  • Can take minutes to generate")
print("  • Popular: Neural Style Transfer (Gatys et al.)")
print()

# ============================================================================
# 8f.4 Image Super-Resolution
# ============================================================================
print("=== 8f.4 Image Super-Resolution ===")
print()
print("Super-Resolution: Enhance image quality and resolution")
print()
print("Applications:")
print("  • Upscale low-resolution images")
print("  • Enhance old photos")
print("  • Improve medical imaging")
print("  • Enhance satellite imagery")
print()

# Create low-resolution image
lr_size = 16
hr_size = 32

# Low-res image (downsampled)
lr_image = np.random.rand(lr_size, lr_size, 3)

# High-res target (for training)
hr_image = np.random.rand(hr_size, hr_size, 3)

print("Super-Resolution Task:")
print(f"  Input: {lr_size}×{lr_size} low-resolution image")
print(f"  Output: {hr_size}×{hr_size} high-resolution image")
print()

# Simple super-resolution model
class SuperResolutionModel(nn.Module):
    """Simple super-resolution model"""
    def __init__(self, scale_factor=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample = nn.ConvTranspose2d(64, 3, scale_factor, stride=scale_factor)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.upsample(x)
        return x

print("Super-Resolution Methods:")
print("  • SRCNN: First CNN-based super-resolution")
print("  • SRGAN: GAN-based, produces realistic details")
print("  • EDSR: Enhanced deep super-resolution")
print("  • Real-ESRGAN: State-of-the-art for real images")
print()

# Visualize super-resolution
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(lr_image)
axes[0].set_title(f'Low-Resolution ({lr_size}×{lr_size})', fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(hr_image)
axes[1].set_title(f'High-Resolution ({hr_size}×{hr_size})', fontsize=12, fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.show()

# ============================================================================
# 8f.5 Depth Estimation
# ============================================================================
print("=== 8f.5 Depth Estimation ===")
print()
print("Depth Estimation: Estimate 3D depth from 2D images")
print()
print("Applications:")
print("  • Autonomous vehicles")
print("  • Robotics navigation")
print("  • 3D reconstruction")
print("  • Augmented reality")
print()

# Create synthetic depth map
depth_img = np.zeros((32, 32))
# Create depth gradient (closer = brighter)
for i in range(32):
    for j in range(32):
        # Distance from center
        dist = np.sqrt((i - 16)**2 + (j - 16)**2)
        depth_img[i, j] = 1.0 - (dist / 32.0)  # Closer = higher value

print("Depth Map:")
print("  • Each pixel = distance from camera")
print("  • Closer objects = brighter")
print("  • Farther objects = darker")
print()

# Visualize depth estimation
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# RGB image
rgb_image = np.random.rand(32, 32, 3)
axes[0].imshow(rgb_image)
axes[0].set_title('RGB Image', fontsize=12, fontweight='bold')
axes[0].axis('off')

# Depth map
im = axes[1].imshow(depth_img, cmap='viridis')
axes[1].set_title('Depth Map (Estimated)', fontsize=12, fontweight='bold')
axes[1].axis('off')
plt.colorbar(im, ax=axes[1], label='Depth')

plt.tight_layout()
plt.show()

# Simple depth estimation model
class DepthEstimationModel(nn.Module):
    """Simple depth estimation model"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),  # Single channel for depth
            nn.Sigmoid()  # Normalize to [0, 1]
        )
    
    def forward(self, x):
        features = self.encoder(x)
        depth = self.decoder(features)
        return depth

print("Depth Estimation Methods:")
print("  • Monocular: Single image (harder)")
print("  • Stereo: Two images (easier, more accurate)")
print("  • Structured Light: Projected patterns")
print("  • LiDAR: Direct depth measurement")
print()

# ============================================================================
# 8f.6 Video Processing
# ============================================================================
print("=== 8f.6 Video Processing ===")
print()
print("Video = Sequence of images (frames)")
print()
print("Video Tasks:")
print("  • Action Recognition: What action is happening?")
print("  • Object Tracking: Follow objects across frames")
print("  • Video Segmentation: Segment objects in video")
print("  • Video Generation: Create new videos")
print()

# Create simple video (sequence of frames)
num_frames = 5
video_frames = []

# Moving object across frames
for frame_idx in range(num_frames):
    frame = np.zeros((32, 32, 3))
    # Object position changes
    obj_x = frame_idx * 5
    obj_y = 16
    # Draw object
    if obj_x < 32:
        frame[obj_y-2:obj_y+2, obj_x-2:obj_x+2, 0] = 1.0  # Red object
    video_frames.append(frame)

print(f"Video: {num_frames} frames, {32}×{32} pixels each")
print("  Object moves from left to right across frames")
print()

# Visualize video frames
fig, axes = plt.subplots(1, num_frames, figsize=(15, 3))
for i, frame in enumerate(video_frames):
    axes[i].imshow(frame)
    axes[i].set_title(f'Frame {i+1}', fontsize=10, fontweight='bold')
    axes[i].axis('off')
plt.tight_layout()
plt.show()

# 3D CNN for video (conceptual)
class VideoCNN(nn.Module):
    """3D CNN for video processing"""
    def __init__(self, num_classes=10):
        super().__init__()
        # 3D convolutions (spatial + temporal)
        self.conv3d1 = nn.Conv3d(3, 32, (3, 3, 3), padding=1)
        self.conv3d2 = nn.Conv3d(32, 64, (3, 3, 3), padding=1)
        self.pool3d = nn.MaxPool3d(2)
        self.fc = nn.Linear(64 * 4 * 4 * 2, num_classes)  # Simplified
    
    def forward(self, x):
        # x shape: (batch, channels, depth, height, width)
        x = F.relu(self.conv3d1(x))
        x = self.pool3d(x)
        x = F.relu(self.conv3d2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

print("Video Processing Approaches:")
print("  • 3D CNNs: Convolve over space and time")
print("  • Two-Stream: RGB + Optical Flow")
print("  • 3D ResNet: Deep video networks")
print("  • Temporal Convolutions: 1D over time")
print()

# ============================================================================
# 8f.7 Comparison of Vision Tasks
# ============================================================================
print("=== 8f.7 Comparison of Vision Tasks ===")
print()

tasks = {
    'Classification': {
        'Input': 'Image',
        'Output': 'Class label',
        'Example': 'Cat vs Dog',
        'Difficulty': 'Easy'
    },
    'Object Detection': {
        'Input': 'Image',
        'Output': 'Bounding boxes + labels',
        'Example': 'Find all cars in image',
        'Difficulty': 'Medium'
    },
    'Segmentation': {
        'Input': 'Image',
        'Output': 'Pixel-wise labels',
        'Example': 'Segment all objects',
        'Difficulty': 'Hard'
    },
    'Style Transfer': {
        'Input': 'Content + Style images',
        'Output': 'Styled image',
        'Example': 'Photo in Van Gogh style',
        'Difficulty': 'Medium'
    },
    'Super-Resolution': {
        'Input': 'Low-res image',
        'Output': 'High-res image',
        'Example': 'Upscale 4x',
        'Difficulty': 'Medium'
    },
    'Depth Estimation': {
        'Input': 'RGB image',
        'Output': 'Depth map',
        'Example': '3D scene understanding',
        'Difficulty': 'Hard'
    }
}

print("Vision Tasks Overview:")
print("-" * 70)
print(f"{'Task':<20} {'Input':<25} {'Output':<20} {'Difficulty':<10}")
print("-" * 70)
for task, info in tasks.items():
    print(f"{task:<20} {info['Input']:<25} {info['Output']:<20} {info['Difficulty']:<10}")
print()

# ============================================================================
# 8f.8 Real-World Applications
# ============================================================================
print("=== 8f.8 Real-World Applications ===")
print()
print("Advanced vision tasks power many applications:")
print()
print("🚗 Autonomous Vehicles:")
print("   • Semantic segmentation: Road, vehicles, pedestrians")
print("   • Depth estimation: Distance to objects")
print("   • Object detection: Identify obstacles")
print()
print("🏥 Medical Imaging:")
print("   • Segmentation: Organ boundaries")
print("   • Super-resolution: Enhance scans")
print("   • Detection: Find anomalies")
print()
print("📱 Augmented Reality:")
print("   • Depth estimation: 3D understanding")
print("   • Object tracking: Follow objects")
print("   • Scene understanding: Real-time processing")
print()
print("🎨 Creative Applications:")
print("   • Style transfer: Artistic filters")
print("   • Super-resolution: Enhance photos")
print("   • Video editing: Automatic effects")
print()
print("🛡️ Security & Surveillance:")
print("   • Person tracking: Follow individuals")
print("   • Action recognition: Detect behaviors")
print("   • Anomaly detection: Unusual events")
print()

# ============================================================================
# 8f.9 Key Architectures
# ============================================================================
print("=== 8f.9 Key Architectures ===")
print()
print("Important architectures for advanced vision:")
print()
print("Segmentation:")
print("  • U-Net: Encoder-decoder with skip connections")
print("  • DeepLab: Atrous convolutions, ASPP")
print("  • Mask R-CNN: Instance segmentation")
print()
print("Super-Resolution:")
print("  • SRCNN: First CNN-based")
print("  • SRGAN: GAN-based")
print("  • EDSR: Enhanced deep super-resolution")
print()
print("Depth Estimation:")
print("  • Monodepth: Monocular depth estimation")
print("  • DORN: Deep ordinal regression")
print("  • MiDaS: Mixed dataset training")
print()
print("Video:")
print("  • 3D ResNet: Deep video networks")
print("  • I3D: Inflated 3D ConvNet")
print("  • SlowFast: Two-pathway architecture")
print()

# ============================================================================
# 8f.10 Challenges
# ============================================================================
print("=== 8f.10 Challenges in Advanced Vision ===")
print()
print("Each task has unique challenges:")
print()
print("Segmentation:")
print("  • Precise boundaries")
print("  • Multiple scales")
print("  • Class imbalance")
print()
print("Style Transfer:")
print("  • Balancing content vs style")
print("  • Slow generation")
print("  • Artifact removal")
print()
print("Super-Resolution:")
print("  • Realistic details")
print("  • Avoiding artifacts")
print("  • Generalization")
print()
print("Depth Estimation:")
print("  • Monocular ambiguity")
print("  • Scale uncertainty")
print("  • Textureless regions")
print()
print("Video:")
print("  • Temporal consistency")
print("  • Computational cost")
print("  • Long-term dependencies")
print()

# ============================================================================
# 8f.11 Summary
# ============================================================================
print("=== 8f.11 Summary ===")
print()
print("✅ You've learned:")
print("  • Image segmentation (semantic, instance)")
print("  • Style transfer concepts")
print("  • Super-resolution")
print("  • Depth estimation")
print("  • Video processing basics")
print()
print("🎯 Key Takeaways:")
print("  1. Vision goes beyond classification")
print("  2. Each task needs specialized architectures")
print("  3. Encoder-decoder common for dense prediction")
print("  4. Temporal information important for video")
print("  5. Many real-world applications")
print()

print("=" * 70)
print("Step 8f Complete! You understand advanced vision tasks!")
print("=" * 70)
