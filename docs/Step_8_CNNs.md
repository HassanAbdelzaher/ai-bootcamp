# Step 8 — CNNs (Convolutional Neural Networks for Images)

> **Goal:** Learn how to process images using Convolutional Neural Networks.  
> **Tools:** Python + PyTorch + NumPy + Matplotlib

---

## 8.1 Big Idea: Why CNNs for Images?

Regular neural networks:
- Treat images as **flat vectors**
- Lose **spatial structure**
- Need **too many parameters**

**CNNs (Convolutional Neural Networks):**
- Preserve **spatial structure**
- Detect **patterns** (edges, shapes, objects)
- **Fewer parameters** through weight sharing
- Perfect for: images, videos, medical scans

🧠 **Key insight:**  
CNNs understand that nearby pixels are related, just like words in a sentence.

---

## 8.2 The Problem with Regular Networks

Imagine classifying a 28×28 image:

**Regular network:**
- Flattens to 784 numbers
- Treats pixel 1 and pixel 784 the same
- Needs 784 × 100 = 78,400 weights for first layer

**CNN:**
- Keeps 2D structure
- Uses small filters (3×3 or 5×5)
- Shares weights across image
- Needs only 3×3×16 = 144 weights for first layer

---

## 8.3 Understanding Images as Data

Images are **2D arrays of pixels**:

```
Grayscale: (height, width)
Color:     (height, width, channels)
```

Example: 28×28 grayscale image
- 784 pixels total
- Each pixel: 0 (black) to 255 (white)
- Normalized: 0.0 to 1.0

---

## 8.4 What is Convolution?

**Convolution** = sliding a filter over the image

```
Image:           Filter:        Result:
[1 2 3]         [-1 0 1]       [Edge detected]
[4 5 6]    *    [-1 0 1]   =   [Pattern found]
[7 8 9]         [-1 0 1]       [Feature map]
```

The filter **detects patterns**:
- Edge detection
- Blurring
- Sharpening
- Feature extraction

---

## 8.5 Building a CNN in PyTorch

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 2)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

🧠 **Key components:**
- `Conv2d`: Convolutional layer
- `MaxPool2d`: Pooling layer
- `Linear`: Fully connected layer

---

## 8.5.1 Model Architecture Diagram

### Visualizing the Network Structure

Understanding the architecture of your CNN is crucial. A visual diagram helps you see how data flows through the network.

```python
from plotting import plot_model_architecture

model = SimpleCNN(num_classes=2)

# Visualize model architecture
plot_model_architecture(model, input_shape="(batch, 1, 28, 28)", 
                       title="SimpleCNN Architecture")
```

**Code Explanation:**
- `plot_model_architecture()`: Creates a visual diagram of the model
- `input_shape`: Shows the expected input shape
- `title`: Title for the diagram

**What you'll see:**
- **Layer flow**: Shows how data moves through each layer
- **Layer types**: Displays the type of each layer (Conv2d, MaxPool2d, Linear)
- **Parameter counts**: Shows how many parameters each layer has
- **Total parameters**: Overall model size

### Understanding the Diagram

**The diagram shows:**
1. **Input**: Images (batch, channels, height, width)
2. **Conv layers**: Extract features from images
3. **Pool layers**: Reduce spatial dimensions
4. **FC layers**: Final classification

**Key information:**
- **Layer names**: Identify each component
- **Layer types**: Understand what each layer does
- **Parameter counts**: See model complexity
- **Total parameters**: Overall model size

**Example output:**
```
Model Architecture:
  Input: (batch, 1, 28, 28)
  Conv1: 16 filters, 3x3
  Pool1: 2x2 max pooling
  Conv2: 32 filters, 3x3
  Pool2: 2x2 max pooling
  FC1: 64 neurons
  FC2: 2 classes
Total Parameters: ~50,000
```

### Why Visualize Architecture?

1. **Understanding flow**: See how data transforms through layers
2. **Debugging**: Identify where problems might occur
3. **Optimization**: Find bottlenecks or unnecessary layers
4. **Communication**: Explain model to others
5. **Documentation**: Keep track of model structure

---

## 8.6 CNN Architecture Components

### 1. Convolutional Layer (`Conv2d`)

**Purpose:** Detect patterns

**Parameters:**
- `in_channels`: Input channels (1 for grayscale, 3 for RGB)
- `out_channels`: Number of filters
- `kernel_size`: Filter size (3×3, 5×5, etc.)
- `padding`: Add zeros around edges

**What it does:**
- Applies filters across the image
- Each filter learns different patterns
- Output: Feature maps

### 2. Pooling Layer (`MaxPool2d`)

**Purpose:** Reduce size, increase robustness

**Parameters:**
- `kernel_size`: Pool size (usually 2×2)
- `stride`: Step size (usually 2)

**What it does:**
- Takes maximum value in each region
- Reduces image size by half
- Makes network translation-invariant

### 3. Fully Connected Layer (`Linear`)

**Purpose:** Final classification

**What it does:**
- Combines all learned features
- Outputs class probabilities

---

## 8.7 Data Shape for CNNs

CNNs need data in shape:

```
(batch_size, channels, height, width)
```

Example:
- Batch of 32 images
- 1 channel (grayscale) or 3 channels (RGB)
- 28×28 pixels each

Shape: `(32, 1, 28, 28)`

---

## 8.8 Training a CNN

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    predictions = model(X)
    loss = loss_fn(predictions, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Same training process, but now processing images!

---

## 8.9 How CNNs Learn Features

CNNs learn **hierarchical features**:

### Early Layers (Conv1, Conv2)
- Detect **simple patterns**
- Edges, lines, corners
- Basic shapes

### Middle Layers
- Detect **complex patterns**
- Shapes, textures
- Combinations of edges

### Late Layers (Fully Connected)
- Detect **high-level concepts**
- Objects, faces
- Complete scenes

---

## 8.10 Why CNNs Work So Well

### ✅ Translation Invariance
- Object in different positions → same detection
- Pooling helps with this

### ✅ Parameter Sharing
- Same filter used across entire image
- Much fewer parameters than fully connected
- Example: 3×3 filter vs 784×100 weights

### ✅ Local Connectivity
- Each neuron connects to small region
- Understands spatial relationships
- More efficient than global connections

---

## 8.11 Real-World Applications

### Image Classification
- **Cats vs Dogs**
- **Handwritten digits** (MNIST)
- **Object recognition** (ImageNet)

### Object Detection
- **Self-driving cars**: Detect pedestrians, cars, signs
- **Security systems**: Face recognition
- **Medical imaging**: Tumor detection

### Image Generation
- **Style transfer**: Make photos look like paintings
- **GANs**: Generate realistic images
- **Super-resolution**: Enhance image quality

### Other Applications
- **Video analysis**: Action recognition
- **Satellite imagery**: Land use classification
- **Medical diagnosis**: X-ray, MRI analysis

---

## 8.12 Famous CNN Architectures

### LeNet (1998)
- First successful CNN
- Handwritten digit recognition
- 7 layers

### AlexNet (2012)
- Deep learning breakthrough
- Won ImageNet competition
- 8 layers, 60M parameters

### VGG (2014)
- Very deep networks
- 16-19 layers
- Simple architecture

### ResNet (2015)
- Residual connections
- Solves vanishing gradient
- 50-152 layers

### Inception (2015)
- Multiple filter sizes
- Efficient computation
- Google's architecture

---

## 8.13 Transfer Learning

**Transfer Learning** = Use pre-trained models

Instead of training from scratch:
1. Start with model trained on ImageNet
2. Freeze early layers (keep learned features)
3. Train only final layers on your data

**Benefits:**
- Faster training
- Less data needed
- Better performance

---

## 8.14 Common CNN Patterns

### Typical Architecture:
```
Input Image
    ↓
Conv → ReLU → Pool
    ↓
Conv → ReLU → Pool
    ↓
Conv → ReLU → Pool
    ↓
Flatten
    ↓
FC → ReLU → Dropout
    ↓
FC → Output
```

### Design Principles:
- **Start small**: Few filters, small kernels
- **Go deeper**: More layers > larger layers
- **Pool regularly**: Reduce size gradually
- **Use dropout**: Prevent overfitting

---

## 8.15 Mini Exercises

### Exercise 1
Modify the CNN:
- Add more convolutional layers
- Change filter sizes
- Experiment with pooling

### Exercise 2
Train on different data:
- Create your own simple images
- Try different shapes/patterns
- Increase dataset size

### Exercise 3
Visualize filters:
- Print first layer weights
- See what patterns they detect
- Compare before/after training

---

## 8.16 Checklist (Before Moving On)

Students should understand:
- ✅ Why CNNs are needed for images
- ✅ How convolution works
- ✅ What pooling does
- ✅ How to build a CNN in PyTorch
- ✅ How CNNs learn hierarchical features
- ✅ Real-world applications

If YES → ready for advanced topics!

---

## 8.17 Next Step Preview

You've now learned:
- ✅ Feedforward networks (Steps 1-5)
- ✅ PyTorch framework (Step 6)
- ✅ Recurrent networks (Step 7)
- ✅ Convolutional networks (Step 8)

**What's next?**
- Advanced CNNs (ResNet, Inception)
- Object detection (YOLO, R-CNN)
- Image generation (GANs, VAEs)
- Transfer learning
- Real-world projects

---

## 🎉 Congratulations!

You've completed Step 8: CNNs!

You now understand:
- How AI processes images
- How to build CNNs
- Applications in computer vision

**You're becoming a real AI engineer!** 🚀
