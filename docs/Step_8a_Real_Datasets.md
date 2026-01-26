# Step 8a — Real Datasets (CIFAR-10, ImageNet)

> **Goal:** Train CNNs on real-world image datasets.  
> **Tools:** Python + PyTorch + torchvision

---

## 8a.1 About CIFAR-10

- 60,000 color images
- 32×32 pixels
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 50,000 training, 10,000 test

---

## 8a.2 About ImageNet

- 14 million images
- 1,000 classes
- 224×224 pixels (standard)
- Used for ImageNet Challenge

---

## 8a.3 Using Real Datasets

```python
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
```

---

## 8a.4 Data Augmentation

Common augmentations:
- Random horizontal flip
- Random rotation
- Color jitter
- Random crop
- Normalization

Benefits:
- More training data
- Better generalization
- Reduces overfitting
