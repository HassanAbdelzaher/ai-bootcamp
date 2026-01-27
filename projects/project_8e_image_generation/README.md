# Project 8e: Image Generation (GANs, VAEs)

> **Generate new images using Generative Adversarial Networks and Variational Autoencoders**

**Difficulty**: ⭐⭐⭐⭐ Expert  
**Time**: 6-8 hours  
**Prerequisites**: Steps 0-8 (Especially Step 8: CNNs)

---

## 📚 Table of Contents

1. [Project Overview](#project-overview)
2. [Problem: Generate Images](#problem-generate-images)
3. [Key Concepts](#key-concepts)
4. [Step-by-Step Guide](#step-by-step-guide)
5. [Expected Results](#expected-results)
6. [Extension Ideas](#extension-ideas)

---

## 🎯 Project Overview

This project teaches you to generate new images using generative models. You'll learn to:

- Build Generative Adversarial Networks (GANs)
- Implement Variational Autoencoders (VAEs)
- Generate realistic images
- Understand generative vs discriminative models

### Why Image Generation?

- **Creative AI**: Generate art, faces, scenes
- **Data augmentation**: Create training data
- **Understanding**: Learn data distribution
- **Cutting-edge**: State-of-the-art research area

---

## 📋 Problem: Generate Images

### Task

Build generative models to create new images:
1. **GAN**: Generator vs Discriminator competition
2. **VAE**: Encoder-Decoder with latent space
3. **Image Quality**: Generate realistic images
4. **Conditional Generation**: Generate specific classes

### Learning Objectives

- Understand generative models
- Build GAN architecture
- Implement VAE
- Generate and evaluate images

---

## 🧠 Key Concepts

### 1. Generative vs Discriminative

**Discriminative**: P(y|x) - Classify existing data
**Generative**: P(x) - Learn data distribution, create new data

### 2. GAN Architecture

**Generator**: Creates fake images from noise
**Discriminator**: Distinguishes real from fake
**Training**: Adversarial process - both improve together

### 3. VAE Architecture

**Encoder**: Maps images to latent space
**Decoder**: Maps latent space to images
**Latent Space**: Low-dimensional representation

---

## 🚀 Step-by-Step Guide

### Step 1: Build Generator

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=1, img_size=64):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        
        # Start from random noise
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # Upsample to image
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, img_channels, 4, stride=2, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, z):
        # z: random noise (batch, latent_dim)
        x = self.fc(z)
        x = x.view(x.size(0), 256, 4, 4)
        x = self.conv_layers(x)
        return x
```

### Step 2: Build Discriminator

```python
class Discriminator(nn.Module):
    def __init__(self, img_channels=1, img_size=64):
        super(Discriminator, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, stride=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.conv_layers(x)
```

### Step 3: GAN Training

```python
def train_gan(generator, discriminator, dataloader, epochs=100):
    """Train GAN"""
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    criterion = nn.BCELoss()
    latent_dim = 100
    
    for epoch in range(epochs):
        for batch_idx, real_images in enumerate(dataloader):
            batch_size = real_images.size(0)
            
            # Train Discriminator
            discriminator.zero_grad()
            
            # Real images
            real_labels = torch.ones(batch_size, 1)
            real_output = discriminator(real_images)
            d_loss_real = criterion(real_output, real_labels)
            
            # Fake images
            noise = torch.randn(batch_size, latent_dim)
            fake_images = generator(noise)
            fake_labels = torch.zeros(batch_size, 1)
            fake_output = discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_output, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            generator.zero_grad()
            fake_labels = torch.ones(batch_size, 1)  # Trick discriminator
            fake_output = discriminator(fake_images)
            g_loss = criterion(fake_output, fake_labels)
            g_loss.backward()
            g_optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: "
                      f"D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}")
```

### Step 4: VAE Implementation

```python
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 128 * 8 * 8)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 128, 8, 8)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
```

### Step 5: VAE Training

```python
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss: reconstruction + KL divergence"""
    recon_loss = nn.BCELoss()(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / x.size(0)  # Average
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

# Training loop
vae = VAE(latent_dim=20)
optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)

for epoch in range(epochs):
    for batch_images in dataloader:
        optimizer.zero_grad()
        recon, mu, logvar = vae(batch_images)
        loss, recon_loss, kl_loss = vae_loss(recon, batch_images, mu, logvar)
        loss.backward()
        optimizer.step()
```

---

## 📊 Expected Results

### GAN Training

```
Epoch 0: D_loss=1.2345, G_loss=2.3456
Epoch 20: D_loss=0.5234, G_loss=1.1234
Epoch 50: D_loss=0.3456, G_loss=0.8765
Epoch 100: D_loss=0.2345, G_loss=0.5678
```

### Generated Images

- GAN: Realistic but may have artifacts
- VAE: Smoother but less sharp
- Both improve with training

---

## 💡 Extension Ideas

1. **Conditional GANs**
   - Generate specific classes
   - Control generation
   - Class-conditional generation

2. **Advanced GANs**
   - DCGAN improvements
   - Progressive GAN
   - StyleGAN concepts

3. **Evaluation Metrics**
   - FID score
   - IS (Inception Score)
   - Visual quality assessment

---

## ✅ Success Criteria

- ✅ Build working GAN
- ✅ Implement VAE
- ✅ Generate recognizable images
- ✅ Understand generative models
- ✅ Compare GAN vs VAE

---

**Ready to generate images? Let's create with AI!** 🚀
