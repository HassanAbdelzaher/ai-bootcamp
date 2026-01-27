"""
Step 8e — Image Generation (GANs, VAEs)
Goal: Introduction to generative models for creating images.
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

print("=== Step 8e: Image Generation (GANs, VAEs) ===")
print("Introduction to generative models")
print()

# 8e.1 Generative vs Discriminative Models
print("=== 8e.1 Generative vs Discriminative Models ===")
print("Discriminative Models (what we've learned):")
print("  - Learn P(y|x): probability of class given image")
print("  - Classify existing images")
print("  - Examples: CNNs, classifiers")
print()
print("Generative Models:")
print("  - Learn P(x): probability distribution of images")
print("  - Create new images")
print("  - Examples: GANs, VAEs, Diffusion models")
print()

# 8e.2 What are GANs?
print("=== 8e.2 What are GANs? ===")
print("GAN = Generative Adversarial Network")
print()
print("Two networks compete:")
print("  1. Generator: Creates fake images")
print("  2. Discriminator: Distinguishes real from fake")
print()
print("Training process:")
print("  - Generator tries to fool discriminator")
print("  - Discriminator tries to catch fakes")
print("  - Both improve together")
print("  - Result: Generator creates realistic images")
print()

# 8e.3 Simple Generator
print("=== 8e.3 Simple Generator ===")
class SimpleGenerator(nn.Module):
    """Simple generator network"""
    def __init__(self, latent_dim=100, img_channels=1, img_size=64):
        super(SimpleGenerator, self).__init__()
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
        # z shape: (batch, 100) - random vector (latent code)
        # This is the "seed" that generates the image
        
        # Step 1: Expand noise to larger feature map
        # self.fc(z): Fully connected layer expands noise
        # Input: (batch, 100) → Output: (batch, 256*4*4) = (batch, 4096)
        x = self.fc(z)
        
        # Step 2: Reshape to image-like tensor
        # x.view(x.size(0), 256, 4, 4): Reshape to (batch, 256, 4, 4)
        # x.size(0): Batch size
        # 256: Number of channels
        # 4, 4: Height and width (small starting size)
        # This creates a small feature map that we'll upsample
        x = x.view(x.size(0), 256, 4, 4)
        
        # Step 3: Upsample through transposed convolutions
        # self.conv_layers: Series of ConvTranspose2d layers
        # Each layer doubles the size: 4x4 → 8x8 → 16x16 → 32x32 → 64x64
        # Result: (batch, img_channels, 64, 64) - generated image
        x = self.conv_layers(x)
        return x

generator = SimpleGenerator(latent_dim=100, img_channels=1, img_size=64)
print("Generator architecture:")
print(generator)
print()

# Test generator
z = torch.randn(1, 100)
fake_image = generator(z)
print(f"Generated image shape: {fake_image.shape}")
print()

# 8e.4 Simple Discriminator
print("=== 8e.4 Simple Discriminator ===")
class SimpleDiscriminator(nn.Module):
    """Simple discriminator network"""
    def __init__(self, img_channels=1, img_size=64):
        super(SimpleDiscriminator, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 8 * 8, 1),
            nn.Sigmoid()  # Probability: real or fake
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

discriminator = SimpleDiscriminator(img_channels=1, img_size=64)
print("Discriminator architecture:")
print(discriminator)
print()

# 8e.5 Create Training Data
print("=== 8e.5 Create Training Data ===")
def create_real_data(num_samples=100, img_size=64):
    """Create simple real images"""
    images = []
    for _ in range(num_samples):
        img = np.zeros((1, img_size, img_size), dtype=np.float32)
        # Add simple pattern
        center = img_size // 2
        y_coords, x_coords = np.ogrid[:img_size, :img_size]
        mask = (x_coords - center)**2 + (y_coords - center)**2 <= (img_size//4)**2
        img[0][mask] = 1.0
        images.append(img)
    return np.array(images)

real_images = create_real_data(num_samples=200, img_size=64)
real_tensor = torch.FloatTensor(real_images)
print(f"Created {len(real_images)} real images")
print()

# 8e.6 GAN Training (Simplified)
print("=== 8e.6 GAN Training (Simplified) ===")
print("Note: This is a simplified example")
print("Real GAN training is more complex and requires careful tuning")
print()

# Optimizers
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

loss_fn = nn.BCELoss()
g_losses = []
d_losses = []

num_epochs = 20
batch_size = 32
latent_dim = 100

print(f"Training GAN for {num_epochs} epochs...")
for epoch in range(num_epochs):
    epoch_g_loss = 0
    epoch_d_loss = 0
    
    for i in range(0, len(real_tensor), batch_size):
        batch_real = real_tensor[i:i+batch_size]
        batch_size_actual = batch_real.size(0)
        
        # Train Discriminator
        d_optimizer.zero_grad()
        
        # Real images
        real_labels = torch.ones(batch_size_actual, 1)
        real_pred = discriminator(batch_real)
        d_loss_real = loss_fn(real_pred, real_labels)
        
        # Fake images
        z = torch.randn(batch_size_actual, latent_dim)
        fake_images = generator(z)
        fake_labels = torch.zeros(batch_size_actual, 1)
        fake_pred = discriminator(fake_images.detach())
        d_loss_fake = loss_fn(fake_pred, fake_labels)
        
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        d_optimizer.step()
        
        # Train Generator
        g_optimizer.zero_grad()
        z = torch.randn(batch_size_actual, latent_dim)
        fake_images = generator(z)
        fake_pred = discriminator(fake_images)
        # Generator wants discriminator to think fakes are real
        g_loss = loss_fn(fake_pred, torch.ones(batch_size_actual, 1))
        g_loss.backward()
        g_optimizer.step()
        
        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()
    
    g_losses.append(epoch_g_loss / (len(real_tensor) // batch_size))
    d_losses.append(epoch_d_loss / (len(real_tensor) // batch_size))
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, G_Loss: {g_losses[-1]:.4f}, D_Loss: {d_losses[-1]:.4f}")

print("Training complete!")
print()

# 8e.7 Learning Curves
print("=== 8e.7 Learning Curves ===")
plot_learning_curve(g_losses, title="Generator Loss", ylabel="Loss")
plot_learning_curve(d_losses, title="Discriminator Loss", ylabel="Loss")

# 8e.8 Generate Sample Images
print("=== 8e.8 Generate Sample Images ===")
generator.eval()
with torch.no_grad():
    z = torch.randn(5, latent_dim)
    generated = generator(z)
    print(f"Generated {len(generated)} images")
    print(f"Image shape: {generated[0].shape}")
print()

# 8e.9 What are VAEs?
print("=== 8e.9 What are VAEs? ===")
print("VAE = Variational Autoencoder")
print()
print("Architecture:")
print("  - Encoder: Image → Latent code")
print("  - Decoder: Latent code → Image")
print()
print("Key difference from GAN:")
print("  - VAE: Learns to reconstruct + generate")
print("  - GAN: Learns to generate realistic images")
print()
print("Benefits:")
print("  ✅ More stable training than GANs")
print("  ✅ Can interpolate in latent space")
print("  ✅ Explicit latent representation")
print()

# 8e.10 Simple VAE
print("=== 8e.10 Simple VAE ===")
class SimpleVAE(nn.Module):
    """Simple Variational Autoencoder"""
    def __init__(self, latent_dim=20):
        super(SimpleVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(64*64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        
        # Latent space (mean and log variance)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 64*64),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(x.size(0), -1))
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

vae = SimpleVAE(latent_dim=20)
print("VAE architecture:")
print(vae)
print()

# 8e.11 VAE Training
print("=== 8e.11 VAE Training ===")
def vae_loss(recon_x, x, mu, logvar):
    """VAE loss: reconstruction + KL divergence"""
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x.view(x.size(0), -1), reduction='sum')
    
    # KL divergence (regularization)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss

print("VAE loss combines:")
print("  - Reconstruction: How well image is reconstructed")
print("  - KL divergence: Regularize latent space")
print()

# 8e.12 GAN vs VAE
print("=== 8e.12 GAN vs VAE ===")
print("GAN:")
print("  ✅ More realistic images")
print("  ✅ State-of-the-art quality")
print("  ❌ Harder to train")
print("  ❌ Mode collapse problem")
print()
print("VAE:")
print("  ✅ More stable training")
print("  ✅ Interpretable latent space")
print("  ✅ Can interpolate")
print("  ❌ Blurrier images")
print()

# 8e.13 Modern Generative Models
print("=== 8e.13 Modern Generative Models ===")
print("Diffusion Models (2020s):")
print("  - DALL-E, Midjourney, Stable Diffusion")
print("  - Generate images from text")
print("  - Very high quality")
print()
print("Transformers for Images:")
print("  - DALL-E uses transformer")
print("  - Generate images autoregressively")
print()

# 8e.14 Applications
print("=== 8e.14 Applications ===")
print("Image generation is used for:")
print("  🎨 Art creation")
print("  🎮 Game assets")
print("  📸 Photo editing")
print("  🎬 Visual effects")
print("  🏥 Medical imaging")
print("  🔬 Scientific visualization")
print()

# 8e.15 Next Steps
print("=== 8e.15 Next Steps ===")
print("You've learned:")
print("  ✅ What generative models are")
print("  ✅ How GANs work (generator vs discriminator)")
print("  ✅ How VAEs work (encoder-decoder)")
print("  ✅ Differences between GANs and VAEs")
print()
print("Try these next:")
print("  - Train GAN on real dataset")
print("  - Use pre-trained models (StyleGAN)")
print("  - Explore diffusion models")
print("  - Try text-to-image generation")
print()

print("🎉 Image generation learning complete!")
print("You now understand how AI creates images!")
