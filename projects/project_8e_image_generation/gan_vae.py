"""
Project 8e: Image Generation (GANs, VAEs)
Generate images using Generative Adversarial Networks and Variational Autoencoders
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path to import plotting utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from plotting import plot_learning_curve

print("=" * 70)
print("Project 8e: Image Generation (GANs, VAEs)")
print("=" * 70)
print()

# ============================================================================
# Step 1: Create Simple Image Dataset
# ============================================================================
print("=" * 70)
print("Step 1: Creating Image Dataset")
print("=" * 70)
print()

def create_simple_images(num_samples=100, img_size=64):
    """Create simple synthetic images"""
    images = []
    
    for i in range(num_samples):
        img = np.zeros((1, img_size, img_size), dtype=np.float32)
        
        # Create pattern based on index
        pattern_type = i % 4
        
        if pattern_type == 0:  # Circle
            center = img_size // 2
            y, x = np.ogrid[:img_size, :img_size]
            mask = (x - center)**2 + (y - center)**2 <= (img_size//4)**2
            img[0][mask] = 1.0
        elif pattern_type == 1:  # Square
            margin = img_size // 4
            img[0, margin:-margin, margin:-margin] = 1.0
        elif pattern_type == 2:  # Horizontal lines
            for j in range(0, img_size, 8):
                img[0, j, :] = 1.0
        else:  # Vertical lines
            for j in range(0, img_size, 8):
                img[0, :, j] = 1.0
        
        images.append(img)
    
    return np.array(images)

images = create_simple_images(num_samples=200, img_size=64)
print(f"Created {len(images)} images")
print(f"Image shape: {images[0].shape}")
print()

# ============================================================================
# Step 2: Build Generator (GAN)
# ============================================================================
print("=" * 70)
print("Step 2: Building Generator (GAN)")
print("=" * 70)
print()

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=1, img_size=64):
        # super(): Call parent class constructor
        super(Generator, self).__init__()
        
        # latent_dim: Dimension of input noise vector
        # Generator starts from random noise (latent code)
        # Larger latent_dim = more control over generation
        self.latent_dim = latent_dim
        
        # Fully connected layer: Expand noise to feature map
        # Start from random noise
        # nn.Linear(latent_dim, 256 * 4 * 4): Projects noise to larger space
        # 256 * 4 * 4 = 4096: Size of initial feature map (256 channels, 4×4 spatial)
        # This creates a small feature map that we'll upsample
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # Upsample to image using transposed convolutions
        # ConvTranspose2d: Upsampling convolution (inverse of regular convolution)
        # Each layer doubles the spatial size while reducing channels
        self.conv_layers = nn.Sequential(
            # Layer 1: 4×4 → 8×8 (256 channels → 128 channels)
            # nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
            #   256: Input channels
            #   128: Output channels
            #   4: Kernel size
            #   stride=2: Double the size (upsample)
            #   padding=1: Padding for correct output size
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            # BatchNorm: Normalizes activations (helps training)
            nn.BatchNorm2d(128),
            # ReLU: Activation function
            nn.ReLU(),
            
            # Layer 2: 8×8 → 16×16 (128 channels → 64 channels)
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Layer 3: 16×16 → 32×32 → 64×64 (64 channels → img_channels)
            # Final layer: Output image
            nn.ConvTranspose2d(64, img_channels, 4, stride=2, padding=1),
            # Tanh: Output in [-1, 1] range
            # This matches GAN training (images normalized to [-1, 1])
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, z):
        # z: random noise (batch, latent_dim)
        # z shape: (batch, 100) - random vector (latent code)
        # This is the "seed" that generates the image
        
        # Step 1: Expand noise to larger feature map
        # self.fc(z): Fully connected layer expands noise
        # Input: (batch, 100) → Output: (batch, 4096)
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
        # Each layer doubles the size: 4×4 → 8×8 → 16×16 → 32×32 → 64×64
        # Result: (batch, img_channels, 64, 64) - generated image
        x = self.conv_layers(x)
        return x

# ============================================================================
# Step 3: Build Discriminator (GAN)
# ============================================================================
print("=" * 70)
print("Step 3: Building Discriminator (GAN)")
print("=" * 70)
print()

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

# ============================================================================
# Step 4: Train GAN
# ============================================================================
print("=" * 70)
print("Step 4: Training GAN")
print("=" * 70)
print()

# Prepare data
images_tensor = torch.FloatTensor(images)
# Normalize to [-1, 1] for GAN
images_tensor = images_tensor * 2.0 - 1.0

generator = Generator(latent_dim=100, img_channels=1, img_size=64)
discriminator = Discriminator(img_channels=1, img_size=64)

g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion = nn.BCELoss()
latent_dim = 100
batch_size = 32
epochs = 50

g_losses = []
d_losses = []

print("Training GAN...")
for epoch in range(epochs):
        # Process data in batches
        for batch_idx in range(0, len(images_tensor), batch_size):
        # Get batch of images
        # images_tensor[batch_idx:batch_idx+batch_size]: Slice to get batch
        batch_images = images_tensor[batch_idx:batch_idx+batch_size]
        batch_size_actual = batch_images.size(0)  # Actual batch size (may be smaller at end)
        
        # ===== TRAIN DISCRIMINATOR =====
        # Discriminator learns to distinguish real from fake
        discriminator.zero_grad()  # Clear gradients
        
        # Real images: Train discriminator to output 1 (real)
        # real_labels: Target is 1 (these are real images)
        # torch.ones(batch_size_actual, 1): Tensor of ones (shape: batch, 1)
        real_labels = torch.ones(batch_size_actual, 1)
        
        # Discriminator prediction on real images
        # discriminator(batch_images): Outputs probability of being real
        # Should be close to 1.0 for real images
        real_output = discriminator(batch_images)
        
        # Loss on real images: How far from 1.0?
        # criterion(real_output, real_labels): Binary cross-entropy
        # Penalizes if discriminator doesn't recognize real images
        d_loss_real = criterion(real_output, real_labels)
        
        # Fake images: Train discriminator to output 0 (fake)
        # Generate fake images from random noise
        # torch.randn(batch_size_actual, latent_dim): Random noise (normal distribution)
        noise = torch.randn(batch_size_actual, latent_dim)
        
        # Generator creates fake images
        # generator(noise): Converts noise to images
        fake_images = generator(noise)
        
        # fake_labels: Target is 0 (these are fake images)
        # torch.zeros(batch_size_actual, 1): Tensor of zeros
        fake_labels = torch.zeros(batch_size_actual, 1)
        
        # Discriminator prediction on fake images
        # .detach(): Detach from computation graph (don't update generator here)
        # Should be close to 0.0 for fake images
        fake_output = discriminator(fake_images.detach())
        
        # Loss on fake images: How far from 0.0?
        # Penalizes if discriminator doesn't recognize fake images
        d_loss_fake = criterion(fake_output, fake_labels)
        
        # Total discriminator loss: Sum of both losses
        # Discriminator wants to minimize this (better at distinguishing)
        d_loss = d_loss_real + d_loss_fake
        
        # Backward pass: Compute gradients
        d_loss.backward()
        
        # Update discriminator weights
        d_optimizer.step()
        
        # ===== TRAIN GENERATOR =====
        # Generator learns to fool discriminator
        generator.zero_grad()  # Clear gradients
        
        # Trick: Make discriminator think fake images are real
        # fake_labels = 1: We want discriminator to output 1 for fake images
        # This trains generator to create more realistic images
        fake_labels = torch.ones(batch_size_actual, 1)  # Trick discriminator
        
        # Discriminator prediction on fake images (without detach this time)
        # Now we want to update generator based on this
        fake_output = discriminator(fake_images)
        
        # Generator loss: How far from 1.0?
        # Generator wants discriminator to output 1 (thinks fake is real)
        # Lower loss = generator is fooling discriminator better
        g_loss = criterion(fake_output, fake_labels)
        
        # Backward pass: Compute gradients
        g_loss.backward()
        
        # Update generator weights
        g_optimizer.step()
        
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/{epochs}: D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}")

print()

# ============================================================================
# Step 5: Generate Images
# ============================================================================
print("=" * 70)
print("Step 5: Generating Images with GAN")
print("=" * 70)
print()

generator.eval()
with torch.no_grad():
    noise = torch.randn(8, latent_dim)
    generated_images = generator(noise)
    generated_images = (generated_images + 1) / 2.0  # Denormalize to [0, 1]

# Visualize generated images
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i in range(8):
    ax = axes[i // 4, i % 4]
    ax.imshow(generated_images[i].squeeze().numpy(), cmap='gray')
    ax.axis('off')
    ax.set_title(f'Generated {i+1}')

plt.suptitle('GAN Generated Images', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('gan_generated_images.png', dpi=150, bbox_inches='tight')
print("Saved: gan_generated_images.png")
print()

# ============================================================================
# Step 6: Variational Autoencoder (VAE)
# ============================================================================
print("=" * 70)
print("Step 6: Building Variational Autoencoder (VAE)")
print("=" * 70)
print()

class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        # super(): Call parent class constructor
        super(VAE, self).__init__()
        
        # ===== ENCODER =====
        # Encodes images to latent space (compression)
        # Maps high-dimensional images to low-dimensional latent codes
        self.encoder = nn.Sequential(
            # Layer 1: 64×64 → 32×32 (1 channel → 32 channels)
            # nn.Conv2d(1, 32, 4, stride=2, padding=1)
            #   1: Input channels (grayscale)
            #   32: Output channels
            #   4: Kernel size
            #   stride=2: Halves size (downsampling)
            #   padding=1: Maintains size calculation
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.ReLU(),  # Activation function
            
            # Layer 2: 32×32 → 16×16 (32 channels → 64 channels)
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            
            # Layer 3: 16×16 → 8×8 (64 channels → 128 channels)
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Fully connected layers: Map features to latent distribution parameters
        # 128 * 8 * 8 = 8192: Flattened feature size (128 channels × 8×8 spatial)
        # latent_dim: Dimension of latent space (e.g., 20)
        
        # mu: Mean of latent distribution
        # Encoder outputs mean of Gaussian distribution in latent space
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        
        # logvar: Log variance of latent distribution
        # Using log variance (not variance) for numerical stability
        # Variance is always positive, but log can be any value
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)
        
        # ===== DECODER =====
        # Decodes latent codes back to images (reconstruction)
        # Maps low-dimensional latent codes to high-dimensional images
        
        # Fully connected: Expand latent code to feature map
        # latent_dim → 8192: Expand to feature map size
        self.fc_decode = nn.Linear(latent_dim, 128 * 8 * 8)
        
        # Transposed convolutions: Upsample feature map to image
        self.decoder = nn.Sequential(
            # Layer 1: 8×8 → 16×16 (128 channels → 64 channels)
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            
            # Layer 2: 16×16 → 32×32 (64 channels → 32 channels)
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            
            # Layer 3: 32×32 → 64×64 (32 channels → 1 channel)
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
            # Sigmoid: Output in [0, 1] range (pixel values)
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode image to latent distribution parameters"""
        # Forward through encoder
        # h: Encoded features (batch, 128, 8, 8)
        h = self.encoder(x)
        
        # Flatten features for fully connected layers
        # h.view(h.size(0), -1): Reshape to (batch, 8192)
        # h.size(0): Batch size
        # -1: Flatten all other dimensions
        h = h.view(h.size(0), -1)
        
        # Compute distribution parameters
        # mu: Mean of latent distribution (batch, latent_dim)
        mu = self.fc_mu(h)
        # logvar: Log variance of latent distribution (batch, latent_dim)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: Sample from latent distribution"""
        # Reparameterization trick: Allows backpropagation through sampling
        # Instead of sampling directly (non-differentiable), we:
        #   1. Sample from standard normal: eps ~ N(0, 1)
        #   2. Transform: z = mu + eps × std
        
        # Compute standard deviation from log variance
        # std = exp(0.5 × logvar) = sqrt(exp(logvar)) = sqrt(variance)
        # exp(0.5 * logvar): Square root of variance
        std = torch.exp(0.5 * logvar)
        
        # Sample noise from standard normal distribution
        # torch.randn_like(std): Random values with same shape as std
        # eps ~ N(0, 1): Standard normal distribution
        eps = torch.randn_like(std)
        
        # Transform: z = mu + eps × std
        # This samples from N(mu, std²) distribution
        # Differentiable because eps is independent of mu and logvar
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent code to image"""
        # Expand latent code to feature map
        # self.fc_decode(z): (batch, latent_dim) → (batch, 8192)
        h = self.fc_decode(z)
        
        # Reshape to feature map: (batch, 8192) → (batch, 128, 8, 8)
        # h.size(0): Batch size
        # 128, 8, 8: Channels, height, width
        h = h.view(h.size(0), 128, 8, 8)
        
        # Upsample through decoder: 8×8 → 64×64
        # self.decoder(h): Reconstructs image from latent code
        return self.decoder(h)
    
    def forward(self, x):
        """Forward pass: encode, sample, decode"""
        # Encode: Image → latent distribution parameters
        # mu, logvar: Parameters of Gaussian distribution in latent space
        mu, logvar = self.encode(x)
        
        # Reparameterize: Sample latent code from distribution
        # z: Sampled latent code (batch, latent_dim)
        z = self.reparameterize(mu, logvar)
        
        # Decode: Latent code → reconstructed image
        # recon: Reconstructed image (should match input x)
        recon = self.decode(z)
        
        # Return: (reconstructed_image, mu, logvar)
        # mu and logvar needed for KL divergence loss
        return recon, mu, logvar

# ============================================================================
# Step 7: Train VAE
# ============================================================================
print("=" * 70)
print("Step 7: Training VAE")
print("=" * 70)
print()

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss: reconstruction + KL divergence"""
    # ===== RECONSTRUCTION LOSS =====
    # Measures how well VAE reconstructs input
    # nn.BCELoss(): Binary Cross-Entropy Loss (for pixel values in [0, 1])
    # recon_x: Reconstructed image (decoder output)
    # x: Original image (input)
    # Lower = better reconstruction (output matches input)
    recon_loss = nn.BCELoss()(recon_x, x)
    
    # ===== KL DIVERGENCE LOSS =====
    # Measures how close latent distribution is to standard normal
    # Encourages latent space to be well-structured (smooth, continuous)
    # Formula: -0.5 × Σ(1 + logvar - mu² - exp(logvar))
    # This is KL divergence between N(mu, exp(logvar)) and N(0, 1)
    # mu: Mean of latent distribution
    # logvar: Log variance of latent distribution (for numerical stability)
    # torch.sum(...): Sum over all dimensions
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Average over batch (divide by batch size)
    # x.size(0): Batch size
    kl_loss = kl_loss / x.size(0)  # Average
    
    # ===== TOTAL LOSS =====
    # Combined loss: reconstruction + KL divergence
    # beta: Weight for KL term (controls regularization strength)
    # beta=1.0: Equal weight to both terms
    # Higher beta = more regularization (smoother latent space)
    # Lower beta = better reconstruction (but less structured latent space)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

vae = VAE(latent_dim=20)
optimizer_vae = optim.Adam(vae.parameters(), lr=0.001)

# Normalize images to [0, 1] for VAE
images_vae = torch.FloatTensor(images)

print("Training VAE...")
epochs_vae = 50
vae_losses = []

for epoch in range(epochs_vae):
    vae.train()
    optimizer_vae.zero_grad()
    recon, mu, logvar = vae(images_vae)
    loss, recon_loss, kl_loss = vae_loss(recon, images_vae, mu, logvar)
    loss.backward()
    optimizer_vae.step()
    
    vae_losses.append(loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/{epochs_vae}: Loss={loss.item():.4f}, "
              f"Recon={recon_loss.item():.4f}, KL={kl_loss.item():.4f}")

print()

# ============================================================================
# Step 8: Generate Images with VAE
# ============================================================================
print("=" * 70)
print("Step 8: Generating Images with VAE")
print("=" * 70)
print()

vae.eval()
with torch.no_grad():
    # Sample from latent space
    z = torch.randn(8, 20)
    generated_vae = vae.decode(z)

# Visualize VAE generated images
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i in range(8):
    ax = axes[i // 4, i % 4]
    ax.imshow(generated_vae[i].squeeze().numpy(), cmap='gray')
    ax.axis('off')
    ax.set_title(f'VAE Generated {i+1}')

plt.suptitle('VAE Generated Images', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('vae_generated_images.png', dpi=150, bbox_inches='tight')
print("Saved: vae_generated_images.png")
print()

# Visualize training
plot_learning_curve(g_losses, title="GAN Generator Loss")
plot_learning_curve(d_losses, title="GAN Discriminator Loss")
plot_learning_curve(vae_losses, title="VAE Training Loss")

print("=" * 70)
print("Project 8e Complete!")
print("=" * 70)
print()
print("Summary:")
print("  ✅ Built Generator and Discriminator (GAN)")
print("  ✅ Trained GAN to generate images")
print("  ✅ Built Encoder and Decoder (VAE)")
print("  ✅ Trained VAE to generate images")
print("  ✅ Compared GAN vs VAE generation")
print()
