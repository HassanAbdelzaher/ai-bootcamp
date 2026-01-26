# Step 8e — Image Generation (GANs, VAEs)

> **Goal:** Introduction to generative models for creating images.  
> **Tools:** Python + PyTorch

---

## 8e.1 Generative vs Discriminative

**Discriminative:**
- Learn P(y|x): probability of class given image
- Classify existing images

**Generative:**
- Learn P(x): probability distribution of images
- Create new images

---

## 8e.2 What are GANs?

**GAN = Generative Adversarial Network**

**Two networks compete:**
1. **Generator**: Creates fake images
2. **Discriminator**: Distinguishes real from fake

**Training:**
- Generator tries to fool discriminator
- Discriminator tries to catch fakes
- Both improve together

---

## 8e.3 What are VAEs?

**VAE = Variational Autoencoder**

**Architecture:**
- Encoder: Image → Latent code
- Decoder: Latent code → Image

**Key difference:**
- VAE: Learns to reconstruct + generate
- GAN: Learns to generate realistic images

---

## 8e.4 GAN vs VAE

**GAN:**
- ✅ More realistic images
- ✅ State-of-the-art quality
- ❌ Harder to train
- ❌ Mode collapse problem

**VAE:**
- ✅ More stable training
- ✅ Interpretable latent space
- ✅ Can interpolate
- ❌ Blurrier images

---

## 8e.5 Modern Generative Models

**Diffusion Models:**
- DALL-E, Midjourney, Stable Diffusion
- Generate images from text
- Very high quality

**Transformers for Images:**
- DALL-E uses transformer
- Generate images autoregressively
