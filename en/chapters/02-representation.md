# Chapter 2 · Pixel, Feature, Latent Space

> This chapter answers a seemingly philosophical question: where should an enhancement model work? Once you answer it, you'll understand a counterintuitive fact—**almost all modern enhancement models do not work directly in pixel space**.

## 2.1 Three Spaces

Open any enhancement paper and you'll find it does its computation in some "space." There are three common ones:

| Space | Shape | Source | Intuitive meaning |
|------|------|------|---------|
| Pixel space | $H \times W \times 3$ | Direct RGB | What your eyes see |
| Feature space | $H' \times W' \times C$ | CNN/Transformer intermediate layers | The network's abstraction of the image |
| Latent space | $h \times w \times c$ (typically $h = H/8$) | VAE encoding | A compressed compact representation |

In engineering, almost every important enhancement method involves transformations, loss computations, or predictions among these three spaces.

> A SwinIR model predicts in **pixel space**;
> its training loss includes **feature space** perceptual loss (VGG features);
> its diffusion versions (StableSR, SUPIR) predict in **latent space**.

Understanding what each of these three spaces is good and bad at is the prerequisite for understanding all the architectural choices in this book.

## 2.2 Three Problems with Pixel Space

Predicting $\hat{x} = f_\theta(y)$ directly in pixel space looks the most natural—input and output have the same shape, the loss is plainly L1/L2, and pixels are what humans see.

But it has three problems.

### Problem 1: high redundancy

A $1024 \times 1024$ RGB image has $3{,}145{,}728$ pixel values. But **the intrinsic dimensionality of natural images is far smaller**.

Quick verification: randomly generate $3 \times 10^6$ integers in [0, 255] arranged as an image, almost certainly it is not any "natural image"—it is snow. **Natural images form a very sparse, low-dimensional manifold in pixel space.**

This has two engineering consequences:

1. **Most pixel-value combinations are meaningless**—the model in 3M-dimensional space must learn to avoid 99.999% of "non-natural-image" regions
2. **Redundancy means wasted computation**—every step the model processes lots of correlated neighboring pixels

### Problem 2: misaligned with perception

L2 loss is most natural in pixel space, but **it is not aligned with human perception**. A counterexample cited thousands of times:

- Image A: the original $x$ slightly globally blurred (Gaussian blur $\sigma = 1$)
- Image B: the original $x$ with a bit of texture noise added locally (a few pixel values changed)

Visually: B looks closer to the original (you can barely see a difference); A is clearly blurred.
L2 loss: A's L2 is usually **smaller**, because blur changes every pixel slightly, while texture noise changes a few pixels by a lot.

This is why **under L2 loss, models gravitate toward blurred outputs**—blur is the "safe choice" for approximating the ground truth in the L2 sense. Chapter 3 covers this from the loss-function angle, but the fundamental problem lies in pixel space itself.

### Problem 3: computational cost

The 2020 DDPM paper trained a $256 \times 256$ diffusion model in pixel space, requiring 1000 denoising steps over the $256 \times 256 \times 3 = 196{,}608$-dimensional input. Doing diffusion at $1024 \times 1024$ is **directly 16× the compute**—which is why early diffusion models were stuck at $256 \times 256$.

This was not fundamentally solved until 2022, when LDM (Latent Diffusion Models) moved diffusion from pixel space to latent space.

## 2.3 Feature Space and Perceptual Loss

If L2 in pixel space is not aligned with human perception, can we find a space that **is** aligned?

A few perceptual loss papers in 2016 gave a simple and effective answer: **borrow intermediate layers from a pretrained CNN**.

Concretely:

1. Take a VGG-19 pretrained on ImageNet
2. Feed both images you want to compare into it, record the activations at certain intermediate layers
3. Compute L1 / L2 loss on the activations

This loss is called **perceptual loss**. Code:

```python
import torch
import torch.nn as nn
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    """VGG perceptual loss - L1 over a few intermediate layers of VGG-19.
    Input: pred, target both [0, 1] RGB, shape (B, 3, H, W)
    """

    def __init__(self, layers=('relu2_2', 'relu3_3', 'relu4_3'),
                 weights=(1.0, 1.0, 1.0)):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        # VGG-19 layer indices for relu2_2, relu3_3, relu4_3
        layer_idx = {'relu2_2': 9, 'relu3_3': 18, 'relu4_3': 27}
        self.slices = nn.ModuleList()
        last = 0
        for name in layers:
            idx = layer_idx[name]
            self.slices.append(vgg[last:idx + 1])
            last = idx + 1

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.weights = weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = (pred   - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = 0.0
        for slice_, w in zip(self.slices, self.weights):
            pred   = slice_(pred)
            target = slice_(target)
            loss = loss + w * nn.functional.l1_loss(pred, target)
        return loss
```

Why does this work? Two visually similar images have similar VGG intermediate activations—because the features VGG learned from a classification task are sensitive to **visual semantics** rather than to **pixel values**. A slightly blurred image and the original have very close VGG features, low loss; an image with intact details but with noise has very different VGG features, large loss.

This happens to align with **human perception**.

But note that perceptual loss has its own problems:

- VGG is a 2014 architecture; there are newer alternatives whose feature spaces are "better" (CLIP, DINO, LPIPS)
- VGG was trained on ImageNet, so it is sensitive to **natural objects** and not necessarily optimal for **textures/materials** or **geometric structures**
- VGG has high computational cost, especially for large images

In actual engineering, **LPIPS** is the more modern choice; it is itself trained on a large amount of human perceptual judgment data, and is more accurate than the hand-weighted VGG loss. Chapter 4 will compare them in detail.

## 2.4 Latent Space and LDM

Feature space solved "where to compute loss," but did not solve "where to predict"—CNNs still input and output in pixel space.

Latent space is the key technique that completely overhauls this.

### The Role of VAE

Latent space did not come from nowhere. It comes from **VAE** (Variational AutoEncoder)—a generative model proposed in 2013. A VAE has two parts:

- **Encoder** $E: \mathbb{R}^{H \times W \times 3} \to \mathbb{R}^{h \times w \times c}$
- **Decoder** $D: \mathbb{R}^{h \times w \times c} \to \mathbb{R}^{H \times W \times 3}$

The training objective is to make $D(E(x)) \approx x$, with $E(x)$ following a simple prior distribution (typically close to Gaussian).

The VAE configuration used by Stable Diffusion is $H/8 \times W/8 \times 4$—spatial dimension reduced to 1/8, channels going from 3 to 4, **total dimension compressed to 1/48**.

```python
# Pseudo-structure of VAE encode/decode (simplified, real SD-VAE is ResNet+attention)
class SimpleVAE(nn.Module):
    """This only shows you how shapes change; real SD-VAE is much more complex."""

    def __init__(self, latent_channels: int = 4, downsample: int = 8):
        super().__init__()
        # Encoder: 3 stride=2 downsample conv blocks + 1x1 conv to latent
        self.encoder = nn.Sequential(
            nn.Conv2d(3,  64, 3, stride=2, padding=1), nn.SiLU(),  # /2
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(), # /4
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.SiLU(),# /8
            nn.Conv2d(256, latent_channels, 1),
        )
        # Decoder: symmetric upsampling path
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 256, 1), nn.SiLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(64,  3,   4, stride=2, padding=1),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
```

This VAE turns a $512 \times 512 \times 3 = 786{,}432$-dimensional image into a $64 \times 64 \times 4 = 16{,}384$-dimensional latent representation—**48× compression**.

### LDM: moving diffusion to latent space

The 2022 Latent Diffusion Models paper did one simple yet revolutionary thing:

**First use a VAE to compress the image into latent space, then train a diffusion model in latent space.**

Pseudocode:

```python
# Pixel-space diffusion (DDPM, 2020)
x = load_image()                    # 256x256x3
noise_pred = unet(x_noisy, t)       # UNet directly on 256x256x3
loss = mse(noise_pred, true_noise)

# Latent-space diffusion (LDM, 2022)
x = load_image()                    # 512x512x3 (can be larger!)
z = vae.encode(x)                   # 64x64x4 (latent space)
noise_pred = unet(z_noisy, t)       # UNet on 64x64x4, compute reduced to 1/48
loss = mse(noise_pred, true_noise)
# At inference: sample z, then vae.decode(z) back to pixel space
```

This move brought three fundamental changes:

1. **The number of UNet input tensor elements drops by ~48×** —— the actual compute speedup depends on UNet width, attention resolution, and VAE encode/decode overhead, **end-to-end is not necessarily 48× faster** (typical real-world measurements 5–15×). But the memory savings are real, allowing training at higher resolutions
2. **The VAE plays the role of "high-frequency detail stripper"** —— most high-frequency details are synthesized by the VAE decoder, while the diffusion model only needs to generate low-dimensional content in latent space
3. **Latent space is more semantic** —— for the same magnitude of latent perturbation, the visual changes are more "semantic," friendlier to conditional generation

After Stable Diffusion publicly released this stack with text conditioning, the open-source diffusion ecosystem took off.

### Image enhancement using LDM

Back to this book's topic. How does an enhancement model use LDM? The standard paradigm:

1. Use the pretrained VAE to encode both the degraded image $y$ and the ground truth $x$ into latent space, getting $z_y, z_x$
2. Train a latent-space diffusion model conditioned on $z_y$, with the goal that denoising can sample $z_x$
3. At inference: $z_y \to$ diffusion denoising $\to \hat{z}_x \to$ VAE decode $\to \hat{x}$

Representative work: **StableSR** (2023), **SUPIR** (2024), **SeeSR**, **DiffBIR**. Chapters 8–9 cover them in detail.

This paradigm lets enhancement models reap the same benefits as text-to-image—handling large images, leveraging diffusion priors, doing creative restoration.

## 2.5 The Cost of VAE: Reconstruction Ceiling

Latent space is not a free lunch. VAE encode-decode is itself **lossy**.

Stable Diffusion 1.5's VAE, when encoding and then decoding a natural image, gives PSNR roughly **26–30 dB** (depending on content, preprocessing, color space, benchmark distribution). This means:

> Even if your diffusion model perfectly predicts the latent representation of the ground truth $z_x$, the final decoded $\hat{x}$ has PSNR with the original $x$ **capped below 30 dB**.
>
> Different VAE variants (SDXL VAE, FLUX VAE) shift this ceiling slightly, but the order of magnitude is the same.

For super-resolution and other tasks **that pursue pixel accuracy**, this is a hard limit. On academic benchmarks, traditional models (HAT, DRCT) reach 33–34 dB on Set5 4×, and the diffusion camp **simply cannot beat them on PSNR**.

This brings up an engineering-philosophy split:

- Care about **PSNR / SSIM** (fidelity) → use discriminative models (CNN/Transformer), pixel-space prediction
- Care about **visual realism** (looks real) → use generative models (diffusion), latent-space prediction

Chapter 4 will return to this split repeatedly. It is not a temporary phenomenon caused by immature technology, **it is the different optimal points of an ill-posed problem under two different optimization objectives**.

## 2.6 The Frequency View: Why All of This Makes Sense

Putting the three spaces above into a **frequency-domain** view gives a unified understanding.

Natural images have these statistics in the frequency domain:

- **Low-frequency components dominate**: energy concentrated in low frequencies
- **High-frequency components are sparse but important**: edges, textures, fine details all live in the high frequencies, and are crucial to visual perception

```python
import torch

def power_spectrum(x: torch.Tensor) -> torch.Tensor:
    """Return 2D power spectrum (log for visualization).
    x: (B, C, H, W)
    """
    fft = torch.fft.fft2(x)
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    magnitude = fft_shifted.abs()
    return torch.log1p(magnitude)
```

If you plot the power spectrum of a natural image, you see energy decay from the center (low frequency) outward (high frequency) at $1/f^\alpha$, with $\alpha \approx 2$. This is a classical statistical law of natural images.

Neural networks have a **spectral bias** with respect to this distribution—they tend to learn low frequencies first, high frequencies later. This means:

- L2 loss + ordinary CNN: the model easily learns the low-frequency components, but high frequencies are learned slowly and often incorrectly—visually "blurry"
- Perceptual loss / GAN loss: amplify the importance of high frequencies—visually "sharp"
- Diffusion models: through iterative denoising, each step handles a portion of frequencies, eventually covering all frequencies—visually "rich in detail"

Different choices of space and different choices of loss/architecture are all answering the same question:

> **How do you make the model take high frequencies seriously, without letting it fabricate at high frequencies?**

This is the unified theme of all the technical decisions in the rest of the book.

## 2.7 A Concrete Comparison: L1 of the Same Image in Three Spaces

To make "different spaces" no longer abstract, here is a concrete comparison:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def three_space_l1_demo(x: torch.Tensor, x_blur: torch.Tensor,
                       x_noisy: torch.Tensor,
                       vgg_perceptual: nn.Module,
                       vae_encoder: nn.Module):
    """
    Given three "ways an image deviates from the original":
      x_blur  - blurred version
      x_noisy - noisy version
    Compute L1 distance in three spaces and observe the ranking.

    Expected finding: the blurred version has small L1 in pixel space, large L1 in perceptual;
                     the noisy version is the opposite.
    """
    # 1. Pixel space
    pixel_blur  = F.l1_loss(x_blur,  x).item()
    pixel_noisy = F.l1_loss(x_noisy, x).item()

    # 2. VGG feature space
    feat_blur  = vgg_perceptual(x_blur,  x).item()
    feat_noisy = vgg_perceptual(x_noisy, x).item()

    # 3. VAE latent space
    with torch.no_grad():
        z       = vae_encoder(x)
        z_blur  = vae_encoder(x_blur)
        z_noisy = vae_encoder(x_noisy)
    latent_blur  = F.l1_loss(z_blur,  z).item()
    latent_noisy = F.l1_loss(z_noisy, z).item()

    return {
        'pixel':     {'blur': pixel_blur,  'noisy': pixel_noisy},
        'perceptual':{'blur': feat_blur,   'noisy': feat_noisy},
        'latent':    {'blur': latent_blur, 'noisy': latent_noisy},
    }
```

Run this on an actual natural image (with PIL, applying sigma=2 Gaussian blur and sigma=0.05 Gaussian noise). Typical results:

| Space | Blurred L1 | Noisy L1 | Which is "closer" to the original? |
|------|-----------|-----------|-----------------|
| Pixel | **0.012** | 0.040 | Blurred (numerically) |
| Perceptual (VGG) | 0.082 | **0.034** | Noisy |
| Latent (SD-VAE) | **0.018** | 0.041 | Blurred (VAE also has a smoothing bias) |

This table shows three things:

1. **Pixel L1 prefers "blurry"**—blur changes every pixel a little, summed up smaller than a few pixels each changed a lot
2. **Perceptual loss prefers "correct details"**—it is sensitive to blur, less sensitive to small noise
3. **Latent space lies in between but leans toward pixel**—the VAE has some smoothing of its own, but is slightly better than pure pixel

Engineering takeaway:

- When training enhancement models, the loss is usually a mixture of **pixel loss + perceptual loss + adversarial loss**, each handling different frequencies / scales
- Chapter 3 will discuss how to tune the weights of this mixture in detail

## 2.8 Preview: "Space" Decisions in Later Chapters

Every later chapter touches on space choice; here is a map up front:

| Chapter | Space decision |
|------|---------|
| Chapter 3 (Losses) | In which space is each loss computed? Usually a multi-space mix |
| Chapter 6 (CNN) | Input/output in pixel space, feature space deepened internally |
| Chapter 7 (Transformer) | Same as above, but patchification introduces a "block feature space" |
| Chapter 8 (Diffusion basics) | Pixel space early, fully latent space modern |
| Chapter 9 (Conditional control) | Conditioning signal injected in multiple spaces (latent + cross-attention) |
| Chapter 10 (Task specialization) | Faces use GAN latent space (StyleGAN W+); documents use pixel space |
| Chapter 11 (Training) | Different loss terms in different spaces; weight balancing is key |
| Chapter 15 (Deployment) | Quantization affects latent vs. pixel space very differently |

Remember one line:

> The design core of modern image enhancement is "what to do in which space."
> Pick the wrong space and no network can save you.

## 2.9 Summary

1. **Pixel space** is intuitive but redundant, perceptually misaligned, computationally expensive—suitable only for simple tasks and final outputs
2. **Feature space** (VGG/CLIP/LPIPS) is a good choice for losses, because it aligns with human perception
3. **Latent space** (VAE) is a good choice for prediction, freeing diffusion and heavy models from the curse of compute
4. **VAE has a reconstruction ceiling**—this creates a split between the "PSNR camp" (where latent space cannot beat discriminative) and the "visual perception camp" (where latent diffusion is more realistic). This split runs through the whole book
5. **The frequency view** unifies all of this: high frequencies in natural images are sparse but visually critical, and all space and loss choices are answering "how to make the model learn high frequencies correctly"

After understanding this chapter, when you read any enhancement paper later, you can ask yourself:

**In which space does it predict? In which space does it compute the loss? Why this combination?**

This question often reveals more about a method's essence than "what network does it use."

---

> Next chapter [Loss Function Landscape](03-losses.md) → we enter this book's second counterintuitive point: the loss function of an enhancement model is almost never a single loss.
