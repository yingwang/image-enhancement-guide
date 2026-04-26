# Chapter 3 · Loss Function Landscape

> The loss of an enhancement model is almost never a single loss. Understanding why—and how to mix them—is what this chapter answers.

## 3.1 Why This Chapter

In LLM training the loss is essentially one: next-token cross-entropy. In classification it's essentially one too: cross-entropy.

Image enhancement is different. Open any mainstream enhancement paper and the loss-function section typically looks like:

$$
\mathcal{L} = \lambda_1 \mathcal{L}_{\text{pixel}} + \lambda_2 \mathcal{L}_{\text{perceptual}} + \lambda_3 \mathcal{L}_{\text{adversarial}} + \lambda_4 \mathcal{L}_{\text{...}}
$$

Four or five loss terms, weighted and mixed. Why? Chapters 1 and 2 already answered:

- An ill-posed problem needs priors, and **each loss term is one way to encode a prior**
- Different losses prefer different frequency components (L2 prefers low frequencies, adversarial prefers high frequencies, perceptual prefers mid-to-high)
- A single loss over-optimizes some axis (L2 → blur, pure adversarial → fake details, pure perceptual → color drift)

> Training an enhancement model is not "making the model approach the ground truth," it is "finding a Pareto-optimal point on multiple competing metrics."
>
> Loss weighting = telling the model which of those metrics you care about most.

This chapter classifies the loss terms commonly used in engineering, and ends with a decision table for mixing strategies.

## 3.2 Pixel-Space Losses

The most basic family. Computed directly on pixel differences.

### L2 (MSE)—mathematically convenient, engineering-unfriendly

$$
\mathcal{L}_2 = \frac{1}{N} \sum_i (\hat{x}_i - x_i)^2
$$

Mathematical properties:

- Corresponds to maximum likelihood under a Gaussian noise assumption
- Differentiable everywhere, convex
- Directly corresponds to the PSNR metric

Engineering problems:

- **Sensitive to outliers**—a single wildly wrong pixel contributes $error^2$ and can dominate the gradient
- **Biased toward the mean**—given $y$, the minimizer of $\mathbb{E}[||\hat{x} - x||^2]$ is $\mathbb{E}[x | y]$, the average of multiple plausible $x$, which looks blurry
- **Misaligned with perception**—Section 2.7 already verified this

In practice L2 is now used in only two places:

1. **Diffusion model denoising loss** (mathematically must be L2, since score matching derives this)
2. **Early ablation experiments** (as a baseline)

### L1 (MAE)—the modern default

$$
\mathcal{L}_1 = \frac{1}{N} \sum_i |\hat{x}_i - x_i|
$$

Properties:

- Corresponds to maximum likelihood under a Laplacian noise assumption
- Not differentiable at 0 (in practice the gradient is the sign function, an engineering non-issue)
- **More robust to outliers**—a wildly wrong pixel contributes $|error|$ and does not dominate the gradient

Visual effect: images trained with L1 are slightly sharper than those trained with L2. Reason: L1 is not "flattened" by the mean—given $y$, the minimizer of $\mathbb{E}[||\hat{x} - x||_1]$ is $\text{median}(x | y)$, which leans toward "some specific plausible $x$" rather than the mean.

**Almost all modern non-diffusion enhancement models use L1 (or Charbonnier) as the pixel-loss base.**

### Charbonnier—a smooth version of L1

$$
\mathcal{L}_{\text{Charb}} = \frac{1}{N} \sum_i \sqrt{(\hat{x}_i - x_i)^2 + \epsilon^2}
$$

$\epsilon$ is typically $10^{-3}$ or $10^{-6}$. When $|error| \gg \epsilon$ it degenerates to L1; when $|error| \approx 0$ it degenerates to L2.

Why use this instead of plain L1?

- L1 has a step gradient of $\pm 1$ around 0, with **gradient that does not decay** as you approach the truth, causing oscillation in late training
- Charbonnier is smooth around 0 and gradients automatically shrink near the truth, training is more stable

```python
import torch

def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor,
                     eps: float = 1e-3) -> torch.Tensor:
    """Charbonnier loss (a continuous version of smooth L1).
    More numerically stable than L1, more robust than L2; the de facto standard in low-level vision.
    """
    diff = pred - target
    return torch.sqrt(diff * diff + eps * eps).mean()
```

Engineering experience: Restormer, NAFNet, SwinIR and others all write "L1 loss" in the paper, but the actual code often uses Charbonnier because it trains more stably.

## 3.3 Smoothness / Gradient Losses

Pixel losses only look at point-to-point. **Gradient losses** look at differences between neighboring points.

### Total Variation

$$
\mathcal{L}_{\text{TV}} = \sum_{i,j} \left( |\hat{x}_{i+1,j} - \hat{x}_{i,j}| + |\hat{x}_{i,j+1} - \hat{x}_{i,j}| \right)
$$

It penalizes the differences between neighboring pixels and encourages **piecewise smoothness**—flat interior, sharp edges.

```python
def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    """TV loss (anisotropic version).
    x: (B, C, H, W)
    """
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw
```

When to use:

- **Denoising scenarios**: TV is a classical prior that suppresses noise
- **Generative models**: diffusion/GAN models tend to produce texture artifacts in flat regions; adding TV holds them down
- **Engineering weight is small** (usually $\lambda = 10^{-5}$ to $10^{-3}$); too large will smear out details

### Gradient-domain loss

A finer version: compute L1 in the gradient domain.

$$
\mathcal{L}_{\text{grad}} = ||\nabla \hat{x} - \nabla x||_1
$$

```python
import torch.nn.functional as F

# Sobel kernels
SOBEL_X = torch.tensor([[-1., 0., 1.],
                        [-2., 0., 2.],
                        [-1., 0., 1.]]).view(1, 1, 3, 3)
SOBEL_Y = SOBEL_X.transpose(-1, -2)

def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Gradient-domain L1 loss. Forces predicted edges to align with ground-truth edges.
    pred, target: (B, C, H, W), assumed range [0, 1]
    """
    sobel_x = SOBEL_X.to(pred).expand(pred.shape[1], 1, 3, 3)
    sobel_y = SOBEL_Y.to(pred).expand(pred.shape[1], 1, 3, 3)

    pred_gx = F.conv2d(pred, sobel_x, padding=1, groups=pred.shape[1])
    pred_gy = F.conv2d(pred, sobel_y, padding=1, groups=pred.shape[1])
    targ_gx = F.conv2d(target, sobel_x, padding=1, groups=target.shape[1])
    targ_gy = F.conv2d(target, sobel_y, padding=1, groups=target.shape[1])

    return (pred_gx - targ_gx).abs().mean() + (pred_gy - targ_gy).abs().mean()
```

When to use:

- Tasks where edges are crucial (deblurring, document enhancement)
- As a lightweight substitute for perceptual loss

## 3.4 Perceptual Loss

Chapter 2 already introduced VGG perceptual loss. Here we add a few variants.

### LPIPS

LPIPS (Learned Perceptual Image Patch Similarity) is 2018 work. In essence it **replaces the hand-weighted perceptual loss with a data-driven one**:

1. Use AlexNet/VGG/SqueezeNet to extract features
2. **Add a learned linear weight on each layer's features** (trained on a large amount of human perceptual judgment data)
3. Use the weighted distance as the perceptual distance

It aligns better with human perception than VGG perceptual loss. But **as a training loss** there are several pitfalls:

- More compute than VGG
- Bias on certain textures
- Using it as the primary loss can lead the model to produce "gamed" outputs (optimized against LPIPS but visually bad)

In practice, LPIPS is more often used as an **evaluation metric**; Chapter 4 covers this in detail. As a training loss, VGG is still primary.

### DISTS

DISTS (Deep Image Structure and Texture Similarity) is 2020 work, splitting perceptual loss into structure and texture parts. Its characteristic is **insensitivity to position**—the same texture in different positions is not penalized.

When to use: texture-generation tasks (skin, hair, fabric) where DISTS is more suitable than VGG.

### CLIP loss

Run the image through a CLIP image encoder, compute distance in CLIP feature space.

Properties:

- Sensitive to **semantic content**, insensitive to low-level details
- Suitable as a "content preservation" constraint, not as a primary loss
- Used a lot in generative enhancement (diffusion), less in discriminative enhancement

Engineering experience: CLIP loss alone cannot train a good enhancement model, but **adding it as a regularizer** (weight 0.01) ensures the model doesn't "drift"—e.g., restoring a photo of a cat won't turn it into a dog.

## 3.5 Adversarial Loss

GAN loss is one of the most complex, easy-to-mess-up, and crucial loss types in this book. Its role: **make the model produce real details**, instead of "safely outputting blur."

### Vanilla GAN—don't use directly

The original GAN loss:

$$
\min_G \max_D \mathbb{E}_{x \sim p_{\text{data}}} [\log D(x)] + \mathbb{E}_{z \sim p_z} [\log(1 - D(G(z)))]
$$

Theoretically nice but engineering-wise **extremely unstable**:

- Vanishing gradient: when $D$ is too well trained, $\log(1 - D(G(z)))$ saturates as $D(G(z)) \to 0$
- Mode collapse: $G$ outputs trend toward a single mode
- Training divergence: $D$ and $G$ fail to reach Nash equilibrium

Real engineering does not use vanilla GAN; it uses the variants below.

### LSGAN—simple and stable

Replace sigmoid + log with MSE:

$$
\mathcal{L}_D = \frac{1}{2}\mathbb{E}[(D(x) - 1)^2] + \frac{1}{2}\mathbb{E}[(D(G(z)))^2]
$$
$$
\mathcal{L}_G = \frac{1}{2}\mathbb{E}[(D(G(z)) - 1)^2]
$$

Properties: gradient does not saturate, training stable, hyperparameters easy to tune.

### Hinge loss—the modern GAN standard

$$
\mathcal{L}_D = -\mathbb{E}[\min(0, D(x) - 1)] - \mathbb{E}[\min(0, -D(G(z)) - 1)]
$$
$$
\mathcal{L}_G = -\mathbb{E}[D(G(z))]
$$

Properties: when $D$ already separates the two well, no further pushing (margin), training more stable. BigGAN, StyleGAN, etc. all use this.

```python
def hinge_d_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """Hinge loss (discriminator side).
    d_real, d_fake: discriminator output logits on real/fake, shape (B,) or (B, 1, h', w')
    """
    loss_real = F.relu(1.0 - d_real).mean()
    loss_fake = F.relu(1.0 + d_fake).mean()
    return loss_real + loss_fake


def hinge_g_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """Hinge loss (generator side)."""
    return -d_fake.mean()
```

### Relativistic GAN—ESRGAN's choice

ESRGAN uses Relativistic average GAN (RaGAN). Its core idea:

> The discriminator should not judge "is this real?"
> It should judge "is this more like the real one than that fake one?"

Formally:

$$
D_{\text{Ra}}(x_r, x_f) = \sigma(D(x_r) - \mathbb{E}[D(x_f)])
$$

$$
D_{\text{Ra}}(x_f, x_r) = \sigma(D(x_f) - \mathbb{E}[D(x_r)])
$$

The discriminator loss simultaneously cares about "real should be more real" and "fake should be less real," giving $D$'s training signal a friendlier shape for $G$.

```python
def relativistic_d_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """RaGAN discriminator loss (ESRGAN style)."""
    real_logits = d_real - d_fake.mean()
    fake_logits = d_fake - d_real.mean()
    return (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
          + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)))


def relativistic_g_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """RaGAN generator loss.
    Note: at call time, d_real and d_fake should be D's current logits on G's output and the ground truth,
    and **D's parameters should be frozen during backward** (set_requires_grad(D, False) in the G step),
    to avoid G's loss accidentally updating D.
    """
    real_logits = d_real - d_fake.mean()
    fake_logits = d_fake - d_real.mean()
    return (F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
          + F.binary_cross_entropy_with_logits(real_logits, torch.zeros_like(real_logits)))
```

In practice: on super-resolution tasks RaGAN improves LPIPS by about 0.05 over LSGAN, and is the long-running standard on the ESRGAN-Real-ESRGAN line.

### Patch GAN

Instead of outputting a single scalar discrimination, output a feature map where each position judges the real/fake of its receptive field.

Properties:

- **Local discrimination**: the model cannot slack off in any region
- **Naturally supports arbitrary resolution** input
- **More stable**: each patch independently provides gradient, not drowned by a global signal

Code (Chapter 6 will expand on the discriminator architecture):

```python
class PatchDiscriminator(nn.Module):
    """70x70 receptive field PatchGAN, Pix2Pix/Real-ESRGAN style."""
    def __init__(self, in_ch: int = 3, base_ch: int = 64):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        for i in range(len(chs) - 1):
            stride = 2 if i < 2 else 1
            layers += [
                nn.Conv2d(chs[i], chs[i+1], 4, stride=stride, padding=1),
                nn.GroupNorm(8, chs[i+1]),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        layers.append(nn.Conv2d(chs[-1], 1, 4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # output (B, 1, h', w'), no sigmoid (logits)
```

### Spectral normalization + dropout

There are also a few engineering tricks for stabilizing GAN training, beyond the loss itself:

- **Spectral Normalization**: spectral-normalize each layer of the discriminator to control the Lipschitz constant, with significant effect
- **R1 / R2 regularization**: gradient penalty on the discriminator at real/fake inputs
- **Two-Time-Scale Update Rule (TTUR)**: use a higher learning rate (4×) for $D$

Chapter 11 will gather these engineering details together.

## 3.6 Diffusion Losses

Diffusion models have their own family of losses, **not in the same framework as the discriminative-loss system above**.

### Simple loss (epsilon prediction)

DDPM's standard objective:

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ ||\epsilon - \epsilon_\theta(x_t, t)||^2 \right]
$$

The model directly predicts the added noise. This is the form you see in most diffusion tutorials.

```python
def diffusion_simple_loss(model, x0, t, noise_scheduler):
    """DDPM standard epsilon prediction loss.
    model: UNet, takes (x_t, t) as input, outputs predicted noise
    x0: original image (B, C, H, W) or latent (B, c, h, w)
    t: timestep (B,)
    noise_scheduler: provides alpha_cumprod
    """
    noise = torch.randn_like(x0)
    sqrt_alpha = noise_scheduler.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = noise_scheduler.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    x_t = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
    pred_noise = model(x_t, t)
    return F.mse_loss(pred_noise, noise)
```

### v-prediction—numerically better

Original DDPM's problem: when $t$ is small (close to $x_0$), $\epsilon$ has already been almost fully "squeezed out," the prediction target signal is weak and SNR is bad.

v-prediction (Salimans & Ho 2022) instead predicts:

$$
v_t = \sqrt{\bar{\alpha}_t} \cdot \epsilon - \sqrt{1 - \bar{\alpha}_t} \cdot x_0
$$

where $\sqrt{\bar{\alpha}_t}$ is the signal scaling coefficient and $\sqrt{1 - \bar{\alpha}_t}$ is the noise scaling coefficient (same definitions as in Section 8.3). This quantity has a relatively uniform numerical range across all $t$, making training more stable, especially in the low-$t$ region. Stable Diffusion 2.x and Imagen use v-prediction; SDXL base model still uses epsilon-prediction (depending on the checkpoint's `prediction_type` configuration).

### x0-prediction

Directly predict the original image $x_0$. On certain tasks (especially restoration tasks) this is more intuitive than epsilon—because what we care about is the quality of $x_0$.

The three prediction targets can be converted into one another, but their training dynamics differ. **Empirically**:

- General text-to-image: v or epsilon
- Enhancement / restoration: x0 or v
- Extremely low-SNR regions: x0 is more stable

### Min-SNR weighting

Loss magnitudes differ greatly across timesteps, and direct averaging causes the model to over-focus on certain $t$. Min-SNR weighting (Hang et al. 2023):

$$
w(t) = \min\left(\text{SNR}(t), \gamma\right) / \text{SNR}(t)
$$

where $\gamma$ is typically 5. This is the standard for modern diffusion training such as SDXL.

## 3.7 Frequency-Domain Losses

Compute the loss directly in the FFT domain, emphasizing high-frequency components.

### Focal Frequency Loss

```python
import torch

def focal_frequency_loss(pred: torch.Tensor, target: torch.Tensor,
                         alpha: float = 1.0) -> torch.Tensor:
    """Focal Frequency Loss (FFL, Jiang et al. 2021).
    Apply FFT to both prediction and ground truth, compute weighted L2 in the frequency domain,
    with larger weight on high-frequency differences.
    pred, target: (B, C, H, W)
    """
    pred_fft   = torch.fft.fft2(pred,   norm='ortho')
    target_fft = torch.fft.fft2(target, norm='ortho')

    diff = pred_fft - target_fft
    distance = (diff.real ** 2 + diff.imag ** 2)  # |.|^2

    # focal weight (harder-to-train frequency components get more weight)
    weight = distance.detach() ** alpha
    weight = weight / (weight.max() + 1e-8)

    return (weight * distance).mean()
```

When to use:

- Model output is obviously blurry (pixel loss converges but PSNR no longer improves)
- A particular frequency band keeps failing to recover (real-world FFT power-spectrum reveals the gap)
- High-factor super-resolution (4×, 8×), where high-frequency weight pays off significantly

Engineering experience: FFL alone tends to make the model produce "grid-like" artifacts; **adding it as an auxiliary loss with weight 0.05–0.1** is reasonable.

## 3.8 Task-Specific Losses

Extra constraints for specific tasks.

### Color consistency loss

Compute L1 on the ab/CbCr channels in Lab or YCbCr color space, constraining color from drifting.

```python
def color_consistency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Heavily blur both prediction and ground truth, then compare, only caring about color, not detail.
    Simple and effective; used in deblurring/super-resolution etc. to prevent color drift.
    """
    pred_blur   = F.avg_pool2d(pred,   kernel_size=11, stride=1, padding=5)
    target_blur = F.avg_pool2d(target, kernel_size=11, stride=1, padding=5)
    return F.l1_loss(pred_blur, target_blur)
```

### Identity preservation loss (face-specific)

Used by GFPGAN/CodeFormer: pass both prediction and ground truth through an ArcFace face recognition model and compare embeddings.

```python
class IdentityLoss(nn.Module):
    """Face enhancement only - extract ID embeddings via pretrained ArcFace, compute cosine distance."""
    def __init__(self, arcface_model: nn.Module):
        super().__init__()
        self.arcface = arcface_model.eval()
        for p in self.arcface.parameters():
            p.requires_grad_(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        emb_pred   = self.arcface(pred)    # (B, 512)
        emb_target = self.arcface(target)
        return 1.0 - F.cosine_similarity(emb_pred, emb_target).mean()
```

This is the key to "identity-preserving" face restoration. Chapter 10 covers this in detail.

### Temporal consistency loss (video-specific)

In video tasks, neighboring frames should satisfy optical-flow constraints. Chapter 13 covers this in detail.

## 3.9 Mixing Strategy—the Core of This Chapter

How do you weight all the loss terms above when combined?

### Classic ESRGAN recipe

$$
\mathcal{L}_G = \lambda_1 \mathcal{L}_1 + \lambda_2 \mathcal{L}_{\text{percep}} + \lambda_3 \mathcal{L}_{\text{adv}}
$$

Weights: $\lambda_1 = 1.0$, $\lambda_2 = 1.0$, $\lambda_3 = 5 \times 10^{-3}$

Note the adversarial loss weight is **very small** (5/1000). Reason: the adversarial loss has a large, noisy gradient; a large weight breaks pixel consistency.

### Real-ESRGAN recipe

Adds a second-order degradation synthesis (this is a data change, not a loss change; see Chapter 5), the loss terms are essentially the same:

$$
\mathcal{L}_G = \mathcal{L}_1 + \mathcal{L}_{\text{percep}} + 0.1 \cdot \mathcal{L}_{\text{adv}}
$$

The adversarial weight is raised to 0.1 instead of 0.005 because the data is "harder" (closer to real degradation), and the adversarial loss needs to contribute more.

### Modern diffusion recipe (SUPIR)

The diffusion model's primary loss is simple loss / v-loss / x0-loss, plus:

- **Latent perceptual loss** (LPIPS in latent space) weight 0.1
- **Decoded pixel L1** (prevent VAE reconstruction drift) weight 0.1
- **CLIP loss** (semantic non-drift) weight 0.01

Details in Chapters 8–9.

### Engineering experience for weight tuning

The most common pitfall for beginners: **adding all losses from the start**. This easily lets adversarial or perceptual loss derail early training, and the model fails to converge.

Recommended two-stage strategy:

**Stage 1 (pretrain)**: only pixel loss (L1 + Charbonnier), train until PSNR converges. In this stage the model learns "basic restoration"; outputs are blurry but stable.

**Stage 2 (finetune)**: add perceptual and adversarial losses, with weights from small to large. The model learns "restoration + sharpening"; PSNR drops a bit but visual quality improves significantly.

The official training pipelines of ESRGAN/Real-ESRGAN both follow these two stages.

### Loss curve diagnostics

During training, log every loss term separately and look for issues:

| Symptom | Possible cause | Adjustment direction |
|------|---------|---------|
| Pixel loss decreases, perceptual loss does not | Model stuck in "safe blur" | Increase perceptual loss weight |
| Adversarial loss oscillates wildly | $D$ is too strong or too weak | Adjust D/G learning-rate ratio, add spectral norm |
| Adversarial loss explodes early | No pretrain before adding adversarial | Pretrain first, then add adversarial |
| Perceptual loss decreases then increases | Overfitting to VGG's adversarial samples | Add LPIPS / reduce VGG weight |
| Color drift | Lacking color constraint | Add color consistency loss |
| FID does not drop while LPIPS does | Insufficient diversity | Check data augmentation, increase adversarial weight |

## 3.10 Loss Selection Decision Table

A starting recipe per task (specific weights need tuning to data/network):

| Task | Primary loss | Auxiliary losses | Notes |
|------|--------|---------|------|
| Classic SR (PSNR-oriented) | Charbonnier | — | For academic benchmarks |
| Real SR (visual-oriented) | L1 + VGG + RaGAN | + 0.05 FFL | Real-ESRGAN style |
| Denoising | Charbonnier | + 0.001 TV | NAFNet style |
| Deblurring | Charbonnier + gradient | + VGG | Restormer style |
| Face restoration | L1 + VGG + Adv + Identity | — | GFPGAN/CodeFormer style |
| Diffusion enhancement | v-prediction MSE | + latent LPIPS + CLIP | SUPIR style |
| Video super-resolution | Charbonnier + VGG | + temporal consistency | BasicVSR++ style |
| Frame interpolation | Charbonnier + VGG + LapPyr | — | RIFE style |

Remember an engineering intuition:

> **Want sharp → add adversarial loss**
> **Want fidelity → up-weight pixel loss**
> **Want correct color → add color consistency**
> **Want correct edges → add gradient loss**
> **Want real details → add perceptual loss**
> **Want non-drifting content → add CLIP loss**

## 3.11 Summary

1. **Enhancement losses are almost never single-term**—four or five terms mixed is the norm
2. **Pixel space uses L1 / Charbonnier instead of L2**—more robust, sharper
3. **Perceptual loss (VGG/LPIPS) makes the model attend to high frequencies and semantics**—but cannot be used alone
4. **Adversarial loss (RaGAN/Hinge) makes the model produce real details**—weight must be small and stable
5. **Diffusion losses form their own system** (simple/v/x0); not mixed with the discriminative-loss system
6. **Frequency-domain and task-specific losses** are patches—targeted but with restrained weights
7. **Mixing strategy = pretrain pixel → finetune adding perceptual and adversarial**
8. **Loss curve diagnostics** tell you where training goes wrong better than looking at final metrics alone

In the architecture chapters (Chapters 6–10), when discussing specific models, we will repeatedly come back to this chapter—each model's "training scheme" subsection is essentially a concrete recipe of loss weighting.

---

> Next chapter [Pitfalls of Evaluation](04-metrics.md) → we'll see that metrics in image enhancement are more deceptive than loss functions.
