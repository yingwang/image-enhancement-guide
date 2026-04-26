# Chapter 8 · Diffusion Model Basics

> This is the most central paradigm shift in this book.
>
> All of the models in Chapters 6-7 are **discriminative** — given $y$, they output a unique $\hat{x}$.
> The diffusion models from this chapter onward are **generative** — given $y$, they output $p(x | y)$, from which several plausible $\hat{x}$ can be sampled.
>
> This difference becomes a qualitative leap in heavily ill-posed scenarios.

## 8.1 Why diffusion matters in image enhancement

Recall the perception-distortion trade-off in Section 4.8 of Chapter 4:

> On ill-posed problems, distortion (PSNR) and perception (FID/LPIPS) cannot be optimal at the same time.

Discriminative models have pushed the distortion end to the extreme — HAT reaches 33.4 dB on Set5 4×. But they have a ceiling on the perception end:

- Given a heavily blurred face LR, the "most likely" $\hat{x}$ is the average face (the mean of multiple plausible $x$)
- A discriminative model must output **a single** definite $\hat{x}$, so it outputs the average face
- The average face is **visually blurry** — it is optimal in PSNR but unrealistic perceptually

Diffusion models attack this problem directly:

> I do not output **a single** $\hat{x}$; I learn the entire distribution $p(x | y)$, and then **sample** a concrete $\hat{x}$ from this distribution.

Each sampled $\hat{x}$ is a concrete point in the distribution — a **specific face** rather than an average face, visually realistic but not necessarily pixel-level consistent with the ground truth.

This is why SUPIR is stunning on heavily degraded old photos but is 5+ dB lower in PSNR than HAT — it walks the other end of the perception-distortion curve.

This chapter clarifies how diffusion models work, why they suit enhancement tasks, and how to use them concretely.

## 8.2 Intuition for diffusion models

The core idea of diffusion models can be summarized in one sentence:

> **Learn a denoising process** — start from pure noise, remove noise step by step, and end up with an image.

The concrete process:

```
Forward (fixed, not learned):
  x_0 (clean image)
    ↓ add a bit of noise
  x_1
    ↓ add a bit of noise
  x_2
    ↓ ...
  x_T (pure noise, T = 1000)

Reverse (learned by the neural network):
  x_T (pure noise)
    ↓ remove a bit of noise (predicted by the neural net)
  x_{T-1}
    ↓ remove a bit of noise
  x_{T-2}
    ↓ ...
  x_0 (clean image)
```

Training objective: **given a noisy image at any time step $x_t$, predict the noise that was added**.

This looks strange — why does this generate images? The key lies in two points:

1. **A noisy image at any $t$ can be sampled in a single step** — there is no need to repeatedly add noise $T$ times from $x_0$, there is a closed-form formula
2. **Learning to predict the noise = learning the score function of $p(x_0)$** — and from noise we can reverse back to an image

We will work out the math below.

## 8.3 The math of the forward process

Define the noise schedule $\beta_1, \beta_2, \dots, \beta_T$ ($T = 1000$, $\beta_t$ increases linearly from $10^{-4}$ to $0.02$).

Each step adds noise:

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} \cdot x_{t-1}, \beta_t \mathbf{I})
$$

Meaning: $x_t$ is $x_{t-1}$ shrunk by a small factor (multiplied by $\sqrt{1-\beta_t}$) plus a small amount of Gaussian noise (variance $\beta_t$).

Define $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$. The **key property**:

$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \cdot x_0, (1 - \bar{\alpha}_t) \mathbf{I})
$$

This means **given $x_0$, we can sample any $x_t$ in a single step**:

$$
x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
$$

This is what makes training efficient — there is no need to actually add noise 1000 times step by step.

```python
import torch


class NoiseScheduler:
    """A DDPM-style linear noise schedule."""

    def __init__(self, num_steps: int = 1000,
                 beta_start: float = 1e-4, beta_end: float = 0.02,
                 device: str = 'cuda'):
        self.num_steps = num_steps
        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # Quantities frequently used during training
        self.sqrt_alphas_cumprod        = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor,
                  noise: torch.Tensor = None) -> tuple:
        """One-step sampling of x_t.
        x0: (B, C, H, W) clean image
        t:  (B,) integer time step
        noise: optional, randomly sampled if not provided
        Returns: (x_t, noise)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x_t = sqrt_alpha * x0 + sqrt_one_minus * noise
        return x_t, noise
```

## 8.4 The reverse process: training objective

In theory the reverse process is $p(x_{t-1} | x_t)$, and what is to be learned is this conditional distribution. But the DDPM paper proved a simplified equivalent objective:

**Train a network $\epsilon_\theta(x_t, t)$ directly to predict the added noise $\epsilon$**.

Training loss:

$$
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$

This is the simple loss covered in Section 3.6 of Chapter 3.

### Three equivalent prediction targets

The model can predict any one of three quantities, all equivalent:

- **$\epsilon$-prediction**: predict the added noise (the DDPM standard)
- **$x_0$-prediction**: predict the original image
- **$v$-prediction**: $v_t = \alpha_t \epsilon - \sigma_t x_0$ (more stable)

The conversions between them:

```python
# Given the model output and the current (x_t, t), convert between the three

def eps_to_x0(x_t, eps_pred, alpha_cumprod_t):
    """Recover x_0 from predicted epsilon."""
    sqrt_alpha_t = alpha_cumprod_t.sqrt()
    sqrt_one_minus = (1 - alpha_cumprod_t).sqrt()
    return (x_t - sqrt_one_minus * eps_pred) / sqrt_alpha_t


def v_to_x0(x_t, v_pred, alpha_cumprod_t):
    """Recover x_0 from predicted v."""
    sqrt_alpha_t = alpha_cumprod_t.sqrt()
    sqrt_one_minus = (1 - alpha_cumprod_t).sqrt()
    return sqrt_alpha_t * x_t - sqrt_one_minus * v_pred


def x0_to_eps(x_t, x0_pred, alpha_cumprod_t):
    """Recover epsilon from predicted x_0."""
    sqrt_alpha_t = alpha_cumprod_t.sqrt()
    sqrt_one_minus = (1 - alpha_cumprod_t).sqrt()
    return (x_t - sqrt_alpha_t * x0_pred) / sqrt_one_minus
```

How to choose the prediction target during training:

- **$\epsilon$-pred**: standard for general text-to-image (SD 1.x)
- **$v$-pred**: more stable for high-resolution training (SD 2.x, SDXL refiner)
- **$x_0$-pred**: intuitive for enhancement / restoration tasks, since the quantity of interest is the quality of $x_0$

## 8.5 A minimal DDPM training loop

```python
import torch
import torch.nn.functional as F
from torch.optim import AdamW

def train_ddpm(model, train_loader, scheduler, epochs=100,
               lr=1e-4, device='cuda'):
    """A minimal DDPM training loop.
    model: UNet, takes (x_t, t) and outputs predicted noise (B, C, H, W)
    """
    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        for x0 in train_loader:                    # x0: (B, C, H, W)
            x0 = x0.to(device)
            B = x0.shape[0]

            # 1. Sample a random time step
            t = torch.randint(0, scheduler.num_steps, (B,), device=device)

            # 2. One-step sampling of x_t
            x_t, noise = scheduler.add_noise(x0, t)

            # 3. Model predicts the noise
            pred_noise = model(x_t, t)

            # 4. MSE loss
            loss = F.mse_loss(pred_noise, noise)

            # 5. Backpropagate
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
```

This loop looks overly simple. The fundamental reason it works:

> Learning "predict the added noise from an image with arbitrary noise" = learning an approximation of $\nabla \log p(x)$ (the score function).
>
> Once the score is learned, Langevin dynamics / a reverse SDE can be used to sample from noise to $p(x)$.

## 8.6 Sampling: DDPM, DDIM, DPM-Solver

After training, how do we generate an image from noise?

### DDPM sampling

Step by step along the reverse chain:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z
$$

where $z \sim \mathcal{N}(0, \mathbf{I})$ and $\sigma_t$ is a noise term.

**Problem**: it requires 1000 steps, each with a UNet forward pass — **slow**.

### DDIM sampling

Song et al. showed that the reverse can be made deterministic:

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1}} \cdot \epsilon_\theta(x_t, t)
$$

where $\hat{x}_0$ is the $x_0$ derived from $x_t$ and the predicted noise. This sampling **can skip steps** — there is no need to walk through all 1000 steps; one can pick 50 time steps to sample at.

DDIM 50 steps ≈ DDPM 1000 steps in quality, **a 20× speedup**.

### The DPM-Solver family

Treat the reverse process as an ODE and use higher-order numerical methods to solve it.

- **DPM-Solver-2**: second-order, ~20 steps reach DDIM 100-step quality
- **DPM-Solver++**: handles the SDE form
- **UniPC**: a further-optimized unified predictor-corrector method

```python
# Use the diffusers samplers
from diffusers import DPMSolverMultistepScheduler

scheduler = DPMSolverMultistepScheduler.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    subfolder="scheduler",
    algorithm_type="dpmsolver++",
    solver_order=2,
)
scheduler.set_timesteps(num_inference_steps=20)
```

Engineering practice (the SOTA configuration in 2026):

- **Academic benchmarks / high-quality inference**: UniPC, 30-50 steps
- **Production deployment**: DPM-Solver++ 2M, 20 steps
- **Extreme speed**: LCM (Latent Consistency Models), distilled, 4 steps

## 8.7 LDM: moving diffusion to latent space

This was already introduced in Section 2.4 of Chapter 2. We supplement engineering details here.

The key observation of LDM (Rombach et al. 2022):

> Pixel-space diffusion wastes a lot of compute on "high-frequency details" — these details can be generated by post-processing through a VAE decoder.
>
> Let the UNet do diffusion only in **latent space**, and leave the high-frequency details to the VAE decoder.

### The specific configuration of Stable Diffusion

| Component | Role | Parameters |
|------|------|--------|
| VAE encoder | RGB → latent (8× downsampling, 4 channels) | ~50M |
| VAE decoder | latent → RGB | ~50M |
| UNet | Latent-space diffusion | ~860M |
| Text encoder (CLIP) | Text → embedding | ~120M |
| Total | | ~1.1B |

The UNet is the bulk; the VAE is relatively small. This is why SD fine-tuning mainly trains the UNet (keeping the VAE fixed).

### Engineering impact of LDM

- **Compute drops to 1/48**: what would have been diffusion on $512 \times 512 \times 3$ now happens on $64 \times 64 \times 4$
- **Large images become tractable**: $1024 \times 1024$ is no longer a memory disaster
- **The VAE is an independent, swappable component**: it can be fine-tuned (e.g. using a better decoder for enhancement scenarios)

## 8.8 The internal structure of the SD UNet

The UNet is the "main network" of a diffusion model. Its structure matters for enhancement tasks — extensions like ControlNet are built on top of this structure.

```
Input (B, 4, 64, 64) latent
  ↓ Conv (4 → 320 channels)
  ↓ DownBlock × 4 (encoder)
    each DownBlock:
      ResBlock × 2 + Spatial Transformer × 2 (cross-attention to text)
      Downsample (×2)
  ↓ MidBlock
    ResBlock + Spatial Transformer + ResBlock
  ↓ UpBlock × 4 (decoder, with skip connections)
    each UpBlock:
      ResBlock × 3 + Spatial Transformer × 3
      Upsample (×2)
  ↓ Conv (320 → 4 channels)
Output (B, 4, 64, 64) predicted noise / v
```

### ResBlock: the core compute unit

```python
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """SD UNet's ResBlock, with time embedding injection."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int = 1280):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim, out_ch)

        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x:     (B, C, H, W)
        # t_emb: (B, time_emb_dim)
        h = self.conv1(F.silu(self.norm1(x)))
        # Inject the time embedding (broadcast)
        h = h + self.time_proj(F.silu(t_emb)).view(*t_emb.shape[:1], -1, 1, 1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)
```

Time embedding injection is unique to diffusion models — the same network has to handle 1000 different time steps and needs to know which one the current step is.

### Spatial Transformer: the cross-attention entry

```python
class SpatialTransformer(nn.Module):
    """The attention block of SD UNet.
    self-attention + cross-attention to text.
    """

    def __init__(self, dim: int, num_heads: int, context_dim: int = 768):
        super().__init__()
        self.norm = nn.GroupNorm(32, dim)
        self.proj_in = nn.Conv2d(dim, dim, 1)

        # Self-attention
        self.attn1 = MultiheadAttention(dim, num_heads)
        # Cross-attention (text → latent)
        self.attn2 = MultiheadAttention(dim, num_heads, kv_dim=context_dim)
        # FFN
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        self.proj_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W); context: (B, T_text, D_text)
        B, C, H, W = x.shape
        h = self.proj_in(self.norm(x))
        h = h.view(B, C, -1).transpose(1, 2)         # (B, HW, C)

        h = h + self.attn1(h, h, h)
        h = h + self.attn2(h, context, context)      # cross-attention
        h = h + self.ff(h)

        h = h.transpose(1, 2).view(B, C, H, W)
        return x + self.proj_out(h)
```

Cross-attention is the key to text-to-image — text influences the features at every spatial position via cross-attention. This mechanism will be reused for condition injection in enhancement tasks in Chapter 9.

## 8.9 Diffusion usage in enhancement tasks

Text-to-image samples from pure noise + a text condition to an image. **Enhancement tasks** sample from pure noise + a degraded image $y$ as the condition to produce $\hat{x}$.

A few paradigms for condition injection:

### Paradigm 1: concat to the input (the simplest)

```python
# UNet input goes from (B, 4, h, w) to (B, 8, h, w)
# The latter 4 channels are the LR latent
def prepare_input(noisy_latent, lr_latent):
    return torch.cat([noisy_latent, lr_latent], dim=1)
```

Representative: StableSR (2023) uses this simple form.

### Paradigm 2: cross-attention (semantic condition)

Inject some embedding of LR via cross-attention. Representative: DiffBIR uses a CLIP image embedding.

### Paradigm 3: ControlNet (structural condition)

Copy a UNet encoder dedicated to processing the conditional input, and add its output to the corresponding layer of the main UNet. Representative: the ControlNet variant of StableSR, SUPIR.

Chapter 9 will discuss these in detail.

## 8.10 Why diffusion can "create something from nothing"

Back to the core question at the start of this chapter: **why can diffusion generate realistic high-frequency details while discriminative methods cannot?**

### 1. Multi-step sampling = repeated random refinement

Discriminative: a single forward pass, output a definite $\hat{x}$.
Diffusion: $T$ forward passes, each one sampled from a random distribution. Each sample is a random choice of "which direction this step should go".

This randomness means the same LR input can produce different $\hat{x}$ — every one of them is plausible.

### 2. Learning the score function

Theoretically, what the diffusion model learns is:

$$
\epsilon_\theta(x_t, t) \approx -\sqrt{1-\bar{\alpha}_t} \cdot \nabla_{x_t} \log p(x_t)
$$

This gradient field tells the model in which direction to move at point $x_t$ to get closer to the natural-image distribution. The score function directly encodes the **geometry of the natural-image manifold**.

### 3. The reverse chain is an SDE solver

It can be rigorously shown that the reverse chain of DDPM is equivalent to solving a stochastic differential equation (SDE):

$$
dx = [f(x, t) - g(t)^2 \nabla_x \log p_t(x)] dt + g(t) dw
$$

where $w$ is a Wiener process (Brownian motion). The stationary distribution of this SDE is the natural-image distribution.

### Putting these three together

A diffusion model is not "approximating the ground truth"; it is "walking T steps along the geometry of the natural-image distribution". Each step is guided by the geometry of the distribution, and each step has a random perturbation that prevents results from repeating.

The final output is a point on the distribution — concrete, plausible, random.

## 8.11 Trade-offs: diffusion camp vs discriminative

Putting the previous two chapters of Part II side-by-side with this chapter:

| Aspect | Discriminative (CNN/Transformer) | Diffusion |
|------|------------------------|------|
| Inference speed | **1× (baseline)** | 20-1000× slower |
| PSNR | **High** | 5-10 dB lower |
| LPIPS / FID | Low | **Better** |
| Visual realism | Medium | **High** |
| Handling severe degradation | Outputs "safe blur" | **Recovers details** |
| Diversity | Single output | **Multiple samples** |
| Control | Only by changing the training data | **Can add prompt / control** |
| Memory usage | Low | High (2-4×) |

**Neither one wins overall** — these two model types have respective strengths in different application scenarios:

- **Forensic evidence enhancement**: discriminative (no fabrication allowed)
- **Academic-benchmark PSNR competitions**: discriminative
- **Photo upscaling, old-photo restoration**: diffusion
- **Extremely lightweight on-device deployment**: discriminative
- **Highly creative applications** (artistic style, 4K live-stream creative enhancement): diffusion

## 8.12 Engineering details for training diffusion models

### Min-SNR weighting (mentioned in Section 3.6 of Chapter 3)

```python
def min_snr_weight(t, alphas_cumprod, gamma=5.0):
    """Min-SNR weighting, makes loss magnitudes consistent across time steps."""
    snr = alphas_cumprod[t] / (1 - alphas_cumprod[t])
    return torch.minimum(snr, torch.full_like(snr, gamma)) / snr
```

### EMA (exponential moving average)

The final weights of a diffusion model are usually the EMA, not the raw weights. The EMA decay is commonly 0.9999:

```python
class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {n: p.clone().detach() for n, p in model.named_parameters()}

    def update(self, model):
        for n, p in model.named_parameters():
            self.shadow[n] = self.decay * self.shadow[n] + (1 - self.decay) * p.detach()

    def apply_to(self, model):
        """Copy the EMA weights to model (used for inference)."""
        for n, p in model.named_parameters():
            p.data.copy_(self.shadow[n])
```

### Importance sampling of time steps

Do not sample $t$ uniformly. Some $t$ ranges contribute more to the final quality (typically the middle range $t \in [200, 800]$) and their sampling probability can be increased.

### Classifier-Free Guidance (CFG)

During training, the condition is dropped 10% of the time (a mixture of unconditional and conditional training). At inference:

$$
\hat{\epsilon} = \epsilon_\theta(x_t, t, \emptyset) + w \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset))
$$

$w > 1$ pushes the generation closer to the condition, but too large overfits to it. In enhancement tasks $w$ is typically 1.5-3.0.

```python
def classifier_free_guidance(model, x_t, t, condition, guidance_scale=2.0):
    """CFG inference."""
    # Conditional prediction
    eps_cond = model(x_t, t, condition)
    # Unconditional prediction (with null/empty condition)
    eps_uncond = model(x_t, t, None)
    # Weighted combination
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)
```

## 8.13 A diffusion training pipeline for an enhancement task

Putting the above concepts together into a training loop for an enhancement task. This is a simplified version of SUPIR/StableSR:

```python
def train_diffusion_enhancement(
    unet, vae, scheduler,
    train_loader,                  # outputs (lr, hr) pairs
    text_encoder=None,             # optional, adds prompt condition
    epochs=50, lr=1e-4,
    device='cuda',
):
    optimizer = AdamW(unet.parameters(), lr=lr)
    ema = EMA(unet)

    # Freeze VAE and text encoder
    vae.eval(); 
    for p in vae.parameters():
        p.requires_grad_(False)

    for epoch in range(epochs):
        for lr_img, hr_img in train_loader:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            # 0. Important: upsample LR to the same size as HR so that the latent
            # spaces align. Otherwise vae.encode(lr_img) gives a smaller latent than
            # hr_latent and they cannot be concatenated directly.
            lr_img_upsampled = F.interpolate(lr_img, size=hr_img.shape[-2:],
                                             mode='bicubic', align_corners=False)

            # 1. Encode to latent space (now hr_latent and lr_latent have the same spatial size)
            with torch.no_grad():
                hr_latent = vae.encode(hr_img).latent_dist.sample() * 0.18215
                lr_latent = vae.encode(lr_img_upsampled).latent_dist.mean * 0.18215

            # 2. Add noise to hr_latent
            B = hr_latent.shape[0]
            t = torch.randint(0, scheduler.num_steps, (B,), device=device)
            x_t, noise = scheduler.add_noise(hr_latent, t)

            # 3. UNet input = (x_t || lr_latent), condition = lr_latent (or text)
            unet_input = torch.cat([x_t, lr_latent], dim=1)
            pred_noise = unet(unet_input, t)

            # 4. Weighted MSE loss
            weights = min_snr_weight(t, scheduler.alphas_cumprod)
            loss = (weights.view(-1, 1, 1, 1) * (pred_noise - noise) ** 2).mean()

            # 5. Backpropagate
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()

            # 6. EMA update
            ema.update(unet)

    # After training, use EMA weights for inference
    ema.apply_to(unet)
    return unet
```

## 8.14 Inference: the full LR-to-HR pipeline

```python
@torch.no_grad()
def diffusion_enhance(unet, vae, scheduler, lr_img,
                       target_size=None,
                       num_inference_steps=20, guidance_scale=2.0):
    """
    Given an LR image, generate an HR estimate.
    target_size: (H, W) HR output size; defaults to 4× LR.
    """
    device = lr_img.device
    if target_size is None:
        target_size = (lr_img.shape[-2] * 4, lr_img.shape[-1] * 4)

    # 1. Important: upsample LR to the target size before passing through the VAE,
    # so that the latent space size equals that of the HR latent.
    lr_upsampled = F.interpolate(lr_img, size=target_size,
                                 mode='bicubic', align_corners=False)
    lr_latent = vae.encode(lr_upsampled).latent_dist.mean * 0.18215

    # 2. Start from pure noise (with the same spatial size as the LR latent)
    x_t = torch.randn_like(lr_latent)

    # 3. Set inference time steps (DPM-Solver / DDIM)
    scheduler.set_timesteps(num_inference_steps)

    # 4. Reverse denoising
    for t in scheduler.timesteps:
        unet_input = torch.cat([x_t, lr_latent], dim=1)
        pred_noise = unet(unet_input, t)
        # CFG (omitted; used in actual production)
        x_t = scheduler.step(pred_noise, t, x_t).prev_sample

    # 5. Decode to pixels
    hr_img = vae.decode(x_t / 0.18215).sample
    return hr_img.clamp(0, 1)
```

Note one key point: **LR must first be upsampled to the target HR size before going through the VAE** — that way the resulting latent has the same size as hr_latent and they can be concatenated directly. If you do `vae.encode(lr_img)` directly, the latent space will be at the LR size (HR/8 is 4× larger than LR/8), and the shapes will not match.

The output resolution is determined by `target_size`, and the image after the VAE decoder is at exactly that size.

## 8.15 Summary

1. **Diffusion is the paradigm shift from discriminative to generative** — the output is not a unique $\hat{x}$ but a sample from $p(x|y)$
2. **Forward noising + reverse denoising**: the training objective is to predict the added noise
3. **Any time step can be sampled in one step**: 1000 noisings are not needed; there is a closed-form formula
4. **The three prediction targets are equivalent** ($\epsilon$ / $x_0$ / $v$); which to choose depends on the task
5. **DDPM 1000 steps → DDIM 50 steps → DPM-Solver 20 steps → LCM 4 steps**: the evolution of samplers
6. **LDM is the engineering key**: moving diffusion to the VAE latent space, reducing compute to 1/48
7. **Stable Diffusion = LDM + text condition + large-scale training**
8. **Diffusion usage in enhancement tasks**: inject LR into the UNet as a condition
9. **Diffusion's "creating something from nothing" = learning the score function + multi-step random sampling**
10. **Trade-offs**: diffusion has strong visual realism but lower PSNR, slower speed, and higher memory usage

The next chapter discusses condition injection — there are several paradigms for injecting LR into a diffusion UNet (concat, cross-attention, ControlNet, IP-Adapter), each suited to different scenarios.

---

> Next chapter [Condition control in diffusion](09-control.md) → from concat to ControlNet to IP-Adapter, the real engineering battlefield of diffusion-based enhancement models.
