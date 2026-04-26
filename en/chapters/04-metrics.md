# Chapter 4 · Pitfalls of Evaluation

> Image enhancement has an open secret: **the metrics in papers and the human eye often disagree**.
>
> This is not because some people fake numbers—the metrics in this field have systematic bias by design.

## 4.1 A Contradictory Phenomenon

Run 4× super-resolution on the same low-resolution cat-face image with three models:

- **A**: direct bicubic upsampling
- **B**: ESRGAN (classic GAN-based SR)
- **C**: SUPIR (diffusion-camp SOTA)

The metrics may look like:

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | Visual impression |
|------|--------|--------|---------|-------|---------|
| A. bicubic | **28.4** | **0.82** | 0.42 | 78 | Too blurry to look at |
| B. ESRGAN | 26.1 | 0.76 | 0.18 | 22 | Sharp but with some noise |
| C. SUPIR | 24.8 | 0.71 | **0.12** | **8** | Rich and creative details |

What's contradictory:

- **PSNR and SSIM rate bicubic the highest**, but bicubic is the blurriest visually
- **LPIPS and FID rate SUPIR the highest**, but SUPIR has the worst pixel consistency (with "creative" modifications)
- **No model wins across all metrics**

This chapter clarifies two things:

1. What each metric measures and what it misses
2. Why these inconsistencies are **theoretically inevitable**, not engineering shortcomings

## 4.2 PSNR

The oldest full-reference metric.

$$
\text{PSNR}(\hat{x}, x) = 10 \log_{10} \left( \frac{\text{MAX}^2}{\text{MSE}(\hat{x}, x)} \right)
$$

where MSE is mean squared error and MAX is the maximum pixel value (255 for 8-bit). The unit is decibels (dB); higher is better.

```python
import torch

def psnr(pred: torch.Tensor, target: torch.Tensor,
         data_range: float = 1.0) -> torch.Tensor:
    """PSNR (Peak Signal-to-Noise Ratio).
    pred, target: (B, C, H, W), assumed [0, 1]
    Returns: (B,) PSNR (dB) per image
    """
    mse = ((pred - target) ** 2).flatten(1).mean(dim=1)
    return 10 * torch.log10((data_range ** 2) / (mse + 1e-12))
```

### What PSNR measures

Directly measures pixel-level L2 error. Equivalent to log-likelihood under an "i.i.d. Gaussian noise" assumption.

### What PSNR misses

**Misses perception**—it doesn't know that:

- A one-pixel small color difference and a one-pixel big misalignment can differ enormously to the eye, but PSNR may give the same score
- An image globally shifted by 1 pixel is barely noticeable to humans, but PSNR drops sharply
- Slight Gaussian blur over the whole image makes it visibly blurry, but PSNR may go up (because blur reduces high-frequency differences)

Engineering implications:

- High-PSNR images **are not necessarily good-looking**, low-PSNR images **are not necessarily bad-looking**
- PSNR has relative meaning when comparing **algorithms** (same degradation, same ground truth), but limited meaning when comparing across datasets

### Why PSNR is still de facto standard

It has a few engineering advantages too strong to suppress:

- **Simple**: one line of MSE computes it
- **Additive**: can be tracked by channel, by region
- **Historical inertia**: every paper since the 1990s reports it; can't be removed

In practice PSNR is still a must-report metric, but **using it alone for model selection** is a beginner's mistake.

## 4.3 SSIM Family

### SSIM

SSIM (Structural Similarity, 2004) tries to fix PSNR's "ignores structure" problem. It splits local similarity into three parts:

$$
\text{SSIM}(x, y) = l(x, y)^\alpha \cdot c(x, y)^\beta \cdot s(x, y)^\gamma
$$

- **luminance** $l$: similarity of means
- **contrast** $c$: similarity of variances
- **structure** $s$: normalized correlation

Computation uses a sliding window (11×11 Gaussian-weighted), and the final value is the global average.

```python
import torch
import torch.nn.functional as F

def gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)
    return window  # (W, W)


def ssim(pred: torch.Tensor, target: torch.Tensor,
         window_size: int = 11, sigma: float = 1.5,
         data_range: float = 1.0) -> torch.Tensor:
    """SSIM, simplified version. For production use torchmetrics or pyiqa.
    pred, target: (B, C, H, W) in [0, data_range]
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    C = pred.shape[1]

    win = gaussian_window(window_size, sigma).to(pred)
    win = win.expand(C, 1, window_size, window_size)

    mu_x = F.conv2d(pred,   win, padding=window_size // 2, groups=C)
    mu_y = F.conv2d(target, win, padding=window_size // 2, groups=C)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred,     win, padding=window_size // 2, groups=C) - mu_x2
    sigma_y2 = F.conv2d(target * target, win, padding=window_size // 2, groups=C) - mu_y2
    sigma_xy = F.conv2d(pred * target,   win, padding=window_size // 2, groups=C) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return (num / den).flatten(1).mean(dim=1)
```

### MS-SSIM

Multi-scale SSIM—repeatedly downsample the image and compute SSIM at each scale, weighted-average. Advantage: captures structural similarity at different scales.

### What SSIM misses

- **Lenient toward blur**: means unchanged, variances barely change, SSIM may only drop 0.05
- **Sensitive to geometric deformation**: an object shifted by 5 pixels causes SSIM to drop a lot, while humans can hardly tell
- **Insensitive to texture**: two different textures with similar statistics get a high SSIM score

Engineering implications:

- SSIM is slightly better than PSNR, but **still not aligned with human perception of texture and detail**
- In "texture replaced but structure preserved" scenarios, SSIM overestimates similarity

## 4.4 LPIPS

LPIPS (Learned Perceptual Image Patch Similarity, 2018) is a paradigm shift in this field. It abandons hand-crafted formulas and uses:

1. AlexNet/VGG/SqueezeNet to extract multi-layer features
2. Spatial alignment + channel normalization on each layer
3. A **learned linear weight** combining the L2 distances over layers and channels
4. The linear weights are fit on a large dataset (BAPPS) of human perceptual judgments

Mathematically:

$$
\text{LPIPS}(x, y) = \sum_l \frac{1}{H_l W_l} \sum_{h, w} ||w_l \odot (\hat{F}_l(x)_{h,w} - \hat{F}_l(y)_{h,w})||_2^2
$$

Direct usage:

```python
import lpips  # pip install lpips

# Use AlexNet backbone; usually use 'vgg' as the loss-function backbone
loss_fn_alex = lpips.LPIPS(net='alex')   # net='alex' is closer to human perception, 'vgg' is more stable as a training loss

def lpips_distance(pred: torch.Tensor, target: torch.Tensor,
                   loss_fn: lpips.LPIPS) -> torch.Tensor:
    """Compute LPIPS distance.
    Input: RGB in [0, 1], shape (B, 3, H, W). The function internally rescales to [-1, 1].
    Returns: (B,) LPIPS distance per image (smaller = more similar)
    """
    return loss_fn(pred * 2 - 1, target * 2 - 1).flatten()
```

### What LPIPS measures

A learned "how visually dissimilar two images are." Trained on large amounts of human "which one is more similar to the reference" annotations.

### Where LPIPS beats PSNR/SSIM

- Sensitive to **blur** (PSNR/SSIM's biggest blind spot)
- Sensitive to **texture replacement** (same object with different textures, LPIPS distinguishes)
- Robust to **global shifts** (small shifts don't crash LPIPS)

### What LPIPS also misses

- **Can be fooled by adversarial samples**—AlexNet itself has adversarial fragility, models can learn "artifacts that game LPIPS"—low LPIPS but bad visuals
- **Insensitive to color drift**—VGG/AlexNet both apply ImageNet normalization, dulling overall color changes
- **Not great at extreme artifacts**—blocking artifacts, colored ringing may have small LPIPS but look terrible
- **Different backbones give different results**—LPIPS-alex and LPIPS-vgg are not directly comparable

Engineering experience: LPIPS is currently the most useful full-reference perceptual metric, **but not the only metric**.

## 4.5 DISTS

DISTS (2020) improves on LPIPS by splitting perceptual distance into two parts:

- **Structure distance**: captures spatial arrangement
- **Texture distance**: captures statistical properties, insensitive to position

DISTS's core innovation: **insensitivity to texture position**.

Example: grass on a lawn with leaves in different positions but consistent overall texture—DISTS gives a high score; LPIPS/SSIM give a low score. In many enhancement scenarios (restoring textures, generating grass/hair/water), DISTS aligns with human eyes better than LPIPS.

```python
# Recommended: pyiqa library (unifies all IQA metrics)
import pyiqa  # pip install pyiqa

dists_metric = pyiqa.create_metric('dists', as_loss=False)

def dists_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """DISTS distance, [0, 1] input."""
    return dists_metric(pred, target)
```

When to use DISTS:

- Tasks involving **texture generation or restoration** (hair, grass, skin, fabric)
- LPIPS already tuned to the limit but improvements stalled
- When the model produces textures that are "statistically right but positionally wrong" and LPIPS misjudges

## 4.6 FID / KID

LPIPS and DISTS are both **full-reference**—needing the ground truth $x$ paired with $\hat{x}$ for comparison. But generative models have a different need:

> I don't care whether a single $\hat{x}$ is close to $x$; I care whether the **overall distribution** of $\hat{x}$ resembles the distribution of natural images.

This is the design objective of FID (Fréchet Inception Distance) and KID.

### FID

Pass all real images and all generated images through InceptionV3, getting two sets of 2048-dimensional features. Assume each set follows a Gaussian, and compute the Fréchet distance between the two Gaussians:

$$
\text{FID}(R, G) = ||\mu_R - \mu_G||^2 + \text{tr}(\Sigma_R + \Sigma_G - 2(\Sigma_R \Sigma_G)^{1/2})
$$

```python
import pyiqa
fid_metric = pyiqa.create_metric('fid')
# fid needs batches; usage differs from PSNR/LPIPS, typically pass two folder paths
# fid = fid_metric(real_image_folder, generated_image_folder)
```

### What FID measures

**The distributional difference of two image sets in Inception feature space.**

Suitable for:

- Generative enhancement models (diffusion SR, generative denoising)
- Tasks where the output space is multi-modal (the same degraded input has multiple plausible outputs)

Not suitable for:

- Discriminative SR/denoising—given $y$, output a unique $\hat{x}$, "distributional difference" is less meaningful
- Single image—FID needs at least thousands of images; FID on a single image is noise

### Common misuse of FID

> "Our model has FID = 12 on 100 test images, theirs has FID = 18, so we're better"

This has two problems:

1. **Sample size too small**: FID has very high variance on small samples; 100 images can yield ±10 standard deviation, and a difference of 6 is meaningless
2. **Bias**: FID has systematic bias with sample size; one must use the same number of samples for comparison

Engineering experience:

- FID requires at least **10K samples** to be stable
- In real engineering FID is mostly looked at for **absolute magnitude and trend** (FID 50+ vs 10+ is qualitatively different); don't sweat decimal points
- KID (based on Maximum Mean Discrepancy) is more stable than FID at small sample sizes; better metric for small data

## 4.7 No-Reference Metrics (NR-IQA)

In real-world scenarios we often **don't have a ground truth**—you only have a bad photo from your phone, no "ideal original" for reference.

In this case use no-reference metrics. Representatives:

- **NIQE** (2013): based on natural-image statistics distance
- **BRISQUE** (2012): a classifier score based on spatial features
- **MUSIQ** (2021): Vision Transformer-learned image quality score
- **MANIQA** (2022): attention + multi-scale, a strong NR-IQA baseline
- **CLIPIQA** (2023): uses CLIP to evaluate "aesthetics / realism"

> What counts as "NR-IQA SOTA" depends heavily on benchmark (PIPAL, LIVE, KonIQ-10k, etc.), and 2024–2026 has seen many new MLLM/CLIP-based methods. In production environments **report multiple NR metrics together**; don't fixate on one.

```python
import pyiqa

niqe   = pyiqa.create_metric('niqe',    device='cuda')
musiq  = pyiqa.create_metric('musiq',   device='cuda')
maniqa = pyiqa.create_metric('maniqa',  device='cuda')

# NIQE: smaller is better; MUSIQ/MANIQA: larger is better
def assess_no_reference(image: torch.Tensor):
    """Give a no-reference quality assessment of a single image."""
    return {
        'niqe':   niqe(image).item(),
        'musiq':  musiq(image).item(),
        'maniqa': maniqa(image).item(),
    }
```

### Core problem of no-reference metrics

They can only tell you "does this image look high quality"—they **cannot tell you whether it is $x$**.

- **Necessary in real degradation scenarios**—the only choice when there's no ground truth
- **Easy to fool**—models can learn to "look high quality to NIQE/MUSIQ" while actually producing weird artifacts
- **Different metrics correlate poorly**—a model with great NIQE but bad MUSIQ is common

Engineering experience:

- For real-scene evaluation, **report multiple NR metrics together** (NIQE + MUSIQ + MANIQA + CLIPIQA); abnormality on any one may indicate failure modes
- No-reference metrics are always used together with **subjective evaluation**; they cannot decide model quality alone

## 4.8 Fidelity vs Perception—a Theoretical Inevitability

Back to the contradictory table at the start. This is not an engineering coincidence; it is **information-theoretically necessary**.

Blau & Michaeli's 2018 paper "The Perception-Distortion Tradeoff" rigorously proved:

> On ill-posed inverse problems, **distortion** (PSNR / SSIM-style fidelity metrics) and **perception** (FID / LPIPS-style perceptual metrics) **cannot be simultaneously optimal**.
>
> Any optimization algorithm picks a point on this trade-off curve.

Intuitive understanding:

- **Distortion-optimal** means the output approximates the "statistical expectation" of the ground truth—the average of multiple plausible $x$, which is blurry
- **Perception-optimal** means the output lies on the natural image distribution—must be a specific $x$, not the mean

These two goals are fundamentally in conflict.

```
         ↑ Perception (FID, LPIPS)
       Bad
    *
       *
         *
           *  <- real models pick a point on this curve
             *
               * 
                 *
                  Best
                   ←————————————→ Distortion (PSNR, SSIM)
                  Good          Bad
```

**The existence of this curve** has several engineering implications:

1. **The PSNR camp and the perceptual camp "fighting" is the norm**; no model will win every metric
2. **When reading benchmarks, look at the position on the curve, not a single point**—HAT lives at high PSNR / low perception, SUPIR at low PSNR / high perception; these two are not competing for the same objective
3. **The application scenario decides the optimal point on the curve**—forensic evidence sits at the distortion end, photo beautification at the perception end

Chapter 8 will return to this trade-off and discuss why diffusion models actively choose to sacrifice PSNR for perception.

## 4.9 Bias of Benchmark Datasets

Almost every academic paper reports metrics on these few datasets. But **all of them have systematic biases**.

### Classic benchmarks

| Dataset | Image count | Type | Bias |
|-------|---------|------|------|
| Set5 | 5 | Mixed | Too few, very high variance |
| Set14 | 14 | Mixed | Same |
| BSD100 | 100 | Natural | Slightly small but the average is still credible |
| Urban100 | 100 | Urban architecture | Biased toward regular structures |
| Manga109 | 109 | Manga | All black-and-white line art |
| DIV2K val | 100 | High-quality natural | Standard but biased toward "pretty" |
| LSDIR val | 250 | Diverse | Newer and fairer |

### Common problem of all these datasets

> **Their "low-resolution versions" are almost all bicubic-downsampled.**

Meaning:

- Training $y$ uses bicubic
- Test $y$ uses bicubic
- **Test and training are under the same degradation assumption**

This is the evaluation-side counterpart of the "train-test mismatch" from Chapter 1. Conclusion:

> "30 dB on Set14 4×" **cannot** tell you "whether it works well on your phone's low-light photos."

### Real degradation datasets

To test real scenes, since 2019 "real degradation pair" datasets have appeared:

- **RealSR** (2019): same camera at different focal lengths shooting the same scene, yielding LR-HR pairs
- **DRealSR** (2020): DSLR cameras, more precise alignment
- **DPED** (2017): low-end phone vs DSLR pairs
- **NTIRE Real-World SR challenge**: yearly competition data

These datasets are small (a few hundred to a few thousand images), but **closer to real at test time**.

Engineering practice:

- **Report both academic benchmarks and real degradation datasets**
- High PSNR on academic benchmarks is meaningless (unless two models are compared under the same degradation)
- Metrics on real degradation datasets are more telling

## 4.10 Which Metrics to Watch During Training

At each epoch / every N steps of validation, which metrics should you compute?

**Minimum set (mandatory)**:

- PSNR: monitor fidelity
- LPIPS: monitor perceptual quality

**Recommended set**:

- PSNR + SSIM + LPIPS + DISTS (full-reference suite)
- Model loss curves (record each loss term separately)

**Generative models additionally**:

- FID (run on a sufficiently large test set every N epochs)
- A fixed input set's visual comparison (intuitively see model changes)

```python
class ValidationMetrics:
    """Validation-time metrics logger during training."""
    def __init__(self, device='cuda'):
        import pyiqa
        self.psnr  = pyiqa.create_metric('psnr',  device=device, as_loss=False)
        self.ssim  = pyiqa.create_metric('ssim',  device=device, as_loss=False)
        self.lpips = pyiqa.create_metric('lpips', device=device, as_loss=False)
        self.dists = pyiqa.create_metric('dists', device=device, as_loss=False)

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        return {
            'psnr':  self.psnr (pred, target).mean().item(),
            'ssim':  self.ssim (pred, target).mean().item(),
            'lpips': self.lpips(pred, target).mean().item(),
            'dists': self.dists(pred, target).mean().item(),
        }
```

## 4.11 What to Report in Papers / Reports

The main table must have:

- PSNR / SSIM / LPIPS (basic full-reference)
- Results on at least one real degradation dataset

Add as appropriate:

- DISTS (if it's a texture-generation task)
- FID (if it's a generative model)
- NIQE / MUSIQ / MANIQA (real scenes)
- Subjective evaluation (MOS / 2AFC, covered in Chapter 12)

**Don't do**:

- Report only PSNR
- Report metrics only on Set5/Set14
- Report FID with sample size below 1000
- Hide failure cases

## 4.12 A Concrete Case: bicubic vs ESRGAN vs SUPIR

Back to the table at the start, now unpacking why each metric is what it is:

| Model | PSNR | SSIM | LPIPS | FID | Interpretation |
|------|------|------|-------|-----|------|
| bicubic | **28.4** | **0.82** | 0.42 | 78 | Output is the "blurred mean" of the truth, naturally high PSNR/SSIM; but no high frequency, bad LPIPS/FID |
| ESRGAN | 26.1 | 0.76 | 0.18 | 22 | GAN generates high-frequency details, "details that are right but maybe not the real ones," PSNR drops but perception greatly improves |
| SUPIR | 24.8 | 0.71 | **0.12** | **8** | Diffusion prior generates more realistic details, but with creativity; pixel-level consistency is the worst |

**No model is "the best"**—the choice depends on scenario:

- Forensic evidence enhancement → bicubic (no fabrication allowed)
- Photo enlargement for printing → ESRGAN (fidelity + clarity balanced)
- Old photo restoration → SUPIR (maximize visual realism)

> **"Which model is best" is the wrong question.**
> **The right question is "in the distortion-perception plane, which point best fits my application."**

## 4.13 Summary

1. **PSNR/SSIM measure fidelity, are insensitive to perception**—using them alone misleads, but they must be reported as baseline
2. **LPIPS/DISTS measure perception**, more aligned with human eyes than PSNR/SSIM, but can't be blindly trusted
3. **FID measures distributional distance**, suitable for generative models but needs a large enough sample size
4. **NR-IQA** (NIQE/MUSIQ/MANIQA) is mandatory in real scenes but easy to fool
5. **Perception-Distortion is a theoretical trade-off**—any model picks a point on this curve
6. **Academic benchmarks all use bicubic degradation**, mismatched with real scenes
7. **At training time at least PSNR + LPIPS**, papers report at least PSNR + SSIM + LPIPS + a real degradation dataset
8. **No single metric tells the whole story**—always report combinations and pair with subjective evaluation

This chapter together with Chapter 3 answers "what is good"—loss defines the training objective, metrics define the evaluation criteria. After understanding these two chapters, the architecture chapters that follow have a basis for "judging good and bad."

---

> Next chapter [Data and Degradation Synthesis](05-degradation-pipeline.md) → Real-ESRGAN's true core contribution is data synthesis. We'll see that the same network with different data pipelines can differ in performance by an order of magnitude.
