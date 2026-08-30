# Chapter 5 · Data and Degradation Synthesis

> This is the most "engineering" chapter of the book.
>
> An adage has long circulated in image enhancement: **Real-ESRGAN's core contribution is not the network, it is the data**.
>
> This chapter unpacks that statement.

## 5.0 Reading Notes

This chapter is the last in Part I and the chapter closest to engineering implementation in the entire book. The concepts built in the previous four chapters all land here in Python code, directory structures, and training loops. After reading, you should be able to directly write a training dataloader you can feed to ESRGAN/Real-ESRGAN/Restormer.

This chapter assumes you have already mastered:

- The image from Chapter 1 of "the degradation operator $D$ is a stack of random compositions" (the data-flow figure in §1.5, which this chapter will expand into a finer-grained version)
- The pixel-space vs feature-space correspondence with loss functions from Chapters 2-3
- The distinction in Chapter 4 between PSNR/SSIM/LPIPS and NIQE/MANIQA (which determines when "ground truth doesn't exist")

It does not assume you have written a Real-ESRGAN dataloader before, or read its 700-line degradation code.

**Abbreviations that appear for the first time in this chapter or will appear repeatedly.** To avoid being turned away by jargon halfway through, here they are upfront:

- **Real-ESRGAN**: the representative 2021 "real-scene" SR model whose core contribution is the degradation-synthesis pipeline, not the network architecture
- **BSRGAN** (Blind Super-Resolution GAN): a contemporaneous, similarly-oriented blind-SR scheme by Zhang et al. that trains a model robust to real images by "randomizing the order of degradations"
- **SRMD** (Super-Resolution with Multiple Degradations): an early work that feeds "degradation parameters" to the network as conditioning input; a non-blind compromise
- **KernelGAN**: a method that estimates the degradation kernel on the test image with a GAN; another route to blind SR
- **RRDB** (Residual in Residual Dense Block): the dense-residual block proposed by ESRGAN; Real-ESRGAN reuses it as backbone (covered in Chapter 6)
- **DiffJPEG**: a differentiable JPEG encode/decode implementation, allowing the JPEG step to be done in batch on the GPU as part of the degradation pipeline
- **DCT** (Discrete Cosine Transform): the core operator of JPEG compression (introduced in §1.0; repeated here)
- **ISP** (Image Signal Processor): the entire in-camera pipeline that turns sensor readouts into 8-bit RGB (also introduced in §1.0)
- **LMDB** (Lightning Memory-Mapped Database): a lightweight key-value store, often used to pre-encode large numbers of small images for cached, training-accelerated access
- **USM** (Unsharp Mask): a classical image-processing sharpening operator—Gaussian-blur the image, subtract the blurred version from the original to extract the high frequencies, then add the high frequencies back
- **chroma subsampling**: after JPEG converts RGB to YCbCr, it downsamples the two chrominance channels Cb/Cr in patterns like 4:2:0; perceptually invisible to humans but very visible to models

When specific terms come up later, we expand them again with one-sentence definitions.

## 5.1 Data > Network

Chapter 1 mentioned a recurring pattern in 2014–2020:

- Synthesize training data with bicubic downsampling
- Train SOTA networks on the synthetic data
- At deployment time, the model barely works on real images

The 2021 Real-ESRGAN paper ran a comparison experiment:

| Network | Training data | Real-image PSNR | Real-image visuals |
|------|---------|--------------|------------|
| ESRGAN (RRDB network) | bicubic degradation | 18.2 dB | Almost doesn't work |
| **Real-ESRGAN (the same RRDB network)** | **complex degradation synthesis** | **23.8 dB** | **Usable** |

**Network unchanged, data changed, performance went from "doesn't work" to "usable"**. This is the most important methodological correction in the field.

Similar stories repeat in denoising, deblurring, video enhancement:

- DnCNN is unbeatable on BSD68 (the standard benchmark) but doesn't work on real phone photos—fine-tuning with SIDD real data improves it greatly
- One key to SUPIR is using billion-scale internet images + complex degradation synthesis
- Video stabilization is perfect on synthetic shake data but needs additional real data for real phone videos

**Engineering philosophy**:

> In image enhancement, **the data pipeline determines the model's capability ceiling**.
> The same network can differ by an order of magnitude on different data; the reverse is not true.

This chapter is about how to design the data pipeline.

## 5.2 Two Sources of Training Data

### Synthetic data

**Method**: take a batch of high-quality images (HR), use code to simulate degradation and obtain (HR, LR) pairs.

```python
# Pseudo-flow
for hr in high_quality_images:
    lr = degradation_pipeline(hr)
    yield (lr, hr)
```

Pros:

- **Infinite**: as long as you have HR, you can synthesize unlimited (LR, HR) pairs
- **Cheap**: no photographic equipment needed
- **Controllable**: can precisely control every parameter of degradation

Cons:

- **Unrealistic**: the synthesized LR may not look like a real bad photo
- **Bias**: any "known to be unrealistic" part will be learned by the model

### Real data

**Method**: collect (HR, LR) pairs in the physical world with cameras/scanners.

Implementation methods:

- **Different focal lengths** (RealSR, DRealSR): same camera with different focal lengths shooting the same scene; the long-focal image is HR, the short-focal image is LR
- **Different devices** (DPED): low-end phone and DSLR shooting the same scene simultaneously
- **Degradation simulation** (rare): take HR through a specific physical pipeline (print + scan) to obtain LR

Pros: **realistic**—every degradation is a physical process, not a synthetic assumption

Cons:

- **Few**: hundreds to thousands of pairs, not enough for data-driven methods
- **Hard to align**: object position, lighting, white balance shift across two captures, requiring complex alignment pipelines
- **Limited degradation modes**: you only capture one type of camera's degradation; the model is only good at that degradation

**Engineering practice**: synthetic data primary + real data fine-tuning.

## 5.3 The Classical Bicubic Pipeline's Problems (a Bit More Depth)

Chapter 1 mentioned the problems of bicubic degradation. Going deeper here.

**Complexity of real low-resolution images.** An image a user casually drops in from their album has typically gone through something like:

```
User photo flow:
  Sensor (photon noise + read noise)
    → demosaicing (RGB reconstruction)
    → ISP (denoise + sharpen + white balance + tone mapping + color matrix)
    → JPEG compression (8-bit, quality 70-95)
    → upload to WeChat/Weibo (recompressed, quality 50-70)
    → someone downloads it (possibly recompressed once more)
```

Each step contributes its own degradation; the final LR is a non-linear composition of all of these. Demosaicing is the step that reconstructs a three-channel color image from the sensor's Bayer-array data (each pixel only captures one of R, G, B); essentially every camera does this internally, but different vendors' algorithms leave different neighbor-pixel correlations.

**Which step does bicubic degradation simulate?** Strictly, **not a single one of them is truly simulated**. Bicubic assumes the LR is the result of HR put through an ideal anti-aliasing filter then downsampled, an entire process that is linear, differentiable, and content-independent, **with no correspondence to the physical chain above**.

ESRGAN trained on bicubic data learns precisely the "inverse bicubic function." That function performs perfectly on bicubic test sets, but the moment the test image is replaced with a real phone-captured low-resolution image, the model does exactly what "inverse bicubic" should: it sharpens noise it shouldn't have sharpened, treating it as "high-frequency signal," and outputs an image full of magnified noise. This transfer failure isn't accidental; the function spaces simply don't overlap.

**The fix direction: make the training data's degradation distribution cover the real-world degradation distribution.** Real-ESRGAN's design heads in this direction. The next section expands.

To prevent §5.4's code from feeling abstract, here is a diagram of this section's claim: the physical process by which a user gets a bad image on the left, the process Real-ESRGAN simulates in code on the right, with arrows indicating which synthesis step corresponds to which physical step.

```mermaid
graph LR
    subgraph Real["Real world (physical chain, single capture)"]
        S1[Photon distribution]
        S2[Sensor readout<br/>+ shot + read noise]
        S3[demosaicing<br/>+ ISP]
        S4[JPEG quality 70-95]
        S5[Network recompression<br/>quality 50-70]
        S1 --> S2 --> S3 --> S4 --> S5
    end

    subgraph Synth["Synthetic world (second-order degradation, parameterized)"]
        D1[blur 1<br/>isotropic/anisotropic Gaussian]
        D2[resize 1<br/>area / bilinear / bicubic]
        D3[noise 1<br/>Gaussian + Poisson]
        D4[JPEG 1<br/>quality 30-95]
        D5[blur 2<br/>smaller]
        D6[resize 2]
        D7[noise 2]
        D8[JPEG 2 / sinc]
        D1 --> D2 --> D3 --> D4 --> D5 --> D6 --> D7 --> D8
    end

    S2 -. simulates .-> D3
    S3 -. simulates .-> D1
    S3 -. simulates .-> D2
    S4 -. simulates .-> D4
    S5 -. simulates .-> D8

    style Real fill:#e3f2fd
    style Synth fill:#fff8e1
```

Several patterns in this figure are worth remembering. First, the physical chain happens once and has a fixed order; the synthesis chain deliberately shuffles the order and randomizes parameters, because the model needs to see not the degradations of one particular camera, but the degradation distribution of "all possible cameras + all possible transmission paths." Second, some steps of the physical chain (such as ISP denoising and sharpening) leave detail coupling in the LR that the synthesis chain cannot precisely replicate, which is why §5.12 emphasizes "fine-tune with real data" as the last mile. Third, the intermediate product between the two stages of JPEG carries blocking artifacts, and the second-stage JPEG further scrambles those block boundaries—this is the fundamental reason §5.4 will emphasize "second-order."

## 5.4 Real-ESRGAN Degradation Pipeline in Detail

The Real-ESRGAN pipeline consists of two key designs:

### Design 1: parameterize each component + randomize

Each kind of degradation (blur, downsampling, noise, JPEG) is not fixed but **sampled from a distribution**. The model sees a wide degradation distribution during training.

### Design 2: high-order / second-order degradation modeling

Run the entire degradation pipeline twice. This is Real-ESRGAN's core innovation, and it is the meaning of "high-order" that the paper title highlights.

```
First-order degradation (simulating in-camera):
  HR → blur₁ → downsample₁ → noise₁ → jpeg₁ → intermediate

Second-order degradation (simulating transmission/recompression):
  intermediate → blur₂ → downsample₂ → noise₂ → jpeg₂ → LR
```

The synthesized LR thus approaches "an image processed by the camera and then compressed multiple times by the internet."

The key parameters of the second order are **deliberately different** from the first:

- First-order blur tends to be larger (simulating actual camera/transmission blur)
- Second-order blur tends to be smaller (avoid making training data too blurry to learn)
- Second order may add **sinc filtering** (a filter based on the sinc function, a rectangular low-pass in the frequency domain producing oscillations in the spatial domain)—simulating the ringing artifacts caused by over-sharpening, an artifact especially common in LCD displays / certain image-processing software outputs
- The order of **resize / sinc / JPEG** in the final stage of the second order is **randomized** in the official code (a random order is picked each training step), not fixed—this exposes the model to more combinations

Drawing "second-order + order randomization" as a state diagram makes it more intuitive:

```mermaid
graph TD
    HR[HR clean image<br/>x in 0,1]
    HR --> B1[blur 1<br/>sigma 0.2-3.0]
    B1 --> R1[resize 1<br/>scale 0.15-1.5]
    R1 --> N1[noise 1<br/>std 1-30/255]
    N1 --> J1[JPEG 1<br/>quality 30-95]
    J1 --> Mid[Intermediate product]
    Mid --> B2[blur 2<br/>sigma 0.2-1.5]
    B2 --> Choice{Random order}
    Choice -->|case A| A1[resize -> sinc -> JPEG]
    Choice -->|case B| A2[resize -> JPEG -> sinc]
    Choice -->|case C| A3[sinc -> resize -> JPEG]
    A1 --> LR[LR degraded]
    A2 --> LR
    A3 --> LR

    style HR fill:#e8f5e9
    style Mid fill:#fff3e0
    style LR fill:#ffebee
```

Two engineering conclusions are worth emphasizing in this figure. First, every step's parameter is sampled from an explicit range, so "the $D$ distribution the model sees during training" is written down in code rather than vaguely "hoping it has seen everything." Second, the final three steps' order randomization exists so that the model sees both "resize-then-JPEG" and "JPEG-then-resize" outputs, the former closer to a camera's in-device processing, the latter closer to a Weibo reshare chain. If the model has only ever seen one order, it overfits to the artifact patterns left by that order.

### A simplified Real-ESRGAN pipeline

```python
import random
import torch
import torch.nn.functional as F
from typing import Optional


class RealESRGANDegradation:
    """Demonstrative main flow of Real-ESRGAN style degradation (not a complete implementation).
    Input: HR clean image (B, 3, H, W) in [0, 1]
    Output: LR degraded image (B, 3, H/scale, W/scale) in [0, 1]

    Differences from official Real-ESRGAN:
    - JPEG is a placeholder (official uses diffjpeg)
    - No sinc filter (official uses it to simulate over-sharpening artifacts)
    - Blur kernels are isotropic Gaussian only (official also has anisotropic, generalized Gaussian, plateau)
    - Degradation order is fixed (in official code the final-stage sinc/resize/JPEG order is randomized)
    The remainder of this section will fill in these missing pieces.
    """

    def __init__(self, scale: int = 4):
        self.scale = scale

        # First-order degradation parameter ranges
        self.blur1_sigma_range = (0.2, 3.0)
        self.noise1_range = (1, 30)         # std on [0, 255]
        self.jpeg1_range = (30, 95)
        self.resize1_range = (0.15, 1.5)    # relative to final size

        # Second-order degradation parameter ranges (overall weaker)
        self.blur2_sigma_range = (0.2, 1.5)
        self.noise2_range = (1, 25)
        self.jpeg2_range = (30, 95)
        self.resize2_range = (0.3, 1.2)

    # ============= Blur =============
    def random_gaussian_blur(self, x: torch.Tensor,
                             sigma_range: tuple) -> torch.Tensor:
        sigma = random.uniform(*sigma_range)
        ksize = max(3, 2 * int(3 * sigma) + 1)
        kernel = self._gaussian_kernel_2d(ksize, sigma).to(x)
        kernel = kernel.expand(x.shape[1], 1, ksize, ksize)
        pad = ksize // 2
        return F.conv2d(F.pad(x, [pad]*4, mode='reflect'),
                        kernel, groups=x.shape[1])

    @staticmethod
    def _gaussian_kernel_2d(ksize: int, sigma: float) -> torch.Tensor:
        ax = torch.arange(ksize).float() - ksize // 2
        gauss = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
        kernel = (gauss[:, None] * gauss[None, :])
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)

    # ============= Downsample =============
    def random_resize(self, x: torch.Tensor, target_size: tuple,
                      scale_range: tuple) -> torch.Tensor:
        """Randomly choose target size (scaled relative to target_size), random interpolation."""
        s = random.uniform(*scale_range)
        h, w = target_size
        new_h = max(1, int(h * s))
        new_w = max(1, int(w * s))
        mode = random.choice(['bilinear', 'bicubic', 'area'])
        kw = {'mode': mode}
        if mode in ('bilinear', 'bicubic'):
            kw['align_corners'] = False
        return F.interpolate(x, size=(new_h, new_w), **kw)

    # ============= Noise =============
    def random_noise(self, x: torch.Tensor,
                     sigma_range_255: tuple) -> torch.Tensor:
        """Three noise types with equal probability: gray Gaussian, color Gaussian, Poisson."""
        choice = random.random()
        sigma_max = random.uniform(*sigma_range_255) / 255.0

        if choice < 0.4:
            # Color Gaussian
            return (x + torch.randn_like(x) * sigma_max).clamp(0, 1)
        elif choice < 0.7:
            # Gray Gaussian (same noise across all channels)
            n = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
            return (x + n * sigma_max).clamp(0, 1)
        else:
            # Poisson (signal-dependent)
            scale = random.uniform(80, 1000)
            photons = (x * scale).clamp(min=0)
            return (torch.poisson(photons) / scale).clamp(0, 1)

    # ============= JPEG (simplified, real version needs diffjpeg) =============
    def random_jpeg(self, x: torch.Tensor, quality_range: tuple) -> torch.Tensor:
        """Placeholder. Real code:
            from diffjpeg import DiffJPEG
            quality = random.randint(*quality_range)
            return DiffJPEG(differentiable=False)(x, quality)
        """
        # Approximate JPEG block effect with an 8x8 box filter (illustrative, not for training)
        return x

    # ============= First-order degradation =============
    def first_order(self, hr: torch.Tensor, target_size: tuple) -> torch.Tensor:
        x = self.random_gaussian_blur(hr, self.blur1_sigma_range)
        x = self.random_resize(x, target_size, self.resize1_range)
        x = self.random_noise(x, self.noise1_range)
        x = self.random_jpeg(x, self.jpeg1_range)
        return x

    # ============= Second-order degradation =============
    def second_order(self, x: torch.Tensor, target_size: tuple) -> torch.Tensor:
        x = self.random_gaussian_blur(x, self.blur2_sigma_range)
        x = self.random_resize(x, target_size, self.resize2_range)
        x = self.random_noise(x, self.noise2_range)
        x = self.random_jpeg(x, self.jpeg2_range)
        return x

    # ============= Main flow =============
    def __call__(self, hr: torch.Tensor) -> torch.Tensor:
        """Full Real-ESRGAN style degradation.
        hr: (B, 3, H, W) in [0, 1]
        return: lr (B, 3, H/scale, W/scale)
        """
        b, c, h, w = hr.shape
        out_h, out_w = h // self.scale, w // self.scale

        # First order: scale to an intermediate size in [0.15*out, 1.5*out]
        intermediate = self.first_order(hr, target_size=(out_h, out_w))

        # Second order: finally scale to (out_h, out_w); resize to target first
        lr = self.second_order(intermediate, target_size=(out_h, out_w))

        # Force final size (second-order resize may drift)
        lr = F.interpolate(lr, size=(out_h, out_w),
                          mode='bicubic', align_corners=False)
        return lr.clamp(0, 1)
```

In practice, the real version is more complex: the full Real-ESRGAN code is around 700 lines. Additions include:

- Multiple blur kernels (anisotropic, generalized Gaussian, plateau)
- Sinc filtering (simulating over-sharpening artifacts)
- Differentiable JPEG (done directly on GPU)
- Probability allocation for grayscale/color images
- USM sharpening (Unsharp Mask) inverse (simulating phone post-processing)

USM's specific procedure: first Gaussian-blur the image to get $x_b$, then enhance high frequencies via $x_{usm} = x + \lambda (x - x_b)$, where $\lambda$ controls sharpening strength. Phone-vendor ISPs almost all bake in USM-style sharpening, which means a user's photo already carries a mild "over-sharpened" signature. Real-ESRGAN also includes a USM step during synthesis so the model can "recognize" inputs that have been pre-sharpened, avoiding stacking another round of sharpening during SR that would look greasy.

But the skeleton is the above; the extensions only replace "kernel sampling" from a single function with a family of functions, and the degradation order from fixed to randomly chosen. After reading §5.5 through §5.8 on each component, you should be able to fill the skeleton above out to a production-grade implementation.

## 5.5 The Blur Kernel Family

Different degradation scenarios need different blur kernels.

### Isotropic Gaussian (most basic)

$$
k(u, v) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{u^2 + v^2}{2\sigma^2}\right)
$$

A single parameter $\sigma$. Simulates defocus, atmospheric blur. Larger $\sigma$ gives stronger blur, and the blur is the same in every direction.

### Anisotropic Gaussian

$$
k(u, v) \propto \exp\left(-\frac{1}{2}\begin{pmatrix}u\\v\end{pmatrix}^T \Sigma^{-1} \begin{pmatrix}u\\v\end{pmatrix}\right)
$$

The covariance matrix $\Sigma$ controls different blur magnitudes in x/y. Simulates camera shake, lens aberrations. Expanding $\Sigma$:

$$
\Sigma = R(\theta)\begin{pmatrix}\sigma_1^2 & 0 \\ 0 & \sigma_2^2\end{pmatrix} R(\theta)^T
$$

where $R(\theta)$ is a rotation matrix, $\theta$ is the blur direction, and $\sigma_1, \sigma_2$ are the blur scales along the two principal axes. The three parameters jointly determine the kernel shape (one angle + two scales), giving two more degrees of freedom than the isotropic case—enough to model the common "horizontal blur is more pronounced than vertical blur" pattern of residual camera shake.

### Generalized Gaussian

$$
k(u, v) \propto \exp\left(-\left(\frac{u^2}{\sigma_x^2} + \frac{v^2}{\sigma_y^2}\right)^\beta\right)
$$

An extra shape parameter $\beta$: $\beta = 1$ is standard Gaussian, $\beta > 1$ has sharper edges (closer to box), $\beta < 1$ has softer edges.

### Plateau-shaped kernel

Real-ESRGAN also uses a family of **plateau kernels**—a flat "plateau" region in the center with steep edges. They and generalized Gaussian are **two independent families**, not simply "$\beta < 1$ equals plateau." Plateau kernels are closer to the shape of a "defocus disc," common in small-aperture / deep-focus scenes, and cannot be precisely expressed by an ordinary Gaussian. When sampling training kernels, Real-ESRGAN probabilistically mixes these families (isotropic Gaussian, anisotropic Gaussian, generalized Gaussian, plateau). The specific proportions can be checked in the official code's `degradations.py`; the most common ratio is roughly isotropic Gaussian 0.55, anisotropic 0.10, generalized Gaussian 0.12, plateau 0.03, with the remaining probability reserved for sinc.

### Motion blur kernel

Simulates linear motion of the camera or subject during capture. If during exposure time $T$ the sensor moves linearly relative to the scene at velocity $v$, every scene point sweeps out a line of length $vT$ on the image plane along the motion direction. Discretizing this line onto the pixel grid and normalizing it gives the simplest form of a motion blur kernel. In the frequency domain it is a sinc function extended along the motion direction, so motion blur not only blurs edges but selectively suppresses high frequencies along that direction.

```python
import numpy as np
import torch

def motion_blur_kernel(length: int, angle_deg: float) -> torch.Tensor:
    """Line-segment motion blur kernel of length `length` at angle `angle_deg` degrees."""
    kernel = np.zeros((length, length), dtype=np.float32)
    angle = np.deg2rad(angle_deg)
    cx, cy = length // 2, length // 2
    for i in range(length):
        dx = int(round(cx + (i - cx) * np.cos(angle)))
        dy = int(round(cy + (i - cx) * np.sin(angle)))
        if 0 <= dx < length and 0 <= dy < length:
            kernel[dy, dx] = 1.0
    kernel /= kernel.sum()
    return torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)
```

More realistic motion kernels add slight curvature to the line (simulating non-uniform-speed hand shake) or stack several short lines at different angles (simulating camera shake + local subject motion). Motion kernels have a low proportion in the official Real-ESRGAN implementation, mainly used as "low-probability hard samples" sampled into the training distribution, to keep the model from overfitting to any single blur type.

### Sinc filter

$$
k(u, v) = \frac{\omega_c}{2\pi r} J_1(\omega_c r), \quad r = \sqrt{u^2 + v^2}
$$

where $J_1$ is the first-order Bessel function. Sinc is an ideal low-pass filter (rect-shaped) in the frequency domain, but a truncated sinc oscillates in the pixel domain—producing **ringing** artifacts.

Why Real-ESRGAN uses sinc: to simulate the ringing produced by sharpening algorithms in certain image processing software (like Adobe products) when overdone. Such artifacts are very common in real "post-processed" images, especially the "looks sharp but the detail is fake" images downloaded from Weibo / Twitter. If the model has not seen sinc-style ringing in training data, it will treat the ringing as real high frequencies and preserve or even amplify it, making the output look more fake.

To form an intuitive picture of the "blur kernel family," the diagram below compares the spatial-domain profiles of various kernels:

```mermaid
graph LR
    A[Isotropic Gaussian<br/>circular bell] --> Aa[1 param sigma]
    B[Anisotropic Gaussian<br/>elliptical bell] --> Bb[3 params sigma1 sigma2 theta]
    C[Generalized Gaussian<br/>bell with adjustable sharpness] --> Cc[extra shape param beta]
    D[Plateau<br/>flat center steep edges] --> Dd[simulates defocus disc]
    E[Motion kernel<br/>line segment] --> Ee[length and direction]
    F[Sinc<br/>oscillating rings] --> Ff[simulates over-sharpening ringing]

    style A fill:#e3f2fd
    style B fill:#e3f2fd
    style C fill:#fff8e1
    style D fill:#fff8e1
    style E fill:#ffebee
    style F fill:#ffebee
```

Color groups reflect "difficulty." The blue group on the left is the Gaussian family—few parameters, cheap to sample. The middle two need extra geometric parameters but can still be batch-generated on GPU. The right two (motion, sinc) have special shapes and are the hardest for the model, requiring dedicated sampling and not approximated by a single Gaussian.

## 5.6 Multiple Downsampling Strategies

Different interpolation kernels behave very differently in the frequency domain:

| Algorithm | Frequency-domain behavior | Visual character | When to use |
|------|---------|---------|-------|
| nearest | Rectangle (severe high-frequency leakage) | Aliasing | Simulate very poor processing |
| bilinear | Triangle | Slightly blurry | Balanced choice |
| bicubic | Close to sinc | Sharp but with ringing | Academic standard |
| lanczos | Truncated sinc | Sharpest, with visible ringing | Simulate certain software outputs |
| area / box | Rectangle (spatial) | Soft, no aliasing | Large downscaling |

Real-ESRGAN's pipeline **randomly picks** one of these on each resize, exposing the model to a variety of interpolation artifacts.

Why this matters can be understood from the frequency domain: bicubic is nearly an ideal low-pass, cleanly chopping off high frequencies; nearest does no low-pass at all and leaves heavy aliasing; area averages each LR pixel as a box filter over the HR patch, with a sinc frequency response that corresponds to a rectangle in the pixel domain—able to remove small-motion blur but pass through large edges. If the model has only ever seen "clean low-frequency images" from bicubic, it will treat residual aliasing from real images (which were typically processed by area or bilinear) as high-frequency signal and "amplify" it—the result is making the jaggies sharper rather than restoring detail.

A small but useful engineering detail: with `align_corners=False`, `torch.nn.functional.interpolate` matches OpenCV's `cv2.resize` defaults; with `align_corners=True`, it matches some old TensorFlow alignments, with the two differing visibly at the 1-2 pixel edge. Make sure the resize library used at inference time matches training; otherwise you get hard-to-spot bugs like "trained with bicubic, but deployment calls ImageMagick's catmull-rom implementation, output size off by one row."

## 5.7 Implementation Details of Noise

Chapter 1 already covered the physical model of noise. Here are a few engineering details.

### Grayscale noise vs color noise

The color correlation of real noise is a subtle issue, to be understood from both the optics and the processing-chain perspectives:

- High-end cameras (full-frame DSLR, medium format): sensor pixels are large, SNR is high, single-frame noise is approximately i.i.d. (per channel independent), with essentially zero three-channel correlation
- Phone cameras: pixels are small, ISOs are often high; after demosaicing (interpolating missing colors from neighbors), **neighbor-pixel noise becomes correlated**; the color matrix also mixes noise across channels, leading to inter-channel correlation
- After ISP denoising: built-in denoising first removes "independent color noise," and the residual noise is often "gray" (correlated across all three channels, looking like a thin layer of gray haze)

Engineering practice: training data **provides both**—50% of the time use three-channel-independent color noise (simulating low-ISP-processing or RAW), 50% of the time use gray noise (a single noise map shared across channels, simulating post-ISP residue). A more refined implementation mixes in "neighbor-correlated" noise at a 9:1 ratio (lightly Gaussian-blur a random noise map before adding it), modeling the actual post-demosaicing state of phones.

### Noise intensity distribution

Don't sample noise sigma uniformly—in real scenes **small noise is more common, large noise less common**.

Common engineering choice:

```python
# Distribution biased toward small noise
sigma = torch.rand(()) ** 2 * sigma_max  # squaring biases toward small values
```

Or sample from an explicit mixture:

```python
if random.random() < 0.5:
    sigma = random.uniform(0.005, 0.02)   # small noise (common)
else:
    sigma = random.uniform(0.02, 0.08)    # large noise (uncommon but should be seen)
```

### Realistic sensor noise model

For more accurate simulation, you can use the physics-based model from "A Physics-based Noise Formation Model for Extreme Low-light Raw Denoising" (CVPR 2020).

A simplified version:

```python
def realistic_sensor_noise(
    x: torch.Tensor,
    iso: int = 1600,
    quantum_efficiency: float = 0.5,
    read_noise_sigma: float = 0.005,
    dark_current: float = 0.001,
    quant_step: float = 1/255.0,
) -> torch.Tensor:
    """Simulate the real sensor noise chain.
    Higher iso, stronger noise across all types.
    """
    # Simulate effect of "camera gain" on noise
    gain = iso / 100.0

    # 1. Photon noise (Poisson, signal-dependent)
    photons = x * 1000 / gain  # calibrate photon count
    photons_noisy = torch.poisson(photons.clamp(min=0))
    shot = photons_noisy / 1000 * gain

    # 2. Dark current noise
    dark = torch.poisson(torch.full_like(x, dark_current * gain)) / 1000

    # 3. Read noise (Gaussian, signal-independent)
    read = torch.randn_like(x) * read_noise_sigma * gain

    # 4. Quantization error
    out = shot + dark + read
    out = (out / quant_step).round() * quant_step

    return out.clamp(0, 1)
```

This model is critical for training-data synthesis in **phone low-light denoising**.

## 5.8 JPEG Compression

JPEG is the step most easily overlooked and most critical in the synthesis pipeline. If training data lacks JPEG, the model fails on any image that has gone through network transmission (i.e. 99% of real inputs), because JPEG blocking artifacts are entirely outside its input distribution.

The JPEG encode/decode flow, summarized briefly to help understand why the "DCT quantization step" is the non-differentiable core:

1. RGB → YCbCr, separating luminance Y from chrominance Cb/Cr
2. Apply chroma subsampling to Cb/Cr (4:2:0 most common, each chroma dimension halved)
3. Apply 8×8 block DCT, yielding 64 frequency coefficients per block
4. Divide and round each coefficient using a quality-dependent quantization table
5. Zig-zag scan + entropy coding to disk

The "round" in step 4 maps continuous values to integers, with **no gradient**. This means propagating a gradient through JPEG in end-to-end training requires a softened replacement of rounding (e.g. straight-through estimator or first-order Taylor expansion), which is exactly what the DiffJPEG library does.

JPEG in the synthesis pipeline must satisfy:

1. **Can be done on GPU** (don't write each image to disk and read back)—disk I/O bottlenecks will stall the entire dataloader
2. **Ideally differentiable** (although gradients are not usually propagated through JPEG in enhancement training, the differentiable version still performs well, and is required if you ever want to compute perceptual loss gradients on post-JPEG images)

Recommended: the **DiffJPEG** library ([github.com/mlomnitz/DiffJPEG](https://github.com/mlomnitz/DiffJPEG)), which implements differentiable JPEG encoding/decoding:

```python
from DiffJPEG import DiffJPEG

# Non-differentiable (for inference)
jpeg_module = DiffJPEG(differentiable=False, quality=80).to(device)

# Differentiable (when you want to backpropagate during training)
jpeg_module = DiffJPEG(differentiable=True, quality=80).to(device)

def random_jpeg_diffjpeg(x: torch.Tensor, quality_range=(30, 95)) -> torch.Tensor:
    quality = random.randint(*quality_range)
    return DiffJPEG(differentiable=False, quality=quality).to(x.device)(x)
```

### Specifics of second-order JPEG

Real-ESRGAN's second-order JPEG simulates "an already compressed image being compressed again." In this case:

- The first-order JPEG's blocking artifacts are further scrambled by the second-order JPEG
- The two JPEGs' 8×8 block boundaries are usually not aligned because a resize sits between them, causing complex non-aligned composite artifacts
- This phenomenon is widespread in real "network-retransmitted" images—a Weibo image you saved has already had its block boundaries "drift" once after a reshare

Calling JPEG twice directly simulates this. A notable detail: the second JPEG's quality does not need to be lower than the first's; the reverse is common (phone → WeChat quality 70 → you save at quality 90). The point is that **the two quantization grids do not align**, not the quality numbers themselves.

### A note on chroma subsampling

JPEG's 4:2:0 chroma subsampling halves the resolution of the Cb/Cr channels. It is almost invisible to the human eye, but models can pick up unreal relations like "green edges are sharper than red edges" from it. If your goal is to repair "network-mangled images," your training JPEG simulation must preserve chroma subsampling; if your goal is repairing "high-fidelity professional captures with mild JPEG traces," you may turn it off. DiffJPEG defaults to 4:2:0.

## 5.9 Dataset Selection

Common datasets, listed by use:

### General SR / denoising / deblurring

| Dataset | Count | Characteristics | Use |
|-------|------|------|------|
| **DIV2K** | 800 + 100 val | High-quality natural images, 2K resolution | Academic standard |
| **Flickr2K** | 2650 | DIV2K-style supplement | Combined with DIV2K to form DF2K |
| **LSDIR** | 84,991 + 250 val | 2023 large-scale dataset | First choice for modern training |
| **BSD500** | 500 | Old but classic | Denoising |
| **OST300** | 300 | Outdoor texture specialized | Real SR auxiliary |
| **WED** | ~5000 | Diverse watermark/exif | Real-scene |

### Real degradation (no synthesis)

| Dataset | Count | Characteristics |
|-------|------|------|
| **RealSR** | 595 (V3) | Canon + Nikon different focal-length pairs |
| **DRealSR** | ~800 | DSLR different focal lengths, more precise alignment |
| **NTIRE Real-World SR** | grows yearly | Competition data, high quality |
| **DPED** | ~16K | Phone vs DSLR |
| **SIDD** | 320 (HR) + 30K (LR) | Real phone noise |
| **DND** | 50 test | Darmstadt Noise Dataset, real noise testing |

### Task-specific

| Dataset | Use |
|-------|------|
| **FFHQ** | 70K high-quality faces, face enhancement |
| **CelebA-HQ** | 30K faces, an older standard |
| **REDS** | Video deblurring/SR |
| **Vimeo-90K** | Video frame interpolation |
| **GoPro** | Video deblurring |
| **UDM10** | Video denoising |

### Dataset combination strategies

The two most common:

1. **Academic standard**: train on DF2K (DIV2K + Flickr2K), test on Set5/14/B100/Urban100/Manga109/DIV2K val
2. **Realistic-oriented**: train on LSDIR + DF2K + OST300, fine-tune with a bit of RealSR, test on DRealSR/RealSR

The academic standard has the benefit of comparability, with the downside that the benchmarks are small (Set5 really is only 5 images), limiting the statistical significance of the conclusions. The realistic-oriented choice better reflects the production distribution but lacks a standardized test set for direct comparison with other papers—you can only compare on your internal real holdout. Running both in production is common: academic benchmarks confirm "we didn't break the basics," real test sets confirm "we deserve this deployment."

## 5.10 Data Augmentation

Data augmentation in low-level vision is very different from classification. **Safe** ones:

```python
# Safe augmentation (does not change the degradation model)
import torchvision.transforms.functional as TF

def safe_augment(hr: torch.Tensor, lr: torch.Tensor):
    """Synchronously apply safe augmentation to the HR-LR pair."""
    # Random horizontal flip
    if random.random() < 0.5:
        hr = TF.hflip(hr); lr = TF.hflip(lr)
    # Random vertical flip
    if random.random() < 0.5:
        hr = TF.vflip(hr); lr = TF.vflip(lr)
    # 90° rotation
    if random.random() < 0.5:
        k = random.choice([1, 2, 3])
        hr = torch.rot90(hr, k, dims=[-2, -1])
        lr = torch.rot90(lr, k, dims=[-2, -1])
    return hr, lr
```

**Use with caution**:

- **Color jitter**: changes the degradation distribution, the model learns wrong color mappings; for real-degradation models, color shift itself is part of $D$ and shouldn't be treated as an "axis you can freely change"
- **Arbitrary-angle rotation**: interpolation introduces extra degradation (effectively another round of bicubic/bilinear resampling), not in the original $D$; the model will learn that "slight interpolation artifacts in the output are normal"
- **Random scaling**: equivalent to changing the scale factor; don't use it in SR—the input-to-output size relation is part of the task definition
- **mixup / cutmix**: rarely used in low-level vision, unclear effect; their design premise is "the task is invariant to category," but pixel-regression tasks have ground truth at every pixel and mixing pixels lacks a clear semantic interpretation

**Cropping strategy**:

```python
def random_crop(hr: torch.Tensor, lr: torch.Tensor,
                lr_size: int, scale: int):
    """Random crop on LR + corresponding HR region."""
    _, _, lr_h, lr_w = lr.shape
    top  = random.randint(0, lr_h - lr_size)
    left = random.randint(0, lr_w - lr_size)
    lr_crop = lr[:, :, top:top+lr_size, left:left+lr_size]
    hr_crop = hr[:, :, top*scale:(top+lr_size)*scale,
                       left*scale:(left+lr_size)*scale]
    return hr_crop, lr_crop
```

## 5.11 Data Loading Pipeline Optimization

Degradation synthesis has non-trivial compute (especially blur + JPEG). If done on CPU it becomes a training bottleneck. Two optimizations:

### Option 1: CPU multi-worker

```python
from torch.utils.data import DataLoader, Dataset

class SRDataset(Dataset):
    def __init__(self, hr_paths, scale=4, crop_size=256):
        self.hr_paths = hr_paths
        self.degrader = RealESRGANDegradation(scale=scale)
        self.crop_size = crop_size

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr = load_image(self.hr_paths[idx])  # PIL or cv2
        hr = random_crop_hr(hr, self.crop_size)
        hr_tensor = to_tensor(hr).unsqueeze(0)  # (1, 3, H, W)
        lr_tensor = self.degrader(hr_tensor)
        return hr_tensor.squeeze(0), lr_tensor.squeeze(0)


loader = DataLoader(
    SRDataset(hr_paths),
    batch_size=16,
    num_workers=8,           # 8 CPU workers
    pin_memory=True,
    persistent_workers=True,
)
```

### Option 2: do degradation on GPU (recommended by Real-ESRGAN)

Move the degradation pipeline to GPU and process the whole batch at once. This option seems counter-intuitive: usually we put data preprocessing on CPU so the GPU can focus on matrix multiplications. But blur, JPEG, and resize inside degradation synthesis are themselves heavy convolutions and frequency-domain operations, and running them on GPU is 20× or more faster than CPU; keeping them on CPU instead drags training down.

```python
# Inside the training loop
for hr_batch in loader:                  # CPU only reads HR
    hr_batch = hr_batch.cuda()
    lr_batch = degrader(hr_batch)        # synthesize all at once on GPU
    pred = model(lr_batch)
    loss = compute_loss(pred, hr_batch)
    ...
```

The official Real-ESRGAN code uses this pattern. Pros:

- Greatly reduces CPU work
- Can use more complex degradations (multi-GPU parallel synthesis)
- Degradation results can leverage GPU randomness (different on every step)

Cons:

- Uses GPU memory (some intermediate tensors)
- Degradation code must support batched + GPU

To keep both "CPU reads" and "GPU degradation" sides from idling, the full data flow should look like:

```mermaid
graph LR
    Disk[HR PNG / LMDB] --> CPU[CPU workers<br/>read + crop + to_tensor]
    CPU --> Queue[pinned memory queue]
    Queue --> GPU1[GPU: degradation<br/>blur/resize/noise/JPEG]
    GPU1 --> GPU2[GPU: model forward]
    GPU2 --> GPU3[GPU: loss + backward]
    GPU3 --> GPU4[GPU: optimizer step]
    GPU4 -.next batch.-> Queue

    style Disk fill:#e3f2fd
    style CPU fill:#fff8e1
    style GPU1 fill:#ffebee
    style GPU2 fill:#ffebee
    style GPU3 fill:#ffebee
    style GPU4 fill:#ffebee
```

Two often-overlooked engineering tips in this figure. First, the pinned-memory queue is the fixed-memory buffer enabled by PyTorch's `pin_memory=True`, which lets CPU-to-GPU copy use asynchronous DMA without blocking the main process. Second, the degradation step must be written as "batched + fully GPU-executable"; any code that accidentally becomes a Python for-loop or includes `.cpu()` drops training throughput by an order of magnitude.

### LMDB storage to accelerate large datasets

LSDIR-scale datasets (80K+ images) make decoding PNGs each epoch a significant cost. Use LMDB to pre-encode the data into a key-value database, with 5–10× faster loading:

```python
import lmdb
import pickle

# Writing
env = lmdb.open('lsdir.lmdb', map_size=int(1e12))
with env.begin(write=True) as txn:
    for i, path in enumerate(hr_paths):
        img_bytes = open(path, 'rb').read()  # or pre-decoded numpy
        txn.put(f"{i:08d}".encode(), pickle.dumps(img_bytes))

# Reading
class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, length):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        self.length = length

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            data = pickle.loads(txn.get(f"{idx:08d}".encode()))
        # decode bytes to image
        ...
```

## 5.12 Real Data Fine-tuning

General best practice:

```
Stage 1: pretrain (synthetic data)
  - Data: DF2K + LSDIR
  - Degradation: Real-ESRGAN pipeline
  - Steps: 500K - 1M
  - Goal: learn general degradation-recovery capability

Stage 2: fine-tune (real data)
  - Data: RealSR / DRealSR (a few hundred images)
  - Steps: 10K - 50K (much fewer than pretrain)
  - Learning rate: 10× smaller than pretrain
  - Goal: calibrate the general model to the real degradation distribution
```

The reason for not training from scratch on real data: too little data (a few hundred images)—the model overfits to the specific objects and scenes in those few hundred images. This is a classic catastrophic-specialization trap: PSNR rises rapidly on the training-set objects, but as soon as you switch to a different scene type, the model collapses. The two-stage training method essentially uses synthetic data to lay down the model's "prior" first, leaving real data only the small task of distribution alignment rather than learning "what natural images should look like" from scratch.

A few more engineering details for fine-tuning. The learning rate should not only be one order of magnitude smaller than pretraining; ideally it follows a cosine schedule that decays nearly to zero over the final few thousand steps, to avoid the model oscillating on the small dataset. EMA (Exponential Moving Average, a shadow copy of weights that follows an exponential moving average) is especially important during fine-tuning—it further smooths training jitter. Finally, the validation set should be from **a different source** of real images than the fine-tuning set (e.g. fine-tune on RealSR, validate on DRealSR); otherwise the gains you see may be entirely overfit.

## 5.13 Data Quality Audit

Before training, **audit your training data itself**:

### Is the HR data really "high quality"?

DIV2K's HR looks high quality, but still has:

- Mild JPEG compression traces
- Occasional noise patterns
- Local over/under-exposure

If HR itself has compression artifacts, the model learns "outputting mild compression artifacts is normal"—your "HR ground truth" isn't truly ground truth.

Engineering check:

```python
# Audit JPEG compression traces in HR dataset
def audit_jpeg_traces(hr_path: str) -> float:
    """Estimate how "pre-compressed" the HR data is by analyzing discontinuity at 8x8 block boundaries.
    Higher value = more obvious JPEG compression traces.
    """
    img = cv2.imread(hr_path)
    img = img.astype(np.float32)
    # Differences at block boundaries (rows/cols at multiples of 8)
    h, w = img.shape[:2]
    block_diff = 0
    for i in range(7, h-1, 8):
        block_diff += np.abs(img[i] - img[i+1]).mean()
    for j in range(7, w-1, 8):
        block_diff += np.abs(img[:, j] - img[:, j+1]).mean()
    return block_diff
```

### Data diversity audit

Models failing on a certain content type is often a data bias:

- If 90% of training data is urban/natural scenery, the model fails on documents/screenshots
- If all training data is daylight, the model fails on night scenes
- If all training data is Western faces, the model fails on Asian/African faces

This kind of bias won't show in metrics—average PSNR looks fine, but failure modes concentrate. Chapter 17 covers this specifically.

The simplest audit:

```python
# Use CLIP to embed each training image, then cluster to inspect distribution
import open_clip

clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32')
embeddings = []
for path in hr_paths:
    img = preprocess(Image.open(path)).unsqueeze(0)
    with torch.no_grad():
        emb = clip_model.encode_image(img).cpu().numpy()
    embeddings.append(emb)

# K-means into 20 clusters, examine what each cluster contains and its proportion
from sklearn.cluster import KMeans
clusters = KMeans(n_clusters=20).fit_predict(np.vstack(embeddings))
```

If a cluster is extremely small (< 1%), consider supplementing data for that part.

### Hidden hazards of imbalanced degradation distributions

Another hidden issue in synthetic data is the distribution of the degradation parameters themselves. Suppose your blur sigma range is [0.2, 3.0]—that looks reasonable, but if you sample uniformly, the [2.5, 3.0] range gets as many samples as [0.2, 0.7], whereas real-world blur is overwhelmingly concentrated in the small-sigma end. This makes the model over-react to "mild blur" inputs (treating clear small details as objects to deblur) and, conversely, perform reasonably well on "severe blur."

A common engineering fix is to swap uniform sampling for log-uniform or beta-distribution sampling:

```python
import math
import random

def log_uniform(low: float, high: float) -> float:
    """Log-uniform distribution: small values get sampled more often."""
    return math.exp(random.uniform(math.log(low), math.log(high)))

# Usage
sigma = log_uniform(0.2, 3.0)
```

The same fix applies to noise sigma, JPEG quality, and every other "degradation strength" parameter. When you observe model anomalies in production, the first thing to check is whether the training-time parameter distribution matches the real distribution you encounter at deploy time.

## 5.14 Summary

1. **Data > network**: in low-level vision, the same network can differ by an order of magnitude on different data
2. **Bicubic degradation was the field's 5-year methodological mistake**, systematically corrected by Real-ESRGAN
3. **Three cores of degradation pipeline**: parameterization + randomization + second order
4. **Each component is a family**: multiple blur kernels, multiple noise types, multiple JPEG qualities, multiple resize algorithms
5. **Dataset choice determines what world the model can see**: DF2K + LSDIR is the general synthetic-data baseline, RealSR/DRealSR for real-degradation fine-tuning
6. **Data loading pipeline optimization**: GPU-side degradation is the de facto Real-ESRGAN-style standard
7. **Restrained data augmentation**: low-level vision uses only flips + 90° rotation; color jitter and similar break the degradation model
8. **Audit your HR data**: HR's own JPEG traces and data biases will be learned by the model
9. **Real data fine-tuning** is the key step in calibrating a synthetically-trained model to the real distribution

This concludes Part I's five chapters. After reading them, you should be able to:

- Look at a bad image and know the D it has gone through
- Look at an enhancement model and know in which space it predicts and what losses it uses
- Look at a set of metrics and know which are trustworthy and which are biased
- Look at a data pipeline and know which part of the real distribution it covers

Part II starts the specific network architectures. The first stop is the CNN era—a line that runs from SRCNN in 2014 to NAFNet in 2022, the field's "traditional heavy artillery."

---

> Next chapter [The CNN Era](06-cnn.md) → from SRCNN's three-layer network to NAFNet removing all nonlinear activations, eight years of CNN's journey through low-level vision.
