# Chapter 10 · Task-Specific Models

> General-purpose enhancement models (Real-ESRGAN, SUPIR) handle most scenarios. But there are several types of tasks where **general-purpose models do not work well** and a task-specific model is required.
>
> This chapter discusses what inductive biases face, document, and medical imaging each need, and how the corresponding models are designed.

## 10.1 Why we need task-specific models

General-purpose SR performs well on natural images, but breaks down in the following scenarios:

- **Faces**: faces restored by general-purpose models often **change appearance** — eyes are the wrong size, the nose shape is altered, skin tone drifts
- **Text / documents**: a general-purpose model restores "日" as "目" — the structure is wrong by one stroke
- **Medical imaging**: a general-purpose model adds nonexistent "lesions" to X-ray/MRI — potentially an accident-grade error
- **Satellite remote sensing**: a general-purpose model treats multispectral channels as RGB, getting the spectral information completely wrong
- **Scientific microscopy**: a general-purpose model does not understand the physical imaging process and reconstructs non-physical structures

The common reason for general-purpose failure: **the prior they have learned is "the general distribution of natural images", not "the special distribution of this class of images"**.

The core of task specialization = injecting knowledge of this special distribution into the model:

- Face: identity preservation + facial geometric constraints
- Document: character-level correctness + line/curve structure
- Medical: physical imaging model + no "creating" allowed
- Remote sensing: multi-channel spectral physics + large-scale ground feature structure
- Microscopy: imaging theory (PSF, diffraction) + physical reconstruction

This chapter is organized by task, with focus on **face** (the most mature, with the richest engineering practice), and an overview of the others.

## 10.2 Face enhancement: the sub-field with the strongest priors and most models

Face enhancement is the **most deeply researched** sub-field of image enhancement, for three reasons:

1. **Abundant data**: FFHQ (70K), CelebA-HQ (30K), VFHQ are all high-quality large-scale datasets
2. **High application value**: old-photo restoration, video conferencing enhancement, video-call beautification — a huge market
3. **Low cost of failure (and high)**: low cost: getting it slightly wrong does not matter for entertainment apps; high cost: identity must not change (a small error means it is a different person)

There are three main types of inductive bias in face enhancement:

### Bias 1: faces have strong priors (learned by StyleGAN)

StyleGAN (2019) and StyleGAN2/3 have already learned the "distribution of faces" very well. Any high-quality face can be **inverted** (GAN inversion) into a W vector in StyleGAN's latent space.

Key insight:

> The face latent space is low-dimensional (the dimension of StyleGAN's W+ depends on the generation resolution: StyleGAN2 for 1024px FFHQ is 18×512 = 9216 dims; 512px is 14×512).
> A high-quality face = a point in this low-dimensional space.
>
> Enhancement = infer the most plausible W vector from the LR, then decode it back to HR with StyleGAN.

This is the core idea of models like **PULSE / GFPGAN / GPEN**.

### Bias 2: identity must be preserved

Face recognition models (ArcFace, FaceNet) have already learned "what determines a face's identity". The enhancement model must **leave the embedding seen by ArcFace unchanged**.

This is the identity preservation loss covered in Section 3.8 of Chapter 3:

```python
class IdentityLoss(nn.Module):
    """ArcFace identity preservation loss."""
    def __init__(self, arcface_path: str):
        super().__init__()
        self.arcface = load_arcface(arcface_path).eval()
        for p in self.arcface.parameters():
            p.requires_grad_(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # Inputs are aligned faces in the range [-1, 1], typically 112x112
        emb_pred   = self.arcface(pred)
        emb_target = self.arcface(target)
        return 1.0 - F.cosine_similarity(emb_pred, emb_target).mean()
```

### Bias 3: alignment matters

Faces have a standard geometric structure — eyes are horizontal, nose in the middle, mouth below. **Aligned faces** (face alignment) let the model achieve the same effect with a simpler network, because the model does not need to learn "where the facial features might be".

Engineering practice: face enhancement models almost always use a face detector + landmark detector to align the face into a fixed crop (typically 512×512), then paste it back into the original image after enhancement.

## 10.3 GFPGAN: StyleGAN prior + general backbone

Wang et al.'s GFPGAN (Generative Facial Prior GAN) of 2021 is a classic representative of face enhancement.

### Core architecture

```
LR Face (degraded)
  ↓ Encoder (similar to ESRGAN backbone)
  ↓ Extract multi-scale features F_1, F_2, ..., F_n
  ↓
  ↓ Modulate the StyleGAN2 generator in some way
  ↓
StyleGAN2 Generator (frozen, FFHQ pretrained)
  ↓
HR Face
```

Key points:

- **The StyleGAN2 generator is frozen and not trained** — it has already learned the face distribution
- **Training trains the encoder + some modulation layers** — they map LR information into the StyleGAN latent space and intermediate features

### CS-SFT (Channel-Split Spatial Feature Transform)

GFPGAN injects LR features at every layer of the StyleGAN generator, but not by direct concat — it uses an SFT (spatial feature transform):

$$
F' = F \odot (1 + \gamma) + \beta
$$

where $\gamma, \beta$ are spatial features predicted from the LR feature $F_{\text{LR}}$. This modulation lets LR information finely affect each spatial location without disrupting the StyleGAN's overall generation capability.

The "CS" (Channel-Split) of CS-SFT: splitting the StyleGAN feature in half along the channel dimension, modulating one half with SFT (accepting LR information) and keeping the other half as the original StyleGAN output (preserving the generative prior). This balances fidelity and quality.

### GFPGAN training loss

The classic combination:

$$
\mathcal{L} = \lambda_1 \mathcal{L}_1 + \lambda_2 \mathcal{L}_{\text{percep}} + \lambda_3 \mathcal{L}_{\text{adv}} + \lambda_4 \mathcal{L}_{\text{id}} + \lambda_5 \mathcal{L}_{\text{component}}
$$

where $\mathcal{L}_{\text{component}}$ is a "facial component local GAN loss" — separate discriminators are trained on eyes, nose, and mouth, forcing each local region to be realistic.

### GFPGAN performance and limitations

Effect: stunning on heavily degraded old photos, far surpassing general-purpose SR.

Limitations:

- **Depends on face alignment**: cannot be used unaligned
- **Depends on the FFHQ distribution**: quality drops on faces uncommon in FFHQ (e.g. elderly people, children, certain ethnicities)
- **Only handles faces**: must be paired with a general-purpose SR for the background

The engineering combination: **general-purpose SR (Real-ESRGAN) for background + GFPGAN for faces**. This was the standard pipeline for most old-photo restoration tools from 2022 to 2024.

## 10.4 CodeFormer: the advantage of discrete codebooks

Zhou et al.'s CodeFormer of 2022 takes a different approach — instead of relying on StyleGAN, it uses a **discrete codebook learned by VQ-VAE**.

### Core idea

Discretize the "local features" of a face into several codes in a codebook (e.g. 1024 codes). A high-quality face = some combination of these codes.

Pipeline:

```
LR Face
  ↓ Encoder
  ↓ Transformer (predicts the code sequence)
  ↓ Look up the codebook to obtain the corresponding features
  ↓ Decoder
HR Face
```

Why is discretization useful?

- **Discretization = a strong prior**: the model can only generate features that have appeared in the codebook, so it does not fabricate meaningless local content
- **Transformers are naturally suited for predicting discrete sequences**: it is the same paradigm as next-token prediction in LLMs
- **The "control strength" is tunable**: CodeFormer provides $w \in [0, 1]$ that lets users tune between "strictly respecting LR" and "fully exploiting the prior"

### Fidelity tuning of CodeFormer

An engineering highlight of CodeFormer is the `fidelity_weight` parameter. At inference:

```python
def codeformer_inference(model, lr_face, fidelity_weight=0.5):
    """
    fidelity_weight ∈ [0, 1]:
      0 -> fully use the codebook prior (high quality but may change appearance)
      1 -> strictly use LR information (low quality but faithful)
    """
    return model(lr_face, w=fidelity_weight)
```

Common engineering choices:

- Old-photo restoration (looking good is enough): $w = 0.3$
- Video-call beautification (must not change appearance): $w = 0.7$
- Legal evidence (absolute fidelity): do not use CodeFormer; use a discriminative model

### CodeFormer vs GFPGAN

| Dimension | GFPGAN | CodeFormer |
|------|--------|-----------|
| Source of prior | StyleGAN2 (continuous) | VQ codebook (discrete) |
| Inference speed | Fast | Slow (multi-step transformer) |
| Fidelity tunable | No (fixed) | Yes, tunable $w$ |
| For extreme degradation | Easily "changes face" | Discretization restricts changes |
| Engineering friendliness | Medium | **High** |

Since 2024 CodeFormer has become the more mainstream choice for face restoration, with engineering flexibility being the key reason.

## 10.5 RestoreFormer / RestoreFormer++

Wang et al.'s RestoreFormer (2022) directly connects LR features and the codebook with **cross-attention**, removing CodeFormer's multi-step prediction and being much faster.

RestoreFormer++ further optimizes it; it offers the best speed/quality balance for face restoration in 2023.

Code skeleton:

```python
class RestoreFormerBlock(nn.Module):
    """A simplified RestoreFormer block."""

    def __init__(self, dim: int, codebook_size: int, num_heads: int):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, lr_features: torch.Tensor) -> torch.Tensor:
        """
        lr_features: (B, N, D) - feature sequence encoded from LR
        cross-attn to the entire codebook
        """
        codes = self.codebook.weight.unsqueeze(0).expand(
            lr_features.shape[0], -1, -1
        )  # (B, K, D)

        out, _ = self.attn(query=lr_features, key=codes, value=codes)
        return self.norm(out + lr_features)
```

## 10.6 A complete old-photo restoration pipeline

Combining the above face techniques to form a production-grade pipeline:

```python
def restore_old_photo(image_path: str, fidelity: float = 0.5):
    """Complete old-photo restoration pipeline."""

    img = load_image(image_path)

    # 1. General-purpose enhancement (background)
    bg_enhanced = real_esrgan.enhance(img, scale=4)

    # 2. Detect + align faces
    faces, landmarks = face_detector.detect(img)

    enhanced_faces = []
    for face_crop in extract_aligned_faces(img, landmarks):
        # 3. Face-specific restoration
        enhanced = codeformer.enhance(face_crop, fidelity_weight=fidelity)
        enhanced_faces.append(enhanced)

    # 4. Paste enhanced faces back into the background
    final = paste_faces_back(bg_enhanced, enhanced_faces, landmarks)

    return final
```

This is the simplified logic of commercial products like Topaz Photo AI, Tencent ARC, and Adobe.

## 10.7 Document and text enhancement

Document enhancement (preprocessing before OCR) has completely different inductive biases.

### The peculiarities of text

- **Discrete structure**: each character is from a finite set (around 5000 Chinese characters + Western characters)
- **Shapes are bilevel**: black strokes + white background
- **Topology preservation is the core**: restoring "日" as "目" is a structural error (an extra stroke), not a detail error

### Failure of general-purpose SR

What general-purpose SR has learned is "the statistical prior of natural images" — smoothness, texture, natural color. These are completely wrong for text:

- General-purpose SR **blurs strokes** → text becomes illegible
- General-purpose SR tries to add "natural texture" → fog appears at character edges
- General-purpose SR does not know stroke structure → errors of one extra/missing stroke

### Solution 1: DocSR / Text-SR

SR models trained specifically with text data. Key points:

- **Training data**: synthetic text images (with known characters) + real scanned-document pairs
- **Loss weighting**: increase the L1 weight in text regions to emphasize pixel-level correctness
- **Failure-mode-targeted training**: deliberately add "degradations that make text wrong" (strong JPEG, low resolution) in the synthetic data

### Solution 2: OCR guidance

Use an OCR model as an additional loss:

```python
class OCRGuidedLoss(nn.Module):
    """OCR-guided loss.
    The output goes through OCR, and locations with character recognition errors get
    increased pixel-loss weight.
    """
    def __init__(self, ocr_model):
        super().__init__()
        self.ocr = ocr_model.eval()

    def forward(self, pred, target):
        # Standard pixel loss
        pixel_loss = F.l1_loss(pred, target, reduction='none')

        # OCR results on pred
        with torch.no_grad():
            chars_pred   = self.ocr(pred)
            chars_target = self.ocr(target)

        # Add 5× weight at locations where the characters mismatch
        char_mask = (chars_pred != chars_target).float()
        char_mask = expand_to_pixel_mask(char_mask)  # (B, 1, H, W)
        weighted = pixel_loss * (1 + 5 * char_mask)

        return weighted.mean()
```

### Solution 3: Bilevel enhancement

Documents are mostly bilevel (black text on white background). One can learn a two-step pipeline of **binarize first, then SR**:

```
LR document
  ↓ Binarization (Otsu / U-Net)
  ↓ Bilevel SR (a specifically trained SR)
  ↓ Post-processing (anti-aliasing)
HR document
```

Engineering practice: commercial document enhancement products (ABBYY, Adobe Scan) use the above pipeline. In the open-source world, the DocSR family and the doc enhance of PaddleOCR follow similar lines.

## 10.8 Medical image enhancement

Medical imaging (X-ray, CT, MRI, ultrasound) has the strictest constraints.

### Core constraint: **no unconstrained "fabrication"**

In medical diagnosis, "adding a nonexistent lesion" is an accident. Therefore:

- **Unconstrained generative restoration is not allowed** — pure GAN/diffusion priors will fabricate
- **Generative models can be used as a prior**, but **must be paired with strict data-consistency constraints** (at every step the estimate is forced to align with the actual measurement in the observed physical space) — the academic literature contains many works (e.g. score-based MRI reconstruction, diffusion + data consistency) of this kind
- **Discriminative models also need care** — even those trained with L1 may "smooth out" lesions
- **Physical constraints are required**: the imaging physics (the Radon transform of CT, k-space sampling of MRI) must be modeled

### Form of the physical constraint

Medical enhancement is usually a precise model of the inverse problem:

$$
y = A x + n
$$

where $A$ is a **known physical operator** (not a synthetic degradation):

- CT: Radon transform (projection to the sinogram domain)
- MRI: FFT + an undersampling mask
- Ultrasound: scattering + attenuation model

**Unrolled networks** can be used — unrolling traditional iterative algorithms (such as ADMM, conjugate gradient) into a neural network, where each step contains both a data-consistency term (forcing $\hat{x}$ to be consistent with $y$) and a learned prior term.

### A simplified unrolled network structure

```python
class UnrolledMRIRecon(nn.Module):
    """A simplified MRI reconstruction unrolled network.
    Each step: data-consistency projection + learned denoising.
    """

    def __init__(self, num_iter: int = 5):
        super().__init__()
        self.num_iter = num_iter
        self.denoisers = nn.ModuleList([UNet(in_ch=2, out_ch=2) for _ in range(num_iter)])

    def forward(self, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        y: (B, 2, H, W) under-sampled k-space (real, imag)
        mask: (B, 1, H, W) sampling mask (1=sampled)
        """
        # Initialize: zero-filled IFFT
        x = ifft2c(y * mask)

        for denoiser in self.denoisers:
            # 1. Denoise step (learned)
            x = denoiser(x)

            # 2. Data consistency step (hard constraint)
            x_kspace = fft2c(x)
            x_kspace = mask * y + (1 - mask) * x_kspace
            x = ifft2c(x_kspace)

        return x
```

The data-consistency step (`mask * y + (1 - mask) * x_kspace`) is the key — the model can only act freely at **un-sampled** k-space positions, while the sampled positions must strictly use the actual measurements. This guarantees from the engineering side that "observed signals are not fabricated".

### Engineering practice

- Do not directly transplant a general-purpose SR model into medical
- When a physical model is available, prefer unrolled networks
- A radiologist must participate in evaluation (cannot rely only on PSNR/SSIM)
- Failure-mode analysis matters more than average metrics

Medical AI is a separate large field; this book only touches on it. Going deeper requires consulting professional medical-imaging-processing textbooks.

## 10.9 Satellite remote sensing and multispectral

Satellite imagery has several special points:

- **Multispectral**: in addition to RGB, there are NIR, SWIR, thermal infrared and other channels (4-13 channels are common)
- **Large scale**: a single image can be more than 10000×10000 pixels
- **Physical meaning**: each pixel value has concrete physical meaning (reflectance, temperature) and cannot be changed arbitrarily
- **The application is analysis (not visual appeal)**: vegetation indices, water-body extraction, land-cover classification — these tasks demand high precision in pixel values

Failures of general-purpose SR in these scenarios:

- Treats multispectral as RGB → spectral information is wholly wrong
- "Beautifies" the output → physical meaning is altered
- Cannot handle large-scale images → must tile, but general-purpose models have boundary issues when tiled

Task-specific directions:

- **HSI-SR** (hyperspectral image super-resolution) specialized models
- **Physics-constrained losses**: ensure that certain band ratios (such as the NDVI vegetation index) remain unchanged
- **Tile + overlap blending** (Section 9.9 of Chapter 9)

## 10.10 Microscopy super-resolution

Microscopy imaging has a complete physical theory:

- **PSF (Point Spread Function)** determines the resolution upper bound
- **The diffraction limit** is a physical constraint
- **Fluorescence microscopy** has a unique noise model (photon noise dominated)

Mainstream methods:

- **Physical modeling + learned priors**: knowing the PSF, use a learned denoiser as post-processing
- **STORM/PALM-style algorithms + neural network acceleration**: super-resolution microscopy
- **Cycle GAN-style**: transfer from one type of microscopy image to another (without paired data)

General-purpose SR models do not apply at all — they have no notion of the diffraction limit.

## 10.11 Task specialization in video enhancement (preview)

Video enhancement also has task-specific needs, which Chapters 13-14 will expand on:

- **Video conferencing face enhancement**: keep the speaker's face clear at low bandwidth
- **Live-stream video enhancement**: real-time (< 30ms/frame) + preserve brand color tone
- **Surveillance video enhancement**: identity preservation + no fabrication allowed (similar to forensic evidence)
- **Old-movie restoration**: film scratch removal + frame-rate up-conversion (with RIFE) + color restoration

These tasks each have their own inductive biases and constraints.

## 10.12 A methodology for designing task-specific models

If you want to design a task-specific model for a new task, follow this process:

### Step 1: identify the inductive bias of the task

Ask several questions:

- What **structural constraints** does this class of images have? (face: facial-feature positions; text: discrete strokes; medical: physical imaging)
- Which information **must absolutely not change**? (face: identity; text: characters; medical: lesions)
- Which information can be "created"? (face: high-frequency details such as wrinkle texture; text: cannot be created)

### Step 2: design prior injection

Encode the identified bias into the model:

- Discrete structure (text, CodeFormer) → codebook
- Strong semantic prior (face) → pretrained generator (StyleGAN)
- Physical process (medical) → unrolled network
- Multispectral physics → multi-channel architecture + spectral loss

### Step 3: design task-specific losses

The general-purpose enhancement losses (L1 + perceptual + adv) are not enough; add:

- **Identity preservation** (face)
- **OCR consistency** (text)
- **Data consistency** (medical)
- **Spectral consistency** (remote sensing)

### Step 4: design task-specific evaluation

General-purpose metrics (PSNR/LPIPS) may not reflect task quality:

- Face: identity similarity + subjective evaluation
- Text: OCR accuracy
- Medical: blind evaluation by radiologists + lesion-detection rate
- Remote sensing: downstream task accuracy (classification, segmentation)

### Step 5: failure-mode-targeted testing

General-purpose enhancement failure cases (covered in detail in Chapter 17) + task-specific failures:

- Face: extreme angles, occlusion, sunglasses, masks
- Text: handwriting, stamps, low contrast
- Medical: rare lesions, artifacts, motion blur

## 10.13 Summary

1. **General-purpose models will inevitably fail in certain domains** — face, text, medical, remote sensing, microscopy each have different inductive biases
2. **Face enhancement** is the most mature sub-field: StyleGAN prior + identity preservation + alignment
3. **GFPGAN** uses StyleGAN2 generator + CS-SFT modulation — a 2021 classic
4. **CodeFormer** uses a VQ codebook + Transformer — engineering-flexible with tunable fidelity
5. **Document enhancement** requires an OCR-guided loss + character-level correctness
6. **Medical enhancement** must use physical constraints + unrolled networks; **generative models cannot be used unconstrained**
7. **Remote sensing** requires multispectral physics + large-scale tiling
8. **Microscopy** must model the PSF + diffraction limit
9. **Design methodology**: identify bias → inject prior → specialized loss → specialized evaluation → failure-mode testing
10. **Task specialization is the enhancement direction with the largest marginal benefit** — general-purpose models can only reach 80 points, while specialized models can reach 99 points in their dedicated scenarios

This concludes all five chapters of Part II. Part III moves to the engineering details of training and evaluation — the previous chapters discussed "what to use", and the next two chapters discuss "how to train stably and how to evaluate accurately".

---

> Next chapter [Training stability](11-training.md) → engineering experience on GAN collapse, diffusion scheduling, and mixed loss weights.
