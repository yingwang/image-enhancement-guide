# Chapter 18 · SOTA models

> This chapter is the book's "quick reference"—7 SOTA models worth remembering plus a selection decision tree.
>
> The models themselves will go out of date; **the ideas will not**—so for each model we cover: core idea, when to use, when not to use, and who succeeds it.

## 18.1 Selection overview

```
General enhancement SOTA (3)
  ┌─ SUPIR        — diffusion-based / creative upscaling
  ├─ Real-ESRGAN  — real-degradation-modeling school
  └─ HAT          — Transformer non-diffusion school

Task-specific (4)
  ┌─ CodeFormer   — face restoration
  ├─ BasicVSR++   — video super-resolution
  ├─ RIFE         — frame interpolation
  └─ Retinexformer — low-light enhancement
```

Seven models, each representing a school of thought.

## 18.2 SUPIR: diffusion-based creative upscaling

### Core idea

Stitch the generative power of SDXL (a 2.6B-parameter text-to-image model) **onto the SR task**:

```
LR Image
  ↓ ControlNet injection
  ↓ SDXL UNet (frozen) + ZeroSFT modulation
  ↓ LLaVA auto-generates prompt as text condition
  ↓ Restoration-Guided Sampling
HR Image (creatively rich)
```

### One-line takeaway

"Use the world knowledge of a text-to-image model to guess the most plausible high-resolution version of a beat-up image."

### When to use

- **Heavily degraded old photos** (blurry faces, lost textures)
- **Applications optimizing for visual realism** (artistic upscaling, social-media restoration)
- **5–10 seconds per image inference is acceptable**

### When not to use

- Strict fidelity required (forensics, surveillance)
- Real-time required (video, live streaming)
- On-device deployment

### Key parameters

```python
{
    'num_inference_steps': 50,      # default 50
    'guidance_scale': 7.5,          # CFG strength
    'control_scale': 0.7,           # ControlNet strength
    's_stage1': -1,                 # adaptive noise start
    'restoration_guidance': True,   # use restoration-guided sampling
}
```

### Current standing

The flagship of diffusion-based SR in 2024. The 2025–2026 direction is moving toward **single-step / distilled**—TSD-SR, AdcSR, and other **one-step** works push diffusion-based SR from 50 steps down to 1–4 steps, with quality approaching SUPIR but speed an order of magnitude or two faster. For new projects in production, look at these distilled successors first rather than full SUPIR.

### Paper and code

- Paper: Yu et al. "SUPIR: Scaling Up Image Super-Resolution" (CVPR 2024)
- Code: [github.com/Fanghua-Yu/SUPIR](https://github.com/Fanghua-Yu/SUPIR)
- Distillation follow-ups: TSD-SR, AdcSR, OSEDiff, etc. (from 2025)

## 18.3 Real-ESRGAN: real-degradation-modeling school

### Core idea

Compared with ESRGAN, **the network barely changes** (still RRDB); **the contribution is entirely in the data**:

- Second-order degradation pipeline (Section 5.4)
- Complex noise synthesis (Gaussian + Poisson + real sensor)
- sinc filtering to simulate over-sharpening artifacts

### One-line takeaway

"The same network, but feeding training with real-world degradations transforms results."

### When to use

- **Generic enhancement in production**—balanced quality, speed, and stability
- **GAN fabrication is unacceptable** (Real-ESRGAN does use a GAN loss, but is far more conservative than diffusion)
- **Must run on a regular GPU / on-device**

### When not to use

- Heavy degradation (diffusion-based is more suitable)
- Ultra-light on-device (use the distilled Real-ESRGAN-Mini)

### Current standing

The de facto industry standard for enhancement models. Topaz Photo AI, CapCut, and many photo apps run on Real-ESRGAN or its variants under the hood.

### Paper and code

- Paper: Wang et al. "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data" (ICCVW 2021)
- Code: [github.com/xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)

## 18.4 HAT: non-diffusion Transformer SOTA

### Core idea

On top of SwinIR, **add more attention modules** to better exploit input information:

- Window Self-Attention (W-MSA + SW-MSA) — inherited from SwinIR
- **Channel Attention Block (CAB)** — adds RCAN-style channel attention
- **Overlapping Cross-Attention Block (OCAB)** — direct cross-window exchange

### One-line takeaway

"Push Transformer capacity to the limit—the current ceiling of the PSNR school."

### When to use

- **Academic benchmark competitions** (PSNR/SSIM-priority)
- **Need objective fidelity** and tolerate slow inference
- **As a teacher for knowledge distillation** (distill HAT into a small model)

### When not to use

- Real-degradation scenarios (HAT is trained on bicubic degradation and is not robust to real degradation)
- Real-time applications
- Resource-constrained settings

### Key numbers

- HAT-L on Set5 4×: **about 33.0–33.4 dB** (depending on training setup / ImageNet pretraining)
- ~40M parameters
- Inference: a single 256×256 takes ~80 ms on A100

### Current standing

The representative baseline for PSNR-oriented Transformer SR. **Not the only current SOTA**—models like DRCT and Hi-IR (2024–2025) trade wins with HAT across benchmarks. But HAT remains a good choice for teaching, distillation teachers, and comparison baselines. Rarely used in production—the field's "engineering ceiling" and "benchmark ceiling" are different things.

### Paper and code

- Paper: Chen et al. "Activating More Pixels in Image Super-Resolution Transformer" (CVPR 2023)
- Code: [github.com/XPixelGroup/HAT](https://github.com/XPixelGroup/HAT)

## 18.5 CodeFormer: face restoration

### Core idea

**Discretize** local face features into a number of codes in a codebook; predict the code sequence with a Transformer:

```
LR Face
  ↓ Encoder
  ↓ Transformer (predict discrete code indices)
  ↓ Codebook lookup
  ↓ Decoder
HR Face
```

Plus a tunable `fidelity_weight ∈ [0, 1]` so the user can choose between "strict fidelity" and "fully lean on the prior".

### One-line takeaway

"Use a VQ codebook—not a StyleGAN—as the strong face prior; the tunable fidelity is the key engineering highlight."

### When to use

- **Any face enhancement scenario**—this is the de facto standard
- **Old-photo faces** (fidelity = 0.5)
- **Video conferencing** (fidelity = 0.7, conservative, doesn't change appearance)

### When not to use

- **Forensic evidence** (no fabrication allowed)
- **Tiny faces** (< 32×32 pixels, no prior can recover them)

### Key parameters

```python
{
    'fidelity_weight': 0.5,      # 0 = pure prior, 1 = pure LR; 0.5 default balances
    'face_align': True,          # alignment is required first
    'crop_size': 512,            # pretraining uses 512 input
}
```

### Current standing

The engineering standard for face restoration. GFPGAN is still in use, but CodeFormer's tunability makes it more popular in products.

### Paper and code

- Paper: Zhou et al. "Towards Robust Blind Face Restoration with Codebook Lookup Transformer" (NeurIPS 2022)
- Code: [github.com/sczhou/CodeFormer](https://github.com/sczhou/CodeFormer)

## 18.6 BasicVSR++: video super-resolution

### Core idea

Bidirectional recurrence + second-order propagation + flow-guided DCN (covered in Section 14.3):

```
All-frame features
  ↓ Forward RNN (second-order propagation)
  ↓ Backward RNN (second-order propagation)
  ↓ Aggregate (concat + conv)
  ↓ Upsample
HR video
```

### One-line takeaway

"Bidirectional recurrence lets each frame use past and future information; second-order propagation routes around optical-flow error accumulation."

### When to use

- **VSR tasks** (de facto standard)
- **Video deblurring, video denoising** (same architecture, different training data)
- **Quality-first, no real-time requirement**

### When not to use

- Real-time live streaming (even BasicVSR++ is too slow; needs a distilled version)
- Extremely long videos (hidden-state error accumulation)

### Key numbers

- REDS4 4× VSR: 32.4 dB (previous SOTA EDVR was 31.1 dB)
- Inference: ~50 ms / 720p frame on A100

### Current standing

The engineering standard for VSR from 2022 to 2026. RVRT/VRT score higher PSNR but are much slower.

### Paper and code

- Paper: Chan et al. "BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment" (CVPR 2022)
- Code: [github.com/open-mmlab/mmagic](https://github.com/open-mmlab/mmagic) (inside MMEditing)

## 18.7 RIFE: frame interpolation

### Core idea

Don't explicitly estimate flow between the two end frames; **directly predict the flow from the middle frame to each end**:

```
F_t, F_{t+1}
  ↓ IFNet (Intermediate Flow Net)
  ↓ flow_to_t, flow_to_{t+1}, fusion_mask
  ↓ warp F_t with flow_to_t → warped_t
  ↓ warp F_{t+1} with flow_to_{t+1} → warped_t+1
  ↓ blend = mask * warped_t + (1-mask) * warped_t+1
F_{t+0.5}
```

### One-line takeaway

"Predict flow from the middle frame to the two ends directly, side-stepping the paradox that the middle frame doesn't yet exist."

### When to use

- **Any frame-interpolation task** (de facto standard)
- **30 fps → 60 fps, 60 fps → 120 fps**
- **Slow motion** (24 fps → 240 fps)
- **Real-time interpolation** (RIFE can hit 30+ FPS on 1080p)

### When not to use

- Extreme displacements (FILM is more robust)
- Severe-occlusion scenes (AMT is better)

### Key parameters

```python
{
    'scale': 1.0,        # 1 = 4K, 0.5 = more stable for 4K
    'tta': False,        # test-time augmentation, slow but more accurate
}
```

### Current standing

The de facto frame-interpolation standard. Successors (FILM, AMT) each have strengths, but RIFE has the best price-performance.

### Paper and code

- Paper: Huang et al. "Real-Time Intermediate Flow Estimation for Video Frame Interpolation" (ECCV 2022)
- Code: [github.com/megvii-research/ECCV2022-RIFE](https://github.com/megvii-research/ECCV2022-RIFE)

## 18.8 Retinexformer: low-light enhancement

### Core idea

Combine the classical Retinex theory (image = reflectance × illumination) with a Transformer:

```
Low-light image
  ↓ Decompose into Reflectance + Illumination (classical Retinex)
  ↓ Transformer (Illumination-Guided Transformer)
  ↓ Combine into the enhanced image
Enhanced image
```

### One-line takeaway

"Use the Retinex physical model as an inductive bias so the model focuses on learning reflectance invariance."

### When to use

- **Low-light photos** (night scenes, dim interiors)
- **Overexposure correction** (partial support)
- **Auxiliary for HDR tone mapping**

### When not to use

- Noise-dominated, very-dark scenes (< 0.1 lux)—the Retinex assumption fails
- Mixed light sources—Retinex simplifies the lighting model

### Current standing

A representative low-light enhancement model. Other choices:

- LLFormer (an earlier Transformer method)
- SCI (a lighter CNN method, on-device-friendly)

### Paper and code

- Paper: Cai et al. "Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement" (ICCV 2023)
- Code: [github.com/caiyuanhao1998/Retinexformer](https://github.com/caiyuanhao1998/Retinexformer)

## 18.9 Selection decision tree

Recommendations by scenario:

```
What's the task?
  │
  ├─ General SR / denoise / deblur
  │   │
  │   ├─ Heavy degradation + creativity-first → SUPIR
  │   ├─ Real-world scenarios + engineering stability → Real-ESRGAN  ←─ default
  │   └─ Academic benchmark / PSNR competition → HAT
  │
  ├─ Face enhancement → CodeFormer (user-tunable fidelity)
  │
  ├─ Video super-resolution → BasicVSR++
  │   ├─ Real-time live streaming → distilled BasicVSR-Mini + TensorRT
  │   └─ Offline high quality → BasicVSR++ or RVRT
  │
  ├─ Frame interpolation → RIFE
  │   ├─ Large displacements → FILM
  │   └─ Severe occlusion → AMT
  │
  ├─ Low-light enhancement → Retinexformer
  │
  ├─ Document enhancement → specialized DocSR + OCR-aware loss (Section 10.7)
  │
  └─ On-device real-time → distilled NAFNet + CoreML/TensorRT FP16
```

## 18.10 Academic SOTA vs. engineering SOTA

A theme that recurs throughout the book:

> The 7 models in this chapter are not "the highest-scoring".
>
> They are **the most worth deploying**—balancing quality, speed, stability, and maintainability.

Academic benchmark SOTA tends to:

- Improve over the baseline by 0.1–0.3 dB
- But have 5–10× worse parameters / inference speed
- Show no obvious gain in real-world scenarios

Engineering cares about:

- **Robustness** (doesn't break on common inputs)
- **Stability** (consistent results across hardware)
- **Deployability** (can be exported to ONNX, quantized, tiled)
- **Maintainability** (official code, ongoing maintenance, community)

These 7 models lead their categories on these dimensions.

## 18.11 SOTAs not on this list

Worth knowing but not given a dedicated section (by category):

### General SR families

- **DiffBIR**: diffusion-based, predates SUPIR, CLIP image cross-attention idea
- **SeeSR**: diffusion-based + semantic prior, more controllable
- **ResShift**: efficient sampling for diffusion-based
- **DRCT**: representative pure-CNN entrant in recent years
- **BSRGAN**: contemporaneous with Real-ESRGAN

### Faces

- **GFPGAN**: StyleGAN2-prior route, a 2021 classic
- **GPEN**: an early work from Tencent ARC
- **RestoreFormer++**: an evolution of CodeFormer

### Video

- **VRT / RVRT**: Transformer-based VSR
- **EDVR**: classic sliding-window approach
- **PropPainter**: representative video inpainting

### Frame interpolation

- **FILM**: from Google, strong on large displacements
- **AMT**: 2023 SOTA, strong on occlusion
- **VFIformer**: Transformer-based

### Low light

- **LLFormer**: early Transformer
- **SCI**: lightweight CNN
- **EnlightenGAN**: GAN-based

### Task-specific

- **DocSR / TextZoom**: text-SR datasets and baselines
- **SwinIR-Light, ESRGAN-Lite**: lightweight on-device

Each subfield has 5–10 models worth tracking. This chapter selected the **most representative and most engineering-friendly** ones.

## 18.12 When to switch SOTA

New models come out every month. **When should you swap your production model?**

Rule of thumb: only consider switching when all of the following hold:

1. **The new model is clearly better on your real test set** (not Set5/14)
2. **No regression on the failure-case suite** (doesn't introduce new problems)
3. **Acceptable inference speed** (no more than 1.5× the current)
4. **Training code and data are available** (reproducible, fine-tunable)
5. **There is active community maintenance** (not a single-paper drop with abandoned code)

Most paper SOTAs do not satisfy these 5—so **be conservative about switching models** is engineering wisdom.

> The cost of swapping a production model is far higher than "training a new one".
>
> A stable old SOTA usually beats an unstable new SOTA.

## 18.13 Closing

The book has covered:

- Part I: understanding the why (central equation, representation spaces, losses, evaluation, data)
- Part II: understanding the how (CNN, Transformer, diffusion, diffusion control, specialization)
- Part III: training and evaluation (stability, methodology)
- Part IV: video
- Part V: engineering deployment (inference, cases, failures)
- Part VI: reference (this chapter)

**Recap of the core viewpoints**:

1. Image enhancement is an **inverse problem**; the model is guessing, not recovering
2. Data > network (the core lesson of Real-ESRGAN)
3. PSNR school vs. perception school is a **theoretical inevitability**, not an engineering flaw
4. Loss functions in enhancement models are **almost never single-term**
5. Modern methods do not work in pixel space (latent space, feature space)
6. Diffusion models' "creating something from nothing" is both an advantage and a danger
7. In engineering, "failure handling" matters more than "average optimization"
8. Video is not just images × N; temporal consistency is an independent problem
9. On-device deployment requires a completely different optimization stack
10. **There is no universal optimum**—each scenario has its own best model combination

After reading this book, I hope you can:

- Look at a beat-up image and know which model combination to use
- Read a new paper and immediately tell which essential problem it tackles
- Design your own enhancement system knowing the trade-off behind each decision
- When something breaks in production, know where to look

The image enhancement field will keep evolving. Architectures will change, models will iterate, benchmarks will be refreshed. But the **thinking framework** this book covers—degradation models, representation spaces, loss design, evaluation methods, training dynamics, deployment constraints, failure modes—will remain valid.

Good luck. I hope this book serves you well.

---

> End of series.
>
> If you want to keep learning: companion books to this one:
> - [The Complete Guide for LLM Training Engineers](https://github.com/yingwang/llm-tutorial) — how to build LLMs
> - [Thinking in LLM](https://github.com/yingwang/thinking-in-llm) — how to use LLMs
