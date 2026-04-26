# Chapter 16 · Real-world cases

> The first 15 chapters covered models, losses, data, training, and deployment.
>
> This chapter **applies all of that to concrete scenarios**—each case provides a complete engineering pipeline and the reasoning behind each decision.
>
> This is the chapter closest to a "product manual" in the book.

## 16.1 Structure of this chapter

Each case contains:

1. **Scenario**: who the user is, what problem they want solved
2. **Key constraints**: quality, speed, device, cost
3. **Complete pipeline**: from input to output
4. **Key decisions**: why each step is the way it is
5. **Failure modes**: the pitfalls encountered in real deployments

Six cases:

```
16.2  Old-photo restoration (consumer product)
16.3  Low-light enhancement (mobile photography)
16.4  UGC video enhancement (short-video platforms)
16.5  4K live-streaming real-time enhancement
16.6  On-device ISP enhancement (mobile post-processing)
16.7  Surveillance video enhancement (security)
```

## 16.2 Case: old-photo restoration

### Scenario

A user has a photo from decades ago—black and white or yellowed, scratched, with a blurry face, possibly creased. They want it restored so it "looks like it was shot with a modern camera."

### Key constraints

- **Offline processing**: a few seconds of waiting after upload is acceptable
- **Quality first**: one bad result and the user never returns
- **Don't be too "creative"**: faces must remain recognizable as the same person

### Pipeline

```
Original old photo
  ↓
┌────────────────────────────┐
│ Step 1: Generic preprocess │
│ - Keep the original file   │
│ - Decode to a lossless     │
│   working format           │
│   (PNG/TIFF or tensor)     │
│ - Detect resolution to     │
│   pick downstream strategy │
└────────────────────────────┘
  ↓
┌────────────────────────────┐
│ Step 2: Local defect repair│
│ - Detect scratches /       │
│   creases / missing parts  │
│ - Inpaint with a model     │
│   (LaMa / SD-Inpaint)      │
└────────────────────────────┘
  ↓
┌────────────────────────────┐
│ Step 3: Colorize (if B&W)  │
│ - DeepRemaster / DDColor   │
│ - Optional style prompt    │
└────────────────────────────┘
  ↓
┌────────────────────────────┐
│ Step 4: General enhance    │
│         (background)       │
│ - SUPIR / Real-ESRGAN      │
│ - 4× super-resolution      │
│ - User-tunable fidelity    │
└────────────────────────────┘
  ↓
┌────────────────────────────┐
│ Step 5: Face-specific      │
│         (detect faces)     │
│ - face detector            │
│ - CodeFormer per face      │
│   fidelity_weight = 0.5    │
│ - Paste back into the image│
└────────────────────────────┘
  ↓
┌────────────────────────────┐
│ Step 6: Post-processing    │
│ - Color balance (Lab space)│
│ - Sharpening (USM, tunable)│
│ - High-quality JPEG (Q=95) │
└────────────────────────────┘
  ↓
Restored photo
```

### Key decisions

**Step 2 uses LaMa, not diffusion inpainting**:
- LaMa is fast and works well on small scratches
- Diffusion inpainting is slow and tends to "create" content (it might invent objects that aren't there)

**Step 4 lets the user tune fidelity**:
- 0.3: lean on the diffusion prior, the old-photo grain is replaced with "modern texture"
- 0.7: keep the original grain and period feel
- Different users have different preferences—don't hard-code

**Step 5 processes face crops individually**:
- General SR is poor on faces (Chapter 10)
- Detect faces → align → CodeFormer → paste back
- fidelity_weight = 0.5: identity preservation + moderate generation

### Failure modes

- **Multi-face scenes**: the detector misses some faces → those faces stay blurry
- **Profile / extreme angles**: the face detector fails
- **Old photos with stamps / handwriting**: the model treats them as "defects" and removes them
- **Clothing patterns**: the model "modernizes" them, losing period feel

Mitigation: provide UI for the user to **manually mark the priority restoration regions** instead of going fully automatic.

### Representative products

- Topaz Photo AI
- Tencent ARC (based on GFPGAN/CodeFormer)
- Microsoft Bringing Old Photos Back to Life

## 16.3 Case: mobile low-light enhancement

### Scenario

A photo taken at night looks dark, noisy, color-shifted, with low dynamic range. The user wants it to "look like iPhone Night Mode."

### Key constraints

- **On-device, real-time**: the result has to appear right after the shot
- **iPhone 14 ANE**: < 200 ms / 12 MP image
- **No "over-processed" look**: the user expects "naturally good"

### Pipeline

```
RAW / multi-frame RGB
  ↓
┌────────────────────────────┐
│ Step 1: Multi-frame fusion │
│         (HDR+)             │
│ - Burst 4–8 frames at shot │
│ - Inter-frame alignment    │
│   (optical flow)           │
│ - Average + weighted       │
│   (luminance-adaptive)     │
└────────────────────────────┘
  ↓
┌────────────────────────────┐
│ Step 2: Tone mapping       │
│ - Compress HDR to 8-bit    │
│ - Learned LUT (small CNN   │
│   predicts)                │
└────────────────────────────┘
  ↓
┌────────────────────────────┐
│ Step 3: Denoise            │
│ - On-device small NAFNet   │
│ - Physical noise model     │
│   (Section 5.7)            │
└────────────────────────────┘
  ↓
┌────────────────────────────┐
│ Step 4: Color restoration  │
│ - White balance correction │
│   (sensor → standard D65)  │
│ - Color saturation LUT     │
└────────────────────────────┘
  ↓
┌────────────────────────────┐
│ Step 5: Local enhance      │
│         (optional)         │
│ - Faces: light GFPGAN-Lite │
│ - Text / signs: legibility │
└────────────────────────────┘
  ↓
8-bit RGB image for display
```

### Key decisions

**Multi-frame fusion instead of single-frame enhancement**:
- A single ISO 6400 frame is overwhelmingly noisy; denoising loses detail
- Multi-frame fusion is equivalent to "more exposure"—it raises SNR at the source
- This is the core idea behind Google Pixel HDR+ and iPhone Night Mode

**On-device NAFNet instead of Restormer**:
- NAFNet has no attention—it's ANE-friendly
- Distilled to 1M parameters, single 12 MP frame inference ~100 ms

**Color restoration** matters more than "quality boost":
- Users can tolerate mild noise but cannot tolerate color shifts
- White balance drifts heavily under low light and must be corrected

### Failure modes

- **Camera moves during the burst**: multi-frame alignment fails → ghosting
- **Fast-moving objects in the scene**: incorrect fusion of moving objects
- **Extreme low light (< 1 lux)**: noise dominates—no number of frames will save it
- **Mixed light sources**: white balance is unsolvable (warm + cool light in the same frame)

### Representative implementations

- iPhone Night Mode (automatic multi-frame)
- Google Pixel Night Sight
- Huawei / Xiaomi AI night mode

## 16.4 Case: UGC video enhancement (short-video platforms)

### Scenario

Users upload low-quality videos (shot on phones, recompressed several times). The platform wants to improve quality automatically so the audience sees a clearer video.

### Key constraints

- **Offline batch processing**: must finish within minutes of upload
- **Massive scale**: millions of videos per day
- **Cost-sensitive**: per-second-of-video processing cost has to be controllable
- **Don't be too aggressive**: users expect "looks better", not "looks different"

### Pipeline

```
Original 720p 30fps video
  ↓
┌────────────────────────────┐
│ Step 1: Content analysis   │
│ - Classify content (face / │
│   landscape / animation)   │
│ - Decide enhancement path  │
│ - Detect low-quality clips │
└────────────────────────────┘
  ↓
┌────────────────────────────┐
│ Step 2: Denoise + de-      │
│         compression art.   │
│ - BasicVSR++ (small)       │
│ - Trained with temporal    │
│   consistency loss         │
└────────────────────────────┘
  ↓
┌────────────────────────────┐
│ Step 3: SR up to 1080p     │
│ - Same BasicVSR++ at 1.5×  │
│ - (Not 4×; perception is   │
│   sufficient)              │
└────────────────────────────┘
  ↓
┌────────────────────────────┐
│ Step 4: Color / brightness │
│ - Simple LUT grading       │
│ - Don't go "beauty filter" │
└────────────────────────────┘
  ↓
┌────────────────────────────┐
│ Step 5: Re-encode          │
│ - H.265 high quality       │
│ - Bitrate adaptive to      │
│   content                  │
└────────────────────────────┘
  ↓
1080p 30fps enhanced video
```

### Key decisions

**1.5× instead of 4× SR**:
- Users watch on phones; 1080p is plenty
- 4× compute is the square of 4×
- "Sharpening" 1080p → 1080p is perceptually close to 4× SR

**BasicVSR++ instead of SUPIR**:
- Video cannot fabricate (consistency issues)
- BasicVSR++ is fast (< 50 ms/frame)
- Cost: < $0.01 per minute-long clip

**Front-load content classification**:
- Landscape video: weaken denoising (preserve texture)
- Face-dominant: enable face enhancement
- Animation: skip SR (animation is already vector-styled)

### Failure modes

- **Source video is already high quality**: the model adds artifacts
- **Stylized video** (anime, oil painting): the model treats it as "low quality" and destroys the style
- **Very low frame-rate sources** (10 fps surveillance): SR makes it look worse
- **Heavy-motion scenes**: temporal consistency breaks

Mitigation: **the model must be able to recognize "no enhancement needed" and bypass**.

### Representative implementations

- TikTok / Douyin video post-processing (built-in SR)
- YouTube video upscaling
- Bilibili 4K restoration

## 16.5 Case: real-time 4K live-stream enhancement

### Scenario

A live-streaming platform wants to enhance hosts' 1080p feeds to 4K in real time so members see higher quality.

### Key constraints

- **Real-time**: < 33 ms/frame (30 fps)
- **Low latency**: end-to-end latency under 2 s (the live-streaming tolerance)
- **Cost**: per-stream GPU cost must amortize down
- **Stable**: must not crash under long-running operation

### Pipeline

```
1080p 30fps live feed
  ↓
┌────────────────────────────┐
│ Step 1: Decode (NVDEC)     │
│ - GPU hardware H.264 decode│
└────────────────────────────┘
  ↓
┌────────────────────────────┐
│ Step 2: Streaming denoise  │
│         + SR               │
│ - Distilled BasicVSR-Mini  │
│ - TensorRT FP16            │
│ - Causal model (history    │
│   only)                    │
│ - Keeps 2-frame hidden     │
│   state                    │
└────────────────────────────┘
  ↓
┌────────────────────────────┐
│ Step 3: Encode (NVENC)     │
│ - GPU hardware H.265 encode│
└────────────────────────────┘
  ↓
4K 30fps live feed (members)
```

### Key decisions

**Distillation + TensorRT are mandatory**:
- Vanilla BasicVSR++ 50 ms/frame → distilled 18 ms/frame → TensorRT 8 ms/frame
- Each A100 can serve 4–6 real-time streams (cost amortized)

**Causal model, no sliding window**:
- A sliding window needs "future frames", which adds latency
- Causal models look at history + current only
- ~0.3 dB PSNR loss, but latency drops from +166 ms to +0 ms

**Full GPU pipeline with NVDEC + NVENC**:
- Decode → enhance → encode all on GPU
- Never goes through CPU memory
- End-to-end latency < 100 ms

### Failure modes

- **Sudden scene cut**: the recurrent model's hidden state is wrong; the first frame breaks
- **Network jitter causing irregular input frames**: the model's real-time guarantee is destroyed
- **Heavy motion** (esports, racing streams): visible temporal inconsistency

Mitigation: **scene-cut detection + hidden-state reset**:

```python
def detect_scene_change(prev_frame, curr_frame, threshold=0.3):
    """Simplified scene-cut detection."""
    diff = (prev_frame - curr_frame).abs().mean()
    return diff > threshold

# Streaming loop
for frame in video_stream:
    if detect_scene_change(prev_frame, frame):
        model.reset_hidden_state()
    enhanced = model(frame)
    prev_frame = frame
    yield enhanced
```

## 16.6 Case: on-device ISP enhancement

### Scenario

A phone vendor wants its phones' photos to "look better than same-priced competitors out of the box." The intervention happens at the ISP (Image Signal Processor) level, not as post-processing.

### Key constraints

- **Sensor RAW input** (not RGB)
- **Multiple cameras** (main / ultra-wide / telephoto / macro)
- **On-device NPU** (Qualcomm Hexagon, Huawei Da Vinci, Apple ANE)
- **Ultra-low power**: shooting 1000 photos cannot drain the battery

### Pipeline (simplified phone ISP)

```
Sensor RAW (Bayer pattern, 12-bit)
  ↓
┌────────────────────────────┐
│ Pre-processing             │
│ - Black level correction   │
│ - Lens shading correction  │
└────────────────────────────┘
  ↓
┌────────────────────────────┐
│ Step 1: Learned demosaic   │
│ - Replaces classic         │
│   demosaicing              │
│ - Light on-device CNN      │
│ - RAW → full-res RGB       │
└────────────────────────────┘
  ↓
┌────────────────────────────┐
│ Step 2: Learned denoise    │
│ - Trained with physical    │
│   noise model              │
│ - On-device NAFNet variant │
└────────────────────────────┘
  ↓
┌────────────────────────────┐
│ Step 3: White balance +    │
│         color matrix       │
│ - Learned white balance    │
│ - Color matrix to standard │
│   color space              │
└────────────────────────────┘
  ↓
┌────────────────────────────┐
│ Step 4: Tone mapping       │
│ - Local tone mapping       │
│ - Learned HDR compression  │
└────────────────────────────┘
  ↓
┌────────────────────────────┐
│ Step 5: Sharpen + beauty   │
│         (optional)         │
│ - USM or learned local     │
│   sharpening               │
│ - Face beautification      │
│   (toggleable)             │
└────────────────────────────┘
  ↓
8-bit JPEG / HEIC
```

### Key decisions

**Learned demosaicing instead of classic**:
- Classic demosaicing (Malvar, AHD) produces zipper artifacts at edges
- Learned demosaicing avoids them and denoises simultaneously
- But it must be trained per sensor

**Bayer-domain vs. RGB-domain denoising**:
- Bayer domain (before demosaic): noise is i.i.d., denoising is simple but resolution drops
- RGB domain (after demosaic): noise is correlated, denoising is harder
- Vendors choose differently; the mainstream is Bayer-domain denoising → learned demosaic

**Face beautification must be toggleable**:
- Preferences differ across regions and cultures
- Legal risk (EU GDPR on biometric processing)
- Provide a user toggle in settings

### Failure modes

- **New sensor generation**: the model degrades on new sensors and must be retrained
- **Extreme scenes** (direct sunlight, near-darkness): training data is sparse, the model is OOD
- **Fast motion**: denoising and sharpening conflict
- **Rare objects**: the CNN's training data is limited

### Representative implementations

- Qualcomm Snapdragon ISP + Spectra
- Apple Photonic Engine
- Huawei XMAGE
- Google Pixel Camera

## 16.7 Case: surveillance video enhancement (security)

### Scenario

Footage from public-space cameras used for after-the-fact investigation. Source video is low quality (long-range, low light, heavily compressed); the goal is to enhance it to make details readable.

### Key constraints

- **No fabrication allowed** (forensic / legal-evidence grade)
- **Interpretable**: every processing step must be auditable
- **Chain of custody**: the processed video must be provably "the same clip" as the source

### Pipeline

```
Original low-quality surveillance video
  ↓
┌────────────────────────────┐
│ Step 1: Hash & archive     │
│ - SHA256 of original video │
│ - Timestamped digital sig. │
└────────────────────────────┘
  ↓
┌────────────────────────────┐
│ Step 2: Discriminative SR  │
│         (no diffusion)     │
│ - HAT / SwinIR             │
│ - Generative models banned │
└────────────────────────────┘
  ↓
┌────────────────────────────┐
│ Step 3: Denoise (mild)     │
│ - Conservative; prefer     │
│   noise over information   │
│   loss                     │
└────────────────────────────┘
  ↓
┌────────────────────────────┐
│ Step 4: Region-of-interest │
│ - User-selected region     │
│   (face / license plate)   │
│ - Region-specific          │
│   enhancement              │
└────────────────────────────┘
  ↓
┌────────────────────────────┐
│ Step 5: Processing record  │
│ - Log every model /        │
│   parameter used           │
│ - Output SHA256 + log      │
└────────────────────────────┘
  ↓
Enhanced video + processing log
```

### Key decisions

**Absolutely no diffusion models**:
- Diffusion synthesizes "plausible but nonexistent" detail
- "The model generated a license plate ABC123" is a courtroom incident
- Discriminative models only (CNN/Transformer)

**Conservative SR ratios**:
- 4× SR distorts severely degraded inputs
- Surveillance commonly uses 2× or even 1.5× (sharpening, not enlarging)

**Specialized for plates / faces**:
- General SR helps little with plate character recognition
- Use OCR-aware SR (Chapter 10)
- Use face SR with identity-preserving losses

### Failure modes

- **Extreme distance** (face < 20 pixels): unrecoverable by information theory
- **Extreme low light (< 0.1 lux)**: noise dwarfs signal
- **Heavy compression (< 200 kbps)**: too much information is lost

**The most important engineering discipline**: clearly tell users that **the model cannot create information out of thin air**—enhancement only makes existing information clearer.

### Representative implementations

- Hikvision / Dahua AI enhancement
- Forensic video tools used by law enforcement
- Adobe Premiere's Detail Boost (labeled "AI enhancement", not used as evidence)

## 16.8 General engineering takeaways

Engineering takeaways aggregated across cases:

### 1. There is no "universal best model"

- Old photos → diffusion-based SUPIR
- Live streaming → distilled BasicVSR-Mini
- Surveillance → discriminative SwinIR
- ISP → on-device NAFNet

Every scenario has its own best choice.

### 2. Pipeline > single model

Real products are always combinations:

- Detect → classify → enhance → evaluate
- A single model rarely covers every scenario

### 3. User-tunable parameters are mandatory

- Let the user choose fidelity
- Don't hard-code "the best" parameters
- A/B test the default

### 4. Failure handling matters more than success optimization

- A model performs well on 95% of scenarios
- How it handles the 5% breakage decides product experience
- "Graceful degradation on failure" beats "icing on the cake on success"

### 5. Continuous iteration

- Collect user failure cases
- Add to training data / failure-case set
- Iterate models on a quarterly cadence

## 16.9 Some anti-patterns

Common mistakes seen in real engineering:

### Anti-pattern 1: Drop in academic SOTA directly

Academic SOTA scores 33 dB on Set5; ported to a product, it may be worse than Real-ESRGAN—because of academic dataset bias.

### Anti-pattern 2: Ignore special scenarios

Optimize only for "typical natural images", ignoring text, barcodes, QR codes—but these constitute 10–20% of user uploads.

### Anti-pattern 3: Bigger model is better

Use SUPIR for real-time live streaming—5 seconds of latency. The user is gone.

### Anti-pattern 4: Skip end-to-end testing

The model is perfect on GPU PyTorch, but after deploying to ONNX Runtime certain ops degrade—quality drop only discovered after launch.

### Anti-pattern 5: Metrics ≠ user experience

PSNR went up by 1 dB, but users like it less—because contrast dropped.

## 16.10 Summary

1. **Each scenario has its own best model combination**—a universal optimum does not exist
2. **Old photos**: diffusion-based (SUPIR) + face-specific (CodeFormer) + user-tunable fidelity
3. **Low light**: multi-frame fusion + physical-noise denoising + color restoration
4. **UGC video**: distilled BasicVSR-Mini + temporal consistency + 1.5× SR
5. **Live streaming**: ultra-light + TensorRT + causal model + full-GPU pipeline
6. **ISP**: Bayer-domain denoising + learned demosaic + on-device NPU
7. **Surveillance**: discriminative + absolutely no diffusion + auditable processing log
8. **Pipeline > single model**: real products are multi-stage combinations
9. **User tuning is mandatory**: fidelity / style / strength
10. **Failure handling decides product experience**

The next chapter focuses on failure modes—the typical ways enhancement models fail in production.

---

> Next chapter: [Failure cases](17-failures.md) → typical ways enhancement models fail in production: fabrication, amplifying adversarial noise, identity drift, video flicker, quantization collapse.
