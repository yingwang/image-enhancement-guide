# Chapter 14 · VSR / Frame Interpolation / Video Inpainting

> Chapter 13 established the fundamental concepts of video enhancement—temporal consistency, optical flow, alignment.
>
> This chapter looks at concrete models: BasicVSR++, RVRT, RIFE/FILM, video stabilization.
>
> These models ground the concepts of Chapter 13, each with different engineering trade-offs.

## 14.1 Structure of this chapter

Four sub-tasks:

```
14.2-14.6  Video super-resolution (VSR): BasicVSR → BasicVSR++ → RVRT
14.7-14.8  Frame interpolation: RIFE, FILM, AMT
14.9       Video deblurring
14.10      Video inpainting
14.11      Video stabilization
```

Each task gets one representative model + a few engineering points.

## 14.2 The evolution of video super-resolution (VSR)

The VSR evolution path is similar to image SR but two years later:

```
2017  VESPCN     - first end-to-end VSR
2018  TDAN       - implicit alignment (DCN)
2019  EDVR       - sliding window + DCN alignment
2020  RBPN       - recurrent + multi-frame compensation
2021  BasicVSR   - bidirectional recurrent + explicit optical flow alignment
2022  BasicVSR++ - second-order propagation + flow-guided DCN
2022  VRT        - temporal Transformer
2023  RVRT       - efficient recurrent Transformer
```

**BasicVSR++** is the de facto standard for 2022-2024—simple, strong, fast. Detailed below.

## 14.3 BasicVSR++ in detail

Chan et al. proposed BasicVSR++ in 2022, a representative of the bidirectional recurrent architecture from Chapter 13, Section 13.8.

### Overall structure

```
                    Feature extraction
LR frames ─────────────→ feature maps F_t
                              │
                              ↓
        ┌─────────────────────┴─────────────────────┐
        │                                            │
        ▼ Forward propagation                        ▼ Backward propagation
        h^f_1 ─→ h^f_2 ─→ ... ─→ h^f_T              h^b_T ─→ h^b_{T-1} ─→ ... ─→ h^b_1
        Each step aligns the previous hidden state with optical flow
        │                                            │
        └─────────────────────┬─────────────────────┘
                              ↓
                      Aggregation (concat + conv)
                              │
                              ↓
                      Upsample (PixelShuffle)
                              │
                              ↓
                      HR frames
```

### Key innovation 1: second-order propagation

A plain bidirectional RNN uses only the previous step at each step:

$$
h_t = G(F_t, h_{t-1})
$$

BasicVSR++ uses **second-order**:

$$
h_t = G(F_t, h_{t-1}, h_{t-2})
$$

Why does this help?

- First-order: $h_{t-1}$ is warped from $h_{t-2}$, which was warped from earlier ones—optical flow errors accumulate along the way
- Second-order: directly access $h_{t-2}$, bypassing the accumulated error of $h_{t-1}$
- **Corrects local errors of the optical flow**

### Key innovation 2: Flow-Guided Deformable Alignment

Combines optical flow with deformable convolution:

- Use optical flow to give the deformable convolution an **initial sampling position**
- Let the DCN learn **a correction to the flow**

```python
class FlowGuidedDCN(nn.Module):
    """Flow-guided deformable alignment (BasicVSR++)."""

    def __init__(self, channels: int, num_groups: int = 8):
        super().__init__()
        # Predict the DCN offset correction (relative to optical flow)
        self.offset_conv = nn.Conv2d(
            channels * 2 + 2,         # h_prev + features + flow
            num_groups * 2 * 9,        # 9 sampling points × 2 dims × num_groups
            3, padding=1,
        )
        # The actual deformable conv (using torchvision's DeformConv2d here)
        from torchvision.ops import DeformConv2d
        self.dcn = DeformConv2d(channels, channels, 3, padding=1, groups=num_groups)

    def forward(self, h_prev, features_t, flow):
        """
        h_prev: previous hidden state (B, C, H, W)
        features_t: current frame features (B, C, H, W)
        flow: optical flow (B, 2, H, W)
        """
        # 1. Use flow to first warp h_prev to the current viewpoint
        warped_h = warp_with_flow(h_prev, flow)

        # 2. Network predicts DCN offset (correction)
        x = torch.cat([warped_h, features_t, flow], dim=1)
        offsets = self.offset_conv(x)
        # offsets shape: (B, num_groups * 2 * 9, H, W)

        # 3. Add the optical flow as the offset baseline (important!)
        # Make the DCN start from the flow's position; what it learns is the correction
        offsets = offsets + flow.repeat(1, offsets.shape[1] // 2, 1, 1)

        # 4. DCN samples at the corrected positions
        return self.dcn(h_prev, offsets)
```

Advantages of the flow + DCN combination: optical flow provides physical meaning, DCN provides local correction capability, **more stable than pure flow on fast-moving / partial-occlusion scenes**.

### Training BasicVSR++

- **Datasets**: REDS (240 video clips) + Vimeo-90K + self-synthesized degraded pairs
- **Degradation**: MM-CelebA-style video degradation + REDS-standard motion blur and compression
- **Loss**: primarily Charbonnier reconstruction loss (on every output frame)—temporal consistency mainly emerges naturally from **the architectural inductive bias of bidirectional propagation + flow-guided alignment**, rather than from an explicit temporal loss term
- **Training duration**: 1.6M steps on 8× A100, about 10 days

### Performance

On REDS4 4× VSR, PSNR is ~32.4 dB, clearly higher than EDVR (31.1) and BasicVSR (31.4). At the same time it **stays real-time**—about 30ms per frame on an A100.

## 14.4 Training data for VSR

Data requirements for VSR are higher than for image SR:

| Dataset | Number of videos | Resolution | Use |
|-------|-------|-------|------|
| **REDS** | 270 (train) + 30 (test) | 720P | General VSR standard |
| **Vimeo-90K** | 64,612 7-frame clips | 448×256 | Frame interpolation + VSR |
| **Vid4** | 4 clips | 720P | Test |
| **UDM10** | 10 clips | 1080P | Test |
| **YouHQ40** | 40 4K clips | 4K | Real high-quality |

### Video degradation synthesis

```python
def synthesize_video_pair(hr_video):
    """Synthesize an LR training pair from an HR video."""
    
    # 1. Temporally consistent degradation (same parameters across the clip)
    blur_kernel = sample_blur_kernel()        # fixed for one clip
    noise_sigma = sample_noise_sigma()        # fixed for one clip
    
    lr_frames = []
    for hr_frame in hr_video:
        x = apply_blur(hr_frame, blur_kernel)
        x = downsample(x, scale=4)
        x = add_noise(x, noise_sigma)
        lr_frames.append(x)
    
    lr_video = torch.stack(lr_frames)
    
    # 2. Video-specific degradation (whole clip together)
    lr_video = h264_compression(lr_video, bitrate=random.uniform(500, 5000))
    
    return lr_video
```

Note: **degradation parameters are fixed for one clip**. This is where it differs from images—if each frame uses different degradation parameters, it introduces "model-learned inconsistency."

## 14.5 VRT and RVRT: Video Restoration Transformer

Liang et al. proposed VRT in 2022 and improved it to RVRT in 2023. This line brings Transformers into VSR.

### Core idea

Drop the recurrent RNN and **directly use self-attention to aggregate across frames**:

```
F_1, F_2, F_3, F_4, F_5
   │   │   │   │   │
   └───┴───┼───┴───┘
           ▼
    Cross-frame attention
           │
           ▼
       Aggregated features
```

### Compute

Naive multi-frame attention has complexity that explodes—attention over 5 frames of $H \times W$ is $(5HW)^2$. VRT uses windows + temporal-axis attention to keep complexity in check.

### VRT vs BasicVSR++

| Dimension | BasicVSR++ | VRT/RVRT |
|------|-----------|---------|
| Performance | Strong | **Stronger** (PSNR +0.5 dB) |
| Speed | Fast | Slow (2-3×) |
| Memory | Medium | **Large** (attention) |
| Real-time | Possible | Difficult |
| Implementation complexity | Simple | Complex |

Engineering practice in 2026:

- Offline high-quality enhancement: RVRT or VRT
- Real-time / near real-time: BasicVSR++
- On-device: BasicVSR or lighter

## 14.6 Frame interpolation (VFI)

Frame interpolation is another video task—taking a low frame-rate video to a high frame rate (24 fps → 60 fps, 60 fps → 240 fps).

### Task definition

Given two adjacent frames $F_t$ and $F_{t+1}$, generate the intermediate frame $F_{t+0.5}$.

Note the two different contexts of "intermediate frame":

- **At inference**: there is no intermediate frame in the user's video—this is the difficulty of the task
- **At training**: the standard practice is to take three consecutive frames from a high-FPS video (e.g. 240 fps GoPro), use the first and third as input and the second as ground truth supervision—so **at training time, ground truth exists**

### RIFE (2022)

Huang et al.'s RIFE (Real-time Intermediate Frame Estimation) is the current de facto standard for frame interpolation. Core innovations:

- Does not explicitly estimate forward/backward optical flow; **directly estimates the optical flow from the intermediate frame to both ends**
- A single IFNet outputs both $F_{0.5 \to 0}$ and $F_{0.5 \to 1}$
- Use these two flows to warp $F_0$ and $F_1$ respectively, fuse to obtain $F_{0.5}$

```python
class RIFEStub(nn.Module):
    """Simplified RIFE skeleton."""

    def __init__(self):
        super().__init__()
        self.ifnet = IFNet()        # outputs flow from intermediate frame to both ends
        self.fusion_net = FusionNet()  # fuses warp results

    def forward(self, f0: torch.Tensor, f1: torch.Tensor) -> torch.Tensor:
        """
        f0, f1: two adjacent frames (B, 3, H, W)
        returns: intermediate frame f_0.5
        """
        # 1. Estimate flow from intermediate frame to both ends
        flow_to_0, flow_to_1, mask = self.ifnet(f0, f1)

        # 2. Warp the two end frames with the flows
        warped_0 = warp_with_flow(f0, flow_to_0)
        warped_1 = warp_with_flow(f1, flow_to_1)

        # 3. Mask-weighted fusion
        f_mid = mask * warped_0 + (1 - mask) * warped_1

        # 4. (Optional) final refinement with a refine network
        f_mid = self.fusion_net(f_mid, f0, f1)
        return f_mid
```

### Advantages of RIFE

- **Fast**: the original RIFE runs at 30 FPS on 1080P
- **Simple**: a single network end-to-end
- **Extensible**: recursive calls give $4\times$, $8\times$ frame-rate boosts

### FILM (Google, 2022)

Reda et al.'s FILM uses a different idea—multi-scale optical flow estimation + progressive synthesis:

- Does not depend on single-step optical flow estimation
- Recursively refines at multiple resolutions
- **More robust to large displacement** (severe-motion scenes)

In practice:

- **Slow motion** (ordinary video): RIFE and FILM are close
- **Fast motion** (sports, dance): FILM beats RIFE

### AMT (2023)

A more recent SOTA: builds on RIFE by adding attention modules; especially strong on **occlusion scenes**.

### Failure modes of frame interpolation

- **Large displacement**: object motion exceeds receptive field, ghosting in the interpolated result
- **New objects appearing** (disocclusion): the adjacent frames have no information about this object, cannot interpolate
- **Semi-transparent objects**: the optical flow assumption fails (glass, smoke)
- **Repetitive textures**: optical flow easily matches the wrong location (fences)

## 14.7 Video deblurring

The difference between video deblurring and image deblurring: **adjacent frames provide a sharp reference**.

### Key observation

Blur in video is often **intermittent**—one frame is blurred (instant of motion), the next is sharp (motion stopped). Exploiting this property substantially improves deblurring quality.

### EDVR

EDVR is not only the de facto classic for VSR but also a representative for video deblurring:

- Sliding window (5 or 7 frames)
- DCN alignment
- Spatio-temporal attention fusion

### MIMO-UNet (multi-input multi-output)

Independently process at different resolutions and then fuse—covers blur at multiple scales.

### Data: GoPro dataset

The standard benchmark for video deblurring: shoot with a high-speed camera (240 fps), average several adjacent frames to obtain a "blurred frame," with the original frame as ground truth.

## 14.8 Video inpainting / restoration

Video restoration includes two categories:

- **Video inpainting**: complete occluded or removed regions
- **Old film restoration**: remove scratches, flicker, missing frames

### Video Inpainting

Given a video and a mask (per-frame annotations of regions to fill), output the inpainted video.

Representative methods: **E2FGVI** (CVPR 2022), **ProPainter** (ICCV 2023)

Core idea:

1. Use optical flow to find the "corresponding pixels" of the mask region in other frames
2. Aggregate that information into the current frame
3. Use a transformer to fuse spatio-temporally

ProPainter's key improvement: a recurrent flow completion module that **first inpaints the optical flow** (the flow within the mask region is also missing), then uses the inpainted flow to guide frame inpainting.

### Old film restoration

Old film degradation has several special modes:

- **Scratches**: lines at random positions
- **Flicker**: whole-frame brightness/contrast jitter
- **Missing frames**: some frames are entirely missing
- **Color decay**: cyan/red shift

Engineering pipeline:

```
Step 1: Scratch removal (use the corresponding positions in adjacent frames)
Step 2: Flicker stabilization (correct inter-frame brightness)
Step 3: Missing frame inpainting (RIFE-class interpolation)
Step 4: Color recovery (Lab space statistical correction + learned color recovery)
Step 5: Enhancement (VSR + frame interpolation to 60 fps)
```

Representative projects:

- **DeepRemaster** (Iizuka & Simo-Serra 2019): old-film colorization + restoration
- **Bringing Old Films Back to Life** (Wan et al. 2022): full film restoration pipeline

## 14.9 Video stabilization

Shake comes from camera motion (handheld phones, action cameras). **Distinguishing it from real motion** is the difficulty.

### Classical methods

1. Use optical flow / feature point tracking to estimate the camera's global motion
2. Smooth this global motion trajectory
3. Use the smoothed trajectory to inverse-warp every frame

```
Original: camera shake + real scene motion
     ↓
Trajectory estimation + smoothing
     ↓
Re-warp: smoothed camera + real scene motion
```

### Modern methods

- **Stabnet** (learned stabilization)
- **Built into Google's Pixel Camera** (each phone vendor has its own scheme)
- Some phones use IMU data as an auxiliary signal (gyroscope is more accurate than image-based optical flow)

### The crop problem

Stabilization inevitably **crops the edges**—after warping the camera, the edges of the frame leave blank space. A typical trade-off:

- Strong stabilization: more cropping (smaller field of view)
- Weak stabilization: less cropping (field of view preserved)

## 14.10 Engineering combinations of video enhancement

Real products are usually not a single model but a pipeline:

### Example: phone video post-processing (vlog enhancement)

```
Original 1080P 30fps video
  ↓ Stabilization (StabNet)
  ↓ Denoising (BasicVSR family)
  ↓ Frame interpolation (RIFE) → 60 fps
  ↓ Super-resolution (BasicVSR++) → 4K
  ↓ Color correction (LUT-based)
4K 60 fps enhanced video
```

Each step may use a different model; the whole thing runs at about 5-10× real time on a GPU (5-10 seconds = 1 second of video).

### Example: livestream video enhancement

Strict real-time requirement (< 33ms/frame):

```
Original 720P 30fps live stream
  ↓ Lightweight denoising (small NAFNet, < 5ms)
  ↓ On-device super-resolution (distilled ESRGAN-Lite, < 20ms) → 1080P
  ↓ Color LUT (< 1ms)
1080P 30fps enhanced stream
```

Cannot use diffusion, cannot use heavy transformers, cannot use sliding windows (too slow)—only ultra-lightweight CNNs.

### Example: old film restoration (offline)

Real-time is not required; quality first:

```
Original 480P 24fps black-and-white old film
  ↓ Scratch removal (DeepRemaster)
  ↓ Flicker stabilization
  ↓ Colorization (DeepRemaster + manual adjustment)
  ↓ Restoration (Bringing Old Films Back to Life)
  ↓ Frame interpolation (RIFE) → 48 fps
  ↓ Super-resolution (Real-ESRGAN per-frame + temporal-consistency post-processing)
2K 48fps colorized restored version
```

## 14.11 Evaluating video enhancement models

Reviewing Chapter 13, Section 13.10, with a few concrete metrics added:

| Metric | Type | Tool |
|------|------|------|
| PSNR / SSIM | Single-frame | Standard |
| LPIPS | Single-frame perceptual | Standard |
| **tOF** | Temporal optical-flow consistency | Self-implemented |
| **tLPIPS** | Temporal perceptual consistency | Self-implemented |
| **VMAF** | Video quality assessment | Open-sourced by Netflix |
| **FVD** (Fréchet Video Distance) | Distribution distance | For generative video |

### VMAF

Netflix's video quality metric introduced in 2016, fusing several sub-metrics (VIF, ADM, motion score), with training data from real human ratings. **The standard for video quality in production**.

```python
# Invoke VMAF (with ffmpeg)
import subprocess

def compute_vmaf(reference_video, distorted_video):
    """Compute the VMAF score with ffmpeg + libvmaf."""
    cmd = [
        'ffmpeg', '-i', distorted_video, '-i', reference_video,
        '-lavfi', 'libvmaf=log_path=vmaf.json:log_fmt=json',
        '-f', 'null', '-'
    ]
    subprocess.run(cmd)
    # Parse vmaf.json
    import json
    with open('vmaf.json') as f:
        result = json.load(f)
    return result['pooled_metrics']['vmaf']['mean']
```

## 14.12 Summary

1. **VSR evolution**: sliding window (EDVR) → bidirectional Recurrent (BasicVSR++) → Transformer (RVRT)
2. **BasicVSR++ is the de facto standard in 2024**: bidirectional recurrent + second-order propagation + flow-guided DCN
3. **VSR has high data requirements**: degradation parameters are fixed for one clip during synthesis to avoid introducing inconsistency
4. **The three giants of frame interpolation (VFI)**: RIFE (fast), FILM (strong on large displacement), AMT (strong on occlusion)
5. **Video deblurring** exploits sharp copies in adjacent frames—an advantage image deblurring does not have
6. **Video restoration**: old films are the classic scenario; engineering is a multi-step pipeline rather than a single model
7. **Video stabilization** hinges on distinguishing camera shake from real motion
8. **The production pipeline is a combination**: not a single model, but a chain of stabilization + denoising + interpolation + SR
9. **Real-time enhancement is strictly constrained**: < 33ms/frame allows only lightweight CNNs
10. **VMAF is the de facto standard for production video quality assessment**

This completes Part IV's two video chapters. Part V moves into engineering deployment—the previous parts covered models themselves, this part covers how to push models into production (quantization, TensorRT, CoreML, mobile, tile).

---

> Next chapter [Inference Optimization](15-inference.md) → quantization, TensorRT, CoreML, torch.compile, dynamic resolution, tile inference.
