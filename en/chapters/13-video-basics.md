# Chapter 13 · Video is Not Just Images Plus Time

> "Run an image enhancement model on each frame and you have video enhancement."
>
> This is a **very common beginner mistake**.
>
> Video enhancement has the independent problem of temporal consistency—making each frame "good" individually does not guarantee the strung-together result is watchable.

## 13.1 An intuitive failure case

Take a low-quality video clip (720P phone recording) and run Real-ESRGAN 4× on each frame independently to upscale to 4K:

- **Looking frame by frame**: each frame is sharper, with crisper details
- **Played as video**: edges **flicker**, textures **shimmer**, details **ghost**

Specific symptoms:

- For a static object, each frame's details are slightly different
- When eyes track a moving object, the object's surface "boils"
- In flat regions (sky, walls), flickering textures appear

Why?

> Real-ESRGAN is a **generative** model—it "generates" details on every image.
> The same object differs slightly between two frames (noise, compression), and the generated details differ.
> Visually: **temporal inconsistency** = flickering.

This is not a bug in Real-ESRGAN; it is **a phenomenon that all models without temporal awareness exhibit on video**.

## 13.2 Temporal consistency is an independent problem

Redefine the goal of video enhancement:

- **Spatial quality**: each frame is sharp enough and detail-rich (the goal of image enhancement)
- **Temporal consistency**: changes between adjacent frames reflect only real scene changes, not model-introduced "noise"

The two goals **partially conflict**:

- More detail generation by the model → details on the same object may differ across frames
- More temporal stability → the model tends to output "safe blur"

The engineering challenge of video enhancement is **optimizing both goals at the same time**.

## 13.3 The physical basis of temporal consistency: optical flow

What does "temporally consistent" mean? Mathematically: between two adjacent frames, the colors of corresponding pixels should be (essentially) unchanged.

"Corresponding pixels" are defined by **optical flow**: the screen-space motion vector field of objects.

$$
F_t \to t+1 (x, y) = (u, v)
$$

Meaning: the pixel at position $(x, y)$ in frame $t$ has moved to $(x + u, y + v)$ in frame $t+1$.

**Brightness Constancy assumption**:

$$
I_{t+1}(x + u, y + v) \approx I_t(x, y)
$$

If the enhancement model obeys this assumption, the output video is temporally consistent; if not, it flickers.

### Optical flow engineering

Mainstream optical flow estimation methods:

- **Classical**: Lucas-Kanade, Farneback, TV-L1 (provided by OpenCV)
- **Deep learning**: FlowNet → PWC-Net → **RAFT** (2020, currently the most used) → GMA, SEA-RAFT

```python
# Estimate optical flow with RAFT (recommended: torchvision's pretrained version)
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

weights = Raft_Large_Weights.C_T_SKHT_V2
preprocess = weights.transforms()
model = raft_large(weights=weights, progress=False).eval().cuda()


def estimate_flow(frame_a: torch.Tensor, frame_b: torch.Tensor) -> torch.Tensor:
    """
    frame_a, frame_b: (B, 3, H, W) RGB in [0, 1]
    returns: (B, 2, H, W) flow from a to b (u, v)

    Notes:
    - RAFT requires H and W to be divisible by 8; otherwise pad to a multiple of 8 then crop back
    - preprocess does the normalization, do not normalize twice
    """
    a, b = preprocess(frame_a, frame_b)
    with torch.no_grad():
        flow_list = model(a, b)
    return flow_list[-1]   # final iteration result
```

### Warping: transforming images with optical flow

Given the optical flow $F_{t \to t+1}$, you can warp frame $t+1$ to the viewpoint of frame $t$:

```python
import torch
import torch.nn.functional as F

def warp_with_flow(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Use optical flow to warp image to the target position.
    image: (B, C, H, W)
    flow: (B, 2, H, W), unit is pixel displacement
    returns: warped image
    """
    B, _, H, W = image.shape
    # Create standard grid
    yy, xx = torch.meshgrid(
        torch.arange(H, device=image.device),
        torch.arange(W, device=image.device),
        indexing='ij',
    )
    grid = torch.stack([xx, yy], dim=0).float()  # (2, H, W)

    # Add flow
    grid = grid.unsqueeze(0) + flow              # (B, 2, H, W)

    # Normalize to [-1, 1] (required by grid_sample)
    grid_x = 2.0 * grid[:, 0] / (W - 1) - 1.0
    grid_y = 2.0 * grid[:, 1] / (H - 1) - 1.0
    grid_norm = torch.stack([grid_x, grid_y], dim=-1)  # (B, H, W, 2)

    return F.grid_sample(image, grid_norm, mode='bilinear',
                         padding_mode='border', align_corners=True)
```

Warping is a basic operation reused throughout video enhancement—use it to align adjacent frames, perform temporal aggregation, and compute temporal consistency loss.

## 13.4 Temporal consistency loss

When training a video enhancement model, add a temporal consistency loss:

```python
def temporal_consistency_loss(
    model_outputs: list,    # [out_t, out_t+1, ...]  enhanced frames
    flows: list,            # [flow_t→t+1, ...]
) -> torch.Tensor:
    """
    Use optical flow to verify: enhanced outputs of adjacent frames should
    satisfy brightness constancy.
    """
    loss = 0.0
    for t in range(len(model_outputs) - 1):
        out_t   = model_outputs[t]
        out_t_1 = model_outputs[t + 1]
        flow    = flows[t]

        # Warp out_{t+1} back to time t
        warped = warp_with_flow(out_t_1, flow)

        # Compute occlusion mask (positions where flow is unreliable)
        occlusion_mask = compute_occlusion_mask(flow)

        # Compute L1 at non-occluded positions
        loss = loss + (occlusion_mask * (out_t - warped).abs()).mean()

    return loss / max(1, len(model_outputs) - 1)
```

Add it to the total training loss:

$$
\mathcal{L} = \mathcal{L}_{\text{spatial}} + \lambda \mathcal{L}_{\text{temporal}}
$$

Empirical $\lambda$ is 0.1-0.5.

## 13.5 Occlusion: where optical flow is unreliable

The brightness constancy assumption fails in two situations:

- **Occlusion**: an object becomes hidden by foreground and disappears in frame $t+1$ (disocclusion is the reverse)
- **Motion blur + large displacement**: the optical flow estimation itself is wrong

The temporal consistency loss must **mask these positions out**. Common occlusion mask estimation:

### Forward-backward consistency

If forward flow $F_{t \to t+1}$ and backward flow $F_{t+1 \to t}$, after warping, can "return to itself," then the pixel is trustworthy:

```python
def forward_backward_check(flow_fwd: torch.Tensor,
                           flow_bwd: torch.Tensor,
                           threshold: float = 1.0) -> torch.Tensor:
    """
    Returns (B, 1, H, W) mask, 1 means the pixel is trustworthy.
    """
    # warp flow_bwd to the viewpoint of frame t
    warped_bwd = warp_with_flow(flow_bwd, flow_fwd)
    # Expectation: flow_fwd + warped_bwd ≈ 0
    diff = (flow_fwd + warped_bwd).norm(dim=1, keepdim=True)
    return (diff < threshold).float()
```

## 13.6 Specifics of video degradation

Video is not just "many images"; its degradation model has video-specific content:

### Video compression artifacts

H.264/H.265/AV1 compression is far more complex than JPEG. Beyond block artifacts, there are:

- **Smearing from motion estimation errors**: the encoder guesses the wrong direction of motion, leaving "ghost" trails on object edges after decoding
- **Periodic quality oscillation tied to keyframes (I-frames)**: I-frame quality is high, B/P-frames gradually degrade
- **GOP (Group of Pictures) boundaries**: every ~30 frames there is a GOP boundary, where compression error accumulation resets
- **Bitrate spikes**: in high-dynamic scenes the bitrate suddenly drops, quality collapses

### Motion blur (motion-related)

Blur from camera shake or fast object motion. This blur is **direction-correlated with motion**—different positions in the same frame may have differently oriented blur kernels.

The key to video deblurring: **use adjacent frames to provide a "sharp reference"**. If a region of frame $t$ is blurred but the same region in frame $t+1$ happens to be sharp (instant when motion stopped), it can be borrowed.

### Rolling shutter

CMOS sensors scan row by row; objects moving during the scan time produce geometric distortions:

- Static objects keep their original shape
- Horizontally moving objects appear "skewed"
- Rotating objects show a "spiral twist"

This is a degradation specific to phone video; old films/celluloid don't have it. Specialized de-rolling algorithms exist but are outside the scope of general video enhancement.

### Video-specific synthesis pipeline

The image degradation pipeline from Chapter 5 needs **further extension** for video:

```python
class VideoDegradation:
    """Video degradation synthesis pipeline."""

    def __init__(self, image_degrader_cls):
        # image_degrader_cls should accept explicit degradation parameters (blur_kernel, noise_sigma, ...)
        # so we can share one set of parameters across a clip rather than resampling per frame
        self.image_degrader_cls = image_degrader_cls

    def __call__(self, hr_video: torch.Tensor) -> torch.Tensor:
        """
        hr_video: (T, 3, H, W) high-quality video frames
        returns: lr_video (T, 3, H/4, W/4) degraded
        """
        T = hr_video.shape[0]

        # 1. Key: a clip shares one set of degradation parameters
        # blur kernel, noise sigma, JPEG quality stay fixed for a clip; otherwise the model
        # will learn the incorrect distribution where "every frame is degraded differently",
        # which introduces temporal inconsistency
        clip_params = self._sample_clip_params()
        image_degrader = self.image_degrader_cls(**clip_params)

        lr_video = torch.stack([
            image_degrader(hr_video[t:t+1])
            for t in range(T)
        ], dim=0).squeeze(1)

        # 2. Video-specific degradations (whole clip together)
        lr_video = self._add_motion_blur(lr_video)
        lr_video = self._video_compression(lr_video)
        lr_video = self._frame_drop_jitter(lr_video)  # occasionally drop / repeat frames

        return lr_video

    def _sample_clip_params(self):
        """Sample one set of degradation parameters, shared by the entire clip."""
        return {
            'blur_sigma': random.uniform(0.5, 3.0),
            'noise_sigma': random.uniform(0.005, 0.05),
            'jpeg_quality': random.randint(40, 95),
        }

    def _video_compression(self, video):
        """Simulate H.264/H.265 compression.
        In practice, encode/decode via ffmpeg, called engineering-wise via torchvision.io
        or external invocation.
        """
        # Simplified: omitted
        return video

    def _add_motion_blur(self, video):
        """Add motion-direction-related motion blur per frame."""
        # Simplified: omitted
        return video

    def _frame_drop_jitter(self, video):
        """Simulate frame drop / duplicate jitter."""
        return video
```

Engineering challenges of video degradation synthesis:

1. **Large data volumes**: 30 frames per second, a multi-minute video = thousands of images
2. **Temporal consistency**: certain degradation parameters should be fixed for one clip, to avoid the model learning a wrong degradation distribution
3. **Differentiable ffmpeg**: ideally a differentiable H.264 encoder; there is no perfect solution at present

## 13.7 The main video enhancement tasks

| Task | Input | Output | Main challenge |
|------|------|------|---------|
| **Video SR (VSR)** | Low-resolution video | High-resolution video | Temporal consistency + leveraging multi-frame information |
| **Video denoising** | Noisy video | Clean video | Leverage "clean copies" from adjacent frames |
| **Video deblurring** | Blurred video | Sharp video | Blur kernel varies in space and time |
| **Frame interpolation (VFI)** | Low frame-rate video | High frame-rate video | Intermediate frames don't exist and must be generated |
| **Video stabilization** | Shaky video | Stabilized video | Distinguish camera shake from real motion |
| **Video colorization** | Black-and-white video | Colored video | Color temporal consistency |
| **Video inpainting** | Scratched / missing-content video | Clean video | Use prior/next frames to fill |

## 13.8 Two paradigms of video enhancement

By "how to leverage multi-frame information," there are two categories:

### Sliding window

Each call processes $N$ frames (typically $N = 5$ or $7$) and outputs the enhancement of the middle frame:

```
Window 1: [F_1, F_2, F_3, F_4, F_5] → F_3'
Window 2: [F_2, F_3, F_4, F_5, F_6] → F_4'
Window 3: [F_3, F_4, F_5, F_6, F_7] → F_5'
...
```

Representatives: EDVR, ToFlow

Pros:

- Simple training (each window is independent)
- Inference can be parallel (each window is independent)

Cons:

- Limited temporal window (cannot see beyond 5 frames)
- Tedious boundary handling (start and end of the video)

### Recurrent

Each frame's processing leverages the hidden state of the previous frame:

```
F_1 → F_1' + h_1
F_2, h_1 → F_2' + h_2
F_3, h_2 → F_3' + h_3
...
```

Representatives: BasicVSR, BasicVSR++, IconVSR

Pros:

- Long-range temporal information available (in theory unlimited)
- Inference memory is fixed (only the current hidden state is kept)

Cons:

- Complex training (BPTT)
- Inference must be serial (cannot parallelize across frames)

### Hybrid: Bidirectional Recurrent

BasicVSR introduced **bidirectional** recurrence:

```
Forward:  F_1 → F_2 → F_3 → ...    produces h^f_t
Backward: ... → F_3 → F_2 → F_1    produces h^b_t
Merge:    F_t' = G(F_t, h^f_t, h^b_t)
```

This lets each frame leverage past and future information, and is the de facto standard for VSR. Discussed in detail in Chapter 14.

## 13.9 Temporal alignment: making multi-frame information actually usable

When aggregating multi-frame information, you must first **align**—warp the other frames to the viewpoint of the current frame.

### Explicit alignment (using optical flow)

```python
def explicit_align(frames: torch.Tensor, ref_idx: int = None) -> torch.Tensor:
    """
    Warp all frames to the viewpoint of the center frame.
    frames: (B, T, C, H, W)
    """
    if ref_idx is None:
        ref_idx = frames.shape[1] // 2

    ref = frames[:, ref_idx]
    aligned = []
    for t in range(frames.shape[1]):
        if t == ref_idx:
            aligned.append(ref)
            continue
        flow = estimate_flow(ref, frames[:, t])
        aligned.append(warp_with_flow(frames[:, t], flow))
    return torch.stack(aligned, dim=1)
```

Pros: clear physical meaning, interpretable.
Cons: depends on optical flow accuracy; flow errors cascade.

### Implicit alignment (deformable convolution)

EDVR and similar models use **DCN (Deformable Convolution Network)** to learn alignment:

- No explicit optical flow estimation
- The convolution kernel's sampling positions are learned
- The network handles "where to look" automatically

Pros: can handle situations where optical flow is hard to estimate (semi-transparent, complex occlusion).
Cons: complex CUDA implementation, hard to interpret.

### Attention alignment

A more modern approach: aggregate across frames with self-attention:

- Each pixel attends to all pixels in adjacent frames
- The network automatically learns the correspondence

Representatives: VRT, RVRT. Heavy compute but strong results.

## 13.10 Evaluation of video enhancement

Video evaluation extends the metrics in Chapter 4 and Chapter 12 with:

### Temporal metrics

- **tOF (temporal Optical Flow consistency)**: use optical flow to verify adjacent-frame consistency
- **tLP (temporal LPIPS)**: LPIPS distance between adjacent frames

```python
def temporal_lpips_warped(frames: torch.Tensor, flows: list,
                          lpips_fn) -> float:
    """
    frames: (T, C, H, W) — enhanced video frames
    flows: length T-1, each is (1, 2, H, W) forward flow t -> t+1
    returns: average LPIPS between adjacent frames (after motion compensation).
    Smaller is more consistent.

    Important: do NOT directly compute LPIPS(frame[t], frame[t+1]); that would
    count real motion as "temporal inconsistency". You must align first by warping
    with optical flow.
    """
    losses = []
    for t in range(frames.shape[0] - 1):
        warped = warp_with_flow(frames[t+1:t+2], flows[t])
        losses.append(lpips_fn(frames[t:t+1], warped).item())
    return sum(losses) / len(losses)
```

### Video-specific subjective evaluation

You cannot just look at single frames—you must watch playback. Procedure:

1. Play a complete video clip (5-10 seconds) for the rater
2. Rate using MOS or 2AFC
3. Loop the playback multiple times (so the rater can notice details)

Subjective evaluation is **even more important for video than for images**—flickering is something the eye sees but per-frame metrics miss.

### Evaluation duration

Sample sizes for video evaluation are larger. Reasons:

- Flicker/artifacts are intermittent
- Different scenes behave differently
- Raters need a few seconds of viewing before forming a judgment

Empirical: video subjective evaluation uses 5-10 second clips, at least 50-100 clips per model.

## 13.11 Differences between video enhancement and video generation

To avoid confusion—**video generation** (Sora, Kling, Veo) and **video enhancement** are different directions:

| Dimension | Video enhancement | Video generation |
|------|---------|---------|
| Input | Existing low-quality video | Text prompt (no video input) |
| Goal | Improve quality, preserve content | Create a new video |
| Temporal constraint | Strictly follows the original timing | Free generation |
| Evaluation | Compare against GT | Subjective rating / FVD |
| Current SOTA | BasicVSR++, Topaz Video AI | Sora, Kling, Veo3 |

This book covers only the former.

## 13.12 Engineering challenges of video enhancement

Beyond the algorithm, video enhancement has several engineering-specific difficulties:

### Data storage and loading

- Video is orders of magnitude larger than images (each clip on the order of GB)
- Random clip sampling at training time
- Decoding H.264/H.265 is a CPU bottleneck
- NVDEC hardware decoding is recommended

```python
# Use torchvision.io for hardware decoding (PyTorch 2.0+)
import torchvision

reader = torchvision.io.VideoReader("video.mp4", "video", num_threads=4)
for frame in reader:
    pts = frame['pts']
    img = frame['data']
    # process...
```

### Memory management at inference

5 frames of 4K video = 5 × 4 × 3840 × 2160 × 4 bytes ≈ 400 MB (FP32). Model activations are tens of GB. **Must use patches + tiles + streaming processing**.

### Real-time enhancement

Video conferencing/livestreaming requires < 33ms per frame (30 FPS). In this scenario:

- Cannot use diffusion models
- Cannot use heavy attention networks
- Must be CNN + on-device acceleration
- May need to trade quality for speed

Chapter 15 will discuss on-device/real-time deployment in detail.

## 13.13 Summary

1. **Video enhancement is not image enhancement × N**—temporal consistency is an independent goal
2. **Flickering comes from temporal inconsistency**—good frames don't imply good video
3. **Optical flow is the foundation of temporal processing**—RAFT is the current standard
4. **Warping + brightness constancy** define "temporally consistent"
5. **Occlusion masks are mandatory**—loss should not be computed where flow is unreliable
6. **Video degradation is more complex**: H.264/H.265 compression, motion blur, rolling shutter
7. **Sliding window vs Recurrent**: two paradigms with different trade-offs
8. **Bidirectional Recurrent (BasicVSR)** is the de facto standard for VSR
9. **Alignment methods**: explicit optical flow, deformable convolution, cross-frame attention
10. **Video evaluation must include temporal metrics** (tOF/tLPIPS) + **subjective playback**
11. **Engineering challenges**: data storage, decoding bottlenecks, memory management, real-time constraints

In the next chapter we look at concrete video enhancement models: BasicVSR++, RIFE frame interpolation, video inpainting—grounding the concepts in this chapter into concrete networks.

---

> Next chapter [VSR / Frame Interpolation / Video Inpainting](14-video-models.md) → BasicVSR++, RIFE, FILM, and video stabilization.
