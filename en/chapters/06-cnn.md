# Chapter 6 · The CNN Era

> From 2014 to 2022, CNNs walked a distinctive path through low-level vision:
>
> from imitating sparse coding (SRCNN) → going deeper plus residual (VDSR/EDSR) → introducing attention (RCAN) → dense connections (RRDB) → reverse simplification (NAFNet).
>
> These eight years are the "classical mechanics" of this field — once you understand them, you can find the corresponding ideas in Transformers and diffusion.

## 6.1 Why start with CNNs

Transformers began making their mark in low-level vision from 2021 (SwinIR, Restormer, HAT), and diffusion has occupied the generative SOTA since 2023 (StableSR, SUPIR). It looks as if CNNs are already a thing of the past.

**That is not actually the case.**

- NAFNet (2022, pure CNN) is still the de facto benchmark for denoising / deblurring tasks
- Real-ESRGAN still uses RRDB (a CNN architecture from 2018)
- The patch embedding, upsample heads, and bottlenecks of nearly all Transformer models are still convolutions
- 99% of enhancement models deployed on mobile are pure CNNs (Transformer attention is far too inefficient on-device)

So the first stop in Part II must be CNNs. This chapter clarifies three things:

1. The evolutionary logic of CNNs in low-level vision (which designs are gradual refinements, which are paradigm shifts)
2. What concrete problem each era's representative network is solving
3. When designing your own CNN enhancement network, what to choose and what not to choose

## 6.2 SRCNN (2014) — the starting point

The first deep-learning super-resolution paper. Dong et al. mapped the traditional sparse-coding super-resolution pipeline to a three-layer CNN:

```
Layer 1 (9×9 conv, 64 ch)  ←→ Patch extraction & representation
Layer 2 (1×1 conv, 32 ch)  ←→ Non-linear mapping
Layer 3 (5×5 conv,  3 ch)  ←→ Reconstruction
```

Input: LR upsampled to target size by bicubic
Output: HR estimate

```python
import torch.nn as nn

class SRCNN(nn.Module):
    """SRCNN, 2014. The pioneering work, a three-layer CNN."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x is the bicubic-upsampled LR, its shape already equals HR
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
```

Performance: ~30.5 dB on Set5 4× (bicubic gives 28.4 dB).

**The value of SRCNN is not in its results**, but in the fact that it **proved end-to-end learning was viable** — all previous super-resolution methods were multi-stage pipelines of "first sparse dictionary learning, then patch classification, then reconstruction"; SRCNN turned all of this into a single CNN, opening up the next decade of development.

**Limitations of SRCNN**:

- **Too shallow**: only 3 layers
- **Upsampling first wastes computation**: all computation is performed at HR size
- **Large convolution kernels (9×9) are inefficient**: many parameters but a limited effective receptive field

All subsequent work was solving these problems.

## 6.3 VDSR (2016) — depth + residual

Kim et al. proposed VDSR (Very Deep SR), with two core contributions:

### Going deeper to 20 layers

VGG-style stacked 3×3 convolutions. The receptive field of 20 layers is much larger than SRCNN's, which lets it leverage broader context.

### Residual learning

Instead of directly learning HR, learn the **residual** $r = x - y_{\text{up}}$ (HR minus the upsampled LR).

Why is residual learning especially important in low-level vision?

- After LR is upsampled, $y_{\text{up}}$ is already close to the low-frequency component of HR
- The model only needs to learn the **high-frequency complement**, rather than relearning the entire image
- The variance of the residual is far smaller than that of the original image, **making optimization easier**
- The residual is mostly close to 0 (in flat regions), gradients are sparser, and the model converges more easily

```python
class VDSR(nn.Module):
    """VDSR, 2016. Deep + residual learning."""
    def __init__(self, num_layers: int = 20, base_ch: int = 64):
        super().__init__()
        layers = [nn.Conv2d(3, base_ch, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers += [nn.Conv2d(base_ch, base_ch, 3, padding=1),
                       nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(base_ch, 3, 3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        # x is the bicubic-upsampled LR
        return x + self.body(x)  # residual learning: output = input + learned residual
```

Performance: ~31.4 dB on Set5 4×, an improvement of about 0.9 dB over SRCNN.

**Residual learning has since become the de facto standard in low-level vision** — every subsequent network (EDSR, RCAN, RRDB, NAFNet) uses residuals.

**Limitations of VDSR**:

- It still goes through the network only after bicubic upsampling, wasting a lot of computation (at 4×, the computation is 16× the LR-space cost)

## 6.4 EDSR (2017) — removing BN, standardizing the residual block

Lim et al. proposed EDSR (Enhanced Deep Residual SR), the "engineering baseline" network for low-level vision. Several decisions in EDSR have influenced all subsequent CNN work.

### Decision one: remove Batch Normalization

The standard ResNet residual block is `Conv → BN → ReLU → Conv → BN`. The EDSR paper found that **removing BN actually works better in low-level vision**.

The reasons (this section is expanded a bit, because it matters):

- BN essentially normalizes features per mini-batch
- High-level vision (classification): normalization stabilizes training, and the classification objective is invariant
- Low-level vision (reconstruction): **the absolute pixel value is the output** — normalization breaks the scale relationship between input and output
- BN behaves differently across batches; **switching to EMA at test time creates a train-test mismatch**
- BN also restricts memory at large batch sizes

**After removing BN**:

- The model becomes more accurate in low-contrast or solid-color regions
- A larger network can be used (the saved memory can be invested in depth/width)
- Training stability actually improves

This is a **very important difference** between low-level and high-level vision: BN/LayerNorm are essential in high-level vision, but in low-level vision they are often harmful.

### Decision two: compute in LR space + PixelShuffle upsampling at the end

EDSR fixes the waste of computing in HR space that VDSR suffered from: all feature extraction is done in LR space, and the final upsampling uses **PixelShuffle** (sub-pixel convolution).

The essence of PixelShuffle: it converts the channel dimension into the spatial dimension. It rearranges $(B, r^2 C, H, W)$ into $(B, C, rH, rW)$.

```python
# PyTorch has built-in PixelShuffle, but understand the math:
# Input  (B, r^2 * C, H, W)
# Output (B, C, rH, rW)
# Rearrangement: every r^2 channels → an r×r spatial block
import torch.nn as nn

class UpsampleBlock(nn.Module):
    """EDSR-style upsample module.
    Supports 2/3/4×. 4× = two 2× steps."""
    def __init__(self, in_ch: int, scale: int):
        super().__init__()
        layers = []
        if scale in (2, 3):
            layers += [
                nn.Conv2d(in_ch, in_ch * scale * scale, 3, padding=1),
                nn.PixelShuffle(scale),
            ]
        elif scale == 4:
            for _ in range(2):
                layers += [
                    nn.Conv2d(in_ch, in_ch * 4, 3, padding=1),
                    nn.PixelShuffle(2),
                ]
        else:
            raise NotImplementedError(f"scale={scale} not supported")
        self.up = nn.Sequential(*layers)

    def forward(self, x):
        return self.up(x)
```

Why PixelShuffle is the de facto standard (vs nearest/bilinear/transpose conv):

- **Transposed conv**: produces checkerboard artifacts when stride and kernel size do not match
- **Nearest + conv** / **bilinear + conv**: works, but has poor parameter efficiency
- **PixelShuffle**: a learned reorder via a convolution with `r²` times the channels, the most parameter-efficient, and free of checkerboard artifacts

In real engineering PixelShuffle is the **default choice**. Unless there is a special reason (e.g. the deployment platform does not support it), do not use transposed conv.

### Decision three: standardizing the residual block

The EDSR residual block is just `Conv → ReLU → Conv` + residual, **with no normalization**. This design became the standard for the years that followed.

```python
class ResidualBlock(nn.Module):
    """EDSR-style residual block: no BN, no activation at the tail of the residual path."""
    def __init__(self, ch: int = 64, res_scale: float = 1.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class EDSR(nn.Module):
    """EDSR, 2017. The engineering baseline of low-level vision."""
    def __init__(self, scale: int = 4, num_blocks: int = 16,
                 ch: int = 64, res_scale: float = 0.1):
        super().__init__()
        self.head = nn.Conv2d(3, ch, 3, padding=1)
        self.body = nn.Sequential(*[
            ResidualBlock(ch, res_scale) for _ in range(num_blocks)
        ])
        self.body_tail = nn.Conv2d(ch, ch, 3, padding=1)
        self.upsample = UpsampleBlock(ch, scale)
        self.tail = nn.Conv2d(ch, 3, 3, padding=1)

    def forward(self, x):
        # x: LR (B, 3, H, W)
        feat = self.head(x)
        body = self.body_tail(self.body(feat)) + feat   # global residual
        out = self.tail(self.upsample(body))
        return out
```

**res_scale = 0.1** is a small engineering detail: it shrinks the output of the residual path by 10×, preventing residual accumulation in deep networks from causing numerical divergence. This trick has been widely adopted since EDSR.

## 6.5 RCAN (2018) — introducing attention

One direction after EDSR is to bring **attention mechanisms** into low-level vision. Zhang et al.'s RCAN is the representative.

### Channel Attention

Different channels learn different features — some channels are sensitive to **textures**, some to **edges**, and some may be redundant. Channel Attention lets the model learn a **channel weighting** that automatically amplifies important channels and suppresses unimportant ones.

```python
class ChannelAttention(nn.Module):
    """SE-style channel attention used by RCAN."""
    def __init__(self, ch: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, ch // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // reduction, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # global average pool -> channel squeeze -> channel restore -> sigmoid normalize
        # Produces per-channel weights (B, C, 1, 1) and broadcast-multiplies them back
        w = self.fc(self.avg_pool(x))
        return x * w
```

The concrete role of channel attention in low-level vision:

- **Suppressing noise channels**: noise features concentrate in certain channels, attention automatically downweights them
- **Amplifying edge channels**: edges are crucial to reconstruction, attention gives them larger weights
- **Adapting to input content**: an image rich in textures and a flat image need different channel weights

### Residual in Residual

RCAN nests residual blocks: each RCAB (a residual block with channel attention) is wrapped with another residual; multiple RCABs form a group, and the group is wrapped again with a residual. This structure can train networks of more than 400 layers stably.

```python
class RCAB(nn.Module):
    """Residual Channel Attention Block. The basic unit of RCAN."""
    def __init__(self, ch: int = 64, reduction: int = 16,
                 res_scale: float = 1.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            ChannelAttention(ch, reduction),
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale
```

Performance: ~32.6 dB on Set5 4×, an improvement of about 0.5 dB over EDSR.

## 6.6 RRDB (ESRGAN, 2018) — dense connections

Wang et al. proposed RRDB (Residual in Residual Dense Block) in ESRGAN. This is the **backbone Real-ESRGAN still uses to this day**.

### Dense Block

ResNet uses residual connections (add); DenseNet uses dense connections (concat). At each layer, the outputs of all previous layers are concatenated as input:

$$
x_{l+1} = H([x_0, x_1, \dots, x_l])
$$

Advantages:

- **Maximum feature reuse**: every layer directly sees all previous features
- **Mitigating gradient vanishing**: gradients can pass through any layer directly back to the input
- **Parameter efficiency**: the per-layer output channel count is small (growth rate), but features are rich

### The specific design of RRDB

Each RRDB contains 3 dense blocks; each dense block consists of 5 conv + LeakyReLU layers. There are dense connections inside each block, residual connections between blocks, and one more residual wrapping the whole RRDB.

```python
class DenseBlock(nn.Module):
    """The 5-layer dense block used by ESRGAN."""
    def __init__(self, ch: int = 64, growth: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(ch + 0 * growth, growth, 3, padding=1)
        self.conv2 = nn.Conv2d(ch + 1 * growth, growth, 3, padding=1)
        self.conv3 = nn.Conv2d(ch + 2 * growth, growth, 3, padding=1)
        self.conv4 = nn.Conv2d(ch + 3 * growth, growth, 3, padding=1)
        self.conv5 = nn.Conv2d(ch + 4 * growth, ch, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * 0.2 + x   # residual scale + residual


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""
    def __init__(self, ch: int = 64, growth: int = 32):
        super().__init__()
        self.db1 = DenseBlock(ch, growth)
        self.db2 = DenseBlock(ch, growth)
        self.db3 = DenseBlock(ch, growth)

    def forward(self, x):
        out = self.db1(x)
        out = self.db2(out)
        out = self.db3(out)
        return out * 0.2 + x   # one more residual
```

ESRGAN stacks 23 RRDBs in total, with about 17M parameters. This network is slightly lower than RCAN on the PSNR metric, but combined with GAN training its **visual quality** is far better than the pure-PSNR-optimized RCAN — this is the split between the PSNR camp and the perceptual camp (Section 4.8 of Chapter 4).

Real-ESRGAN reuses the RRDB network and only swaps the data pipeline and training losses, with completely transformed results. **This once again validates the conclusion of Section 5.1 in Chapter 5: data > network.**

## 6.7 NAFNet (2022) — counter-trend simplification

The trend in low-level vision from 2018 to 2022 was "adding more bells and whistles": transformer blocks, all sorts of attention, complex normalization. Chen et al.'s NAFNet went the other way — **removing everything that is not necessary** — and yet achieved SOTA on denoising / deblurring.

### Removal list

- ❌ Batch Normalization
- ❌ GELU / Swish (replaced by the simpler SimpleGate)
- ❌ Self-Attention (only an extremely simple channel attention is kept)
- ❌ ReLU (the Plain net uses no activation function at all)

Note: NAFNet **keeps a simplified LayerNorm2d** (once at the start of each block's spatial path and once at the start of the channel path); it does not remove all normalization layers. What it removes are the two "fancy architecture" categories of nonlinearity + attention; normalization is in fact essential for stable training.

### SimpleGate (replacing GELU)

GELU is $x \cdot \Phi(x)$ (input multiplied by the Gaussian CDF). NAFNet noticed that this is essentially "input multiplied by a gating signal" and simplified it into:

$$
\text{SimpleGate}(x) = x_1 \odot x_2
$$

where $x = [x_1, x_2]$ is the input split in half along the channel dimension and the two halves are multiplied element-wise.

```python
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
```

Visual impact: nonlinear capability similar to GELU, **with no learnable parameters and cheaper than GELU** (just chunk + element-wise multiplication, no erf/exp approximation).

### Simplified Channel Attention (SCA)

```python
class SCA(nn.Module):
    """NAFNet's simplified channel attention - keeps only average pool + 1x1 conv."""
    def __init__(self, ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        return x * self.conv(self.pool(x))
```

Note:

- No reduction (RCAN uses ch//16 → ch)
- No ReLU
- No sigmoid (multiplied directly, the weights are not normalized)

This "looks wrong" design actually works — it is a counter-intuitive finding from the NAFNet paper.

### The full NAFBlock

```python
class NAFBlock(nn.Module):
    """The core block of NAFNet."""
    def __init__(self, ch: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        # Spatial mixing
        self.norm1 = LayerNorm2d(ch)
        self.conv1 = nn.Conv2d(ch, ch * dw_expand, 1)
        # Depthwise
        self.dwconv = nn.Conv2d(ch * dw_expand, ch * dw_expand, 3,
                                padding=1, groups=ch * dw_expand)
        self.gate1 = SimpleGate()  # output channels halved
        self.sca = SCA(ch * dw_expand // 2)
        self.conv2 = nn.Conv2d(ch * dw_expand // 2, ch, 1)

        # Channel mixing (FFN)
        self.norm2 = LayerNorm2d(ch)
        self.conv3 = nn.Conv2d(ch, ch * ffn_expand, 1)
        self.gate2 = SimpleGate()
        self.conv4 = nn.Conv2d(ch * ffn_expand // 2, ch, 1)

        # Trainable scale
        self.beta  = nn.Parameter(torch.zeros((1, ch, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, ch, 1, 1)))

    def forward(self, x):
        # Spatial path
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.gate1(y)
        y = self.sca(y)
        y = self.conv2(y)
        x = x + y * self.beta

        # Channel path (FFN)
        y = self.norm2(x)
        y = self.conv3(y)
        y = self.gate2(y)
        y = self.conv4(y)
        return x + y * self.gamma


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for 2D feature maps."""
    def __init__(self, ch):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ch))
        self.bias = nn.Parameter(torch.zeros(ch))
        self.eps = 1e-6

    def forward(self, x):
        # (B, C, H, W) -> normalize over C
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
```

### What NAFNet teaches us

The most interesting part of the NAFNet paper is not its specific design but its **ablation study**:

- Replacing SE with SCA: no change in quality
- Replacing GELU with SimpleGate: no change in quality
- Removing all LayerNorms: a slight drop, but small
- ……

Conclusion:

> **The key for low-level vision is the allocation of compute budget, not fancy architecture.**
>
> Given the same FLOPs, simple and complex networks differ very little in quality;
> the "complexity" of complex networks mostly brings training instability and deployment difficulty.

This conclusion has a major impact on engineering practice — **prefer simple CNNs in production environments**, unless there is clear evidence that a complex network brings a qualitative leap.

## 6.8 Comparison of upsampling methods

The position and method of upsampling are critical design decisions in low-level vision.

| Method | Description | Pros | Cons |
|------|------|------|------|
| **bicubic + conv** | Input is bicubic-upsampled to HR, then CNN | Simple | Computation in HR space is expensive |
| **transpose conv** | Deconvolution | Learns upsampling | Checkerboard artifacts |
| **nearest + conv** | Nearest-neighbor copy + convolution | No artifacts | Poor parameter efficiency |
| **bilinear + conv** | Bilinear + convolution | Smooth | Slightly blurry |
| **PixelShuffle** | Sub-pixel conv | Highest parameter efficiency | Possible checkerboard early in training |
| **PixelShuffle (ICNR init)** | Initialization fix | Fixes the checkerboard issue | Slightly more complex implementation |

**ICNR initialization** (Initialization for Convolutional NN with sub-pixel convolutions):

```python
def icnr_init(tensor: torch.Tensor, scale: int = 2):
    """Convolution weight initialization before PixelShuffle, avoids checkerboard
    artifacts in the early stages of training.
    Essence: make the initial weights of the r^2 sub-groups identical."""
    out_ch = tensor.shape[0]
    sub_ch = out_ch // (scale ** 2)
    sub_kernel = torch.zeros(sub_ch, *tensor.shape[1:])
    nn.init.kaiming_normal_(sub_kernel)
    sub_kernel = sub_kernel.repeat(scale ** 2, 1, 1, 1)
    tensor.copy_(sub_kernel)
    return tensor
```

Engineering experience: **PixelShuffle + ICNR initialization by default**, the artifact problem is essentially eliminated.

## 6.9 Choice of normalization layer

The choice of normalization layer is very different in low-level vision than in high-level vision.

| Norm | Dim | In low-level vision | Recommendation |
|-------|---------|-------------|-------|
| **BatchNorm** | (B, H, W) | **Harmful** (breaks scale, unstable at test) | ❌ Don't use |
| **GroupNorm** | (G, H, W) per channel group | Acceptable | ⭐⭐ Acceptable |
| **InstanceNorm** | (H, W) per channel | Used for style transfer, not suitable for denoising | ⭐ Use with caution |
| **LayerNorm 2d** | (C) per pixel | Modern Transformer standard | ⭐⭐⭐ Recommended |
| **No Norm** | — | NAFNet/EDSR, etc. | ⭐⭐⭐ Recommended |

**Why is LayerNorm OK in Transformer-based low-level vision while BN is not?**

- BN normalizes across the batch dimension — the same pixel of the same image behaves differently in different batches
- LayerNorm normalizes across the channel dimension — it only looks at the feature vector of the current pixel; each image is independent

LayerNorm is deterministic for a single image, **with no train-test mismatch**.

Engineering practice:

- Pure CNN networks: **No Norm** (EDSR/NAFNet style)
- Transformer-based: **LayerNorm** (SwinIR/Restormer style)
- Never use BN

## 6.10 Choice of activation function

| Activation | Form | In low-level vision |
|------|------|-------------|
| **ReLU** | $\max(0, x)$ | Used by EDSR, simple and stable |
| **LeakyReLU** | $\max(0.01x, x)$ | Used by ESRGAN, avoids dead neurons |
| **GELU** | $x \cdot \Phi(x)$ | Standard for Transformers |
| **SiLU/Swish** | $x \cdot \sigma(x)$ | Modern default |
| **SimpleGate** | $x_1 \odot x_2$ | NAFNet, zero compute |

Engineering experience:

- **Conservative choice**: LeakyReLU(0.2). Stable in all GAN-style training, avoids gradient death
- **For efficiency**: SimpleGate
- **For quality**: SiLU (Swish)

## 6.11 Evolution of attention modules

Attention in CNNs is mainly channel attention. The evolution path:

| Module | Complexity | Origin |
|------|-------|------|
| **SE** | $C^2/r$ params | Squeeze-Excitation, 2018 |
| **CA (RCAN)** | Same as SE | Brought into low-level vision |
| **ECA** | $k$ params (k=3 or 5) | Efficient CA, 2020 |
| **SCA** | $C^2$ params, no reduction | NAFNet, 2022 |
| **Sim-AM** | 0 params | Energy-function-based |

**ECA (Efficient Channel Attention)** replaces SE's two FC layers with a 1D convolution:

```python
class ECA(nn.Module):
    """Efficient Channel Attention. Almost no parameters."""
    def __init__(self, ch: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        # x: (B, C, H, W)
        y = self.avg_pool(x).squeeze(-1).squeeze(-1)  # (B, C)
        y = y.unsqueeze(1)                             # (B, 1, C)
        y = self.conv(y).squeeze(1)                    # (B, C)
        y = torch.sigmoid(y).unsqueeze(-1).unsqueeze(-1)
        return x * y
```

### Spatial Attention in low-level vision

Modules like CBAM add spatial attention (letting the model focus on certain image positions). But **spatial attention has limited gains in low-level vision** — pixels at any position matter; there is no "position to ignore".

In real engineering, channel attention is the mainstream and spatial attention is rarely used. The self-attention of Chapter 7's Transformer is the real spatial-information modeling.

## 6.12 A complete EDSR-style model

Putting the above concepts together into a production-ready SR network. **This is the engineering baseline** — when you tackle a new task and don't know what to choose, start with this.

```python
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, ch: int, res_scale: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class UpsampleBlock(nn.Module):
    def __init__(self, ch: int, scale: int):
        super().__init__()
        layers = []
        if scale == 4:
            for _ in range(2):
                layers += [nn.Conv2d(ch, ch * 4, 3, padding=1),
                           nn.PixelShuffle(2)]
        elif scale in (2, 3):
            layers += [nn.Conv2d(ch, ch * scale * scale, 3, padding=1),
                       nn.PixelShuffle(scale)]
        self.up = nn.Sequential(*layers)

    def forward(self, x):
        return self.up(x)


class SRBaseline(nn.Module):
    """Production-ready SR baseline.
    Depth: 16 residual blocks + 64 channels, about 1.5M parameters.
    Real-world performance: on RealESRGAN-pipeline data, with L1+VGG+RaGAN losses,
            it reaches visual quality close to ESRGAN, with 2-3× faster training.
    """

    def __init__(self, scale: int = 4, num_blocks: int = 16,
                 ch: int = 64, res_scale: float = 0.1):
        super().__init__()
        self.head = nn.Conv2d(3, ch, 3, padding=1)
        self.body = nn.Sequential(
            *[ResidualBlock(ch, res_scale) for _ in range(num_blocks)]
        )
        self.body_tail = nn.Conv2d(ch, ch, 3, padding=1)
        self.up = UpsampleBlock(ch, scale)
        self.tail = nn.Conv2d(ch, 3, 3, padding=1)

        # ICNR initialization for the conv before PixelShuffle
        # Note on scale: 4× total uses two ×2 steps, so ICNR scale=2 here matches
        # the per-step scale inside UpsampleBlock. For ×3 direct upscaling, pass scale=3.
        per_step_scale = 2 if scale == 4 else scale
        for m in self.up.modules():
            if isinstance(m, nn.Conv2d):
                self._icnr_init(m.weight, scale=per_step_scale)

    @staticmethod
    def _icnr_init(weight, scale: int = 2):
        """ICNR: initialize the convolution weights before PixelShuffle so that
        the r^2 sub-groups have identical initial weights.
        scale must equal the upsampling factor of PixelShuffle.
        """
        out_ch = weight.shape[0]
        sub_ch = out_ch // (scale * scale)
        sub_kernel = torch.empty(sub_ch, *weight.shape[1:])
        nn.init.kaiming_normal_(sub_kernel)
        sub_kernel = sub_kernel.repeat(scale * scale, 1, 1, 1)
        weight.data.copy_(sub_kernel)

    def forward(self, x):
        feat = self.head(x)
        body = self.body_tail(self.body(feat)) + feat
        out = self.tail(self.up(body))
        return out
```

## 6.13 Engineering experience for designing an SR network

If you want to design a CNN enhancement network for a new task from scratch, consider the following in this order:

### Step 1: set the compute budget

- On-device (phone NPU): < 0.5G FLOPs, params < 1M
- Real-time on a desktop GPU: < 50G FLOPs
- Backend service (A100): up to 200G+ FLOPs
- Offline processing: no limit

### Step 2: allocate the budget between depth and width

Experience: **add depth before width**. Depth contributes more to receptive field and expressive power; width is just more channels and the marginal benefit decreases quickly.

Typical configurations:

- 1M params: 32 channels × 16 blocks
- 5M params: 64 channels × 16 blocks
- 16M params: 64 channels × 23 blocks (the ESRGAN configuration)
- 50M+ params: consider switching to a Transformer architecture

### Step 3: choose the upsampling location and method

- 99% use PixelShuffle, placed at the tail of the network
- Global residual from input LR to output HR (so the network only learns the high-frequency complement)

### Step 4: choose attention

- Generic tasks: add an SCA or ECA (extremely low overhead, gains 0.1-0.2 dB in PSNR)
- Tasks that especially need attention (faces, specific textures): consider CBAM or a Transformer block

### Step 5: training loss

- PSNR-oriented: a single Charbonnier loss
- Real-world SR: L1 + VGG + RaGAN (see Chapter 3)

## 6.14 Summary

1. **The evolution of CNNs in low-level vision is gradual**: SRCNN → VDSR (depth + residual) → EDSR (no BN) → RCAN (CA) → RRDB (dense) → NAFNet (simplification)
2. **Residual learning is the de facto standard in low-level vision** — every modern network uses it
3. **Batch Normalization is harmful in low-level vision** — it breaks scale; just remove it
4. **PixelShuffle + ICNR initialization is the de facto standard for upsampling**
5. **Channel Attention is an effective small improvement** (0.2-0.5 dB); SCA/ECA are low-overhead choices
6. **NAFNet's lesson**: in low-level vision, what matters is compute-budget allocation, not fancy architecture
7. **The order for designing a new network**: set budget → depth > width → PixelShuffle → add SCA → choose loss
8. **The success of Real-ESRGAN proves**: using 2018's RRDB + 2021's data pipeline outperforms using a 2022 new architecture + old data

The CNN story ends here. The next chapter brings Transformers into low-level vision and adds a new dimension to think about — **long-range dependency**. The receptive field of a CNN is local; the self-attention of a Transformer lets every pixel see every other pixel. This has different engineering consequences for denoising, deblurring, and super-resolution.

---

> Next chapter [Transformers in low-level vision](07-transformer.md) → SwinIR, Restormer, HAT, and how attention mechanisms take over part of the CNN's role.
