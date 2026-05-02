# 第 9 章 · 扩散的条件控制

> 第 8 章讲了扩散模型的基础——给定噪声、预测去噪。
>
> 但影像增强任务的关键不在去噪能力，在**怎么让扩散模型听话**——既利用它的生成能力（创造合理细节），又严格遵守 LR 输入（不偏离原图）。
>
> 这一章是过去三年这个领域最活跃的工程战场。

## 9.1 核心问题：fidelity vs creativity

第 8 章末尾讲过扩散的"无中生有"能力——这是它的优势，也是它的危险。

放大同一张老人脸 LR 图，扩散模型可能：

- **过强遵守 LR**：输出和 LR 一模一样，糊得不行（没利用生成能力）
- **过弱遵守 LR**：编造细节，脸变了样（生成能力失控）
- **平衡**：遵守 LR 的整体结构、用生成能力补合理细节

如何控制这个平衡点，就是本章的主题。

> 影像增强里"条件控制"的实质：
>
> 让模型在每一步去噪时，都把"$\hat{x}_0$ 应该接近 $y$"这个约束**注入**进采样过程。

不同的注入方式（concat、cross-attention、ControlNet、IP-Adapter）效果差别很大。这一章把它们讲清楚。

## 9.2 五种条件注入范式

总览：

| 范式 | 注入位置 | 代表方法 | 训练成本 | 控制强度 |
|------|---------|---------|---------|---------|
| **Input Concat** | UNet 输入通道 | StableSR v1 | 低（改输入） | 中 |
| **Cross-Attention** | UNet 内部 attention | DiffBIR | 中（训 cross-attn） | 弱（语义级） |
| **ControlNet** | UNet 中间层加和 | StableSR v2、SUPIR | 高（复制 encoder） | 强 |
| **IP-Adapter** | 解耦 cross-attention | 风格保持 | 中 | 中 |
| **Tile + ControlNet** | 局部条件 | 大图增强 | （推理 trick） | 强 |

**没有哪个范式全胜**——选哪个看任务和预算。

## 9.3 Fidelity vs Creativity 的工程含义

把 perception-distortion trade-off（第 4 章 4.8 节）放到扩散语境下：

- **Fidelity 高**：输出像素一致性强，PSNR/SSIM 高，但视觉死板
- **Creativity 高**：模型自由发挥，视觉惊艳但可能编造（"幻觉"）

两个极端对应不同的应用：

- 监控录像→ 高 fidelity（不允许编人脸）
- 老照片修复 → 中等（结构保留，细节生成）
- 艺术放大、4K 直播创意增强 → 高 creativity（视觉冲击为主）

这一章讲的所有技术都是为了**让用户能在这条曲线上选点**——不仅训练时选，**推理时也能调**。

## 9.4 范式一：Input Concat

最简单的条件注入：把 LR 的 latent **拼到** UNet 的输入通道。

UNet 输入从 $(B, 4, h, w)$ 变成 $(B, 8, h, w)$，前 4 通道是当前噪声 latent，后 4 通道是 LR latent。

```python
def diffusion_step_concat(unet, x_t, lr_latent, t):
    """Input concat 风格的条件注入。"""
    inp = torch.cat([x_t, lr_latent], dim=1)  # (B, 8, h, w)
    return unet(inp, t)
```

修改 UNet 第一层 conv 接受 8 通道输入：

```python
import torch.nn as nn

# 原 UNet 第一层
old_conv = unet.conv_in  # in_channels=4

# 替换成 8 通道输入
new_conv = nn.Conv2d(8, old_conv.out_channels, kernel_size=3, padding=1)

# 重要: 只 copy 前 4 通道的权重, 后 4 通道初始化为 0
with torch.no_grad():
    new_conv.weight[:, :4] = old_conv.weight
    new_conv.weight[:, 4:] = 0
    new_conv.bias[:] = old_conv.bias

unet.conv_in = new_conv
```

后 4 通道初始化为 0 让训练初期 UNet 行为接近原模型——LR 信号慢慢起作用。

### Input Concat 的优缺点

**优点**：

- 最简单，几行代码改完
- 训练时只需要 fine-tune（UNet 大部分权重保留）
- 推理时无额外开销

**缺点**：

- LR 信息只在第一层注入，**深层信息会被稀释**
- 不容易调"控制强度"
- 对 LR 的尊重度不够（高 t 时，UNet 更"自由发挥"）

**StableSR v1** 用这种简单形式，效果不错但 fidelity 不够强，所以 v2 转向 ControlNet。

## 9.5 范式二：Cross-Attention 注入

不在输入层注入，在 UNet 内部的 cross-attention 注入——把 LR 通过某个 image encoder 编成 token，作为 cross-attention 的 KV。

最常见的 image encoder：CLIP。流程：

```
LR image
  ↓ CLIP image encoder
  ↓ (B, T_img, D)  image tokens (代替原 SD 的 text tokens)
  ↓ 注入到 UNet 每层的 cross-attention
```

```python
import open_clip

class CLIPImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip, _, _ = open_clip.create_model_and_transforms('ViT-L-14')
        self.proj = nn.Linear(self.clip.visual.output_dim, 768)  # 对齐 SD context dim

    def forward(self, lr_img):
        # 取 CLIP 倒数第二层的 patch tokens (而不是最终 [CLS])
        feat = self.clip.encode_image(lr_img, return_tokens=True)
        # feat: (B, T, D), 通常 T=257 for ViT-L/14
        return self.proj(feat)
```

UNet 的 cross-attention 不变（参考第 8 章 8.8 节），只是 KV 来源从 text encoder 换成 image encoder。

### Cross-Attention 注入的特点

**适合**：

- LR 信息以**语义级**为主（需要保持是猫还是狗，不需要逐像素一致）
- 风格 transfer 类任务

**不适合**：

- 高 fidelity SR（CLIP embedding 丢失了像素级信息）
- 文档/小字增强（结构信息靠 patch token 不够）

**DiffBIR**（2023）用 CLIP image encoder 注入语义条件，配合 ControlNet 注入结构条件——两者结合，是这个方向的代表设计。

## 9.6 范式三：ControlNet —— 这一章的主角

Zhang & Agrawala (2023) 的 ControlNet 是扩散控制的**事实标准**。它的核心设计：

> 复制一份 UNet 的 encoder，专门处理"控制信号"，输出**加和**到主 UNet 对应层的 skip connection。

主 UNet 完全不动（保留预训练权重），ControlNet 是一个**外挂**。这种设计的优势：

1. **保留预训练知识**：主 UNet 的所有能力（包括 text-to-image 的语义理解）不变
2. **训练参数比全量微调小**：ControlNet 复制主 UNet 的 encoder + mid block，可训练参数约为主 UNet 的 0.4–0.5×（SD 1.5 上 ControlNet ≈ 360M vs 主 UNet ≈ 860M）。但**显存开销不可控**——前向时主 UNet 仍要全量参与算 skip features，反向只有 ControlNet 那部分有梯度。SDXL 上单卡训 ControlNet 实测仍要 40GB+，不是 LoRA 那种"小成本"
3. **可以堆叠**：多个 ControlNet 同时作用（一个管 LR、一个管 edge map、一个管 depth map）

### ControlNet 的具体结构

```
                  Main UNet (frozen)
                  
LR ──→ Encoder copy ──→ Mid block copy
        (trainable)       (trainable)
              │              │
              ↓ zero conv    ↓ zero conv
              │              │
              ▼              ▼
       UNet skip 1-12    UNet mid
              │              │
              └──→ 加到主 UNet 对应位置
              
Output (B, 4, h, w) noise prediction
```

### Zero Convolution —— 核心 trick

ControlNet 输出加到主 UNet 之前，过一个 **zero-initialized 1×1 conv**——初始权重全为 0。

为什么这个细节重要？

- 训练初期，ControlNet 输出乘 0 = 0，**对主 UNet 完全无影响**
- 这意味着训练前期模型行为等同于原 SD（没有质量退化风险）
- ControlNet 的权重慢慢学到非零，控制信号渐强

```python
class ZeroConv(nn.Module):
    """ControlNet 的 zero convolution。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)
```

数学上，假设 ControlNet 输出 $h_c$，主 UNet 在某层的特征是 $h_m$。修改后：

$$
h_m' = h_m + Z(h_c)
$$

其中 $Z$ 是 zero conv。初始 $Z(h_c) = 0$，$h_m' = h_m$ —— 模型行为不变。训练让 $Z$ 学到非零权重后，控制信号开始作用。

### 简化的 ControlNet 实现

```python
import torch
import torch.nn as nn
import copy


class ControlNetForSR(nn.Module):
    """简化的 SR ControlNet。
    主 UNet 是预训练的 SD UNet (frozen)。
    ControlNet 复制其 encoder + mid block, 加 zero conv。
    """

    def __init__(self, sd_unet: nn.Module, lr_input_channels: int = 3,
                 latent_channels: int = 4):
        super().__init__()
        # 复制主 UNet 的 encoder + mid (浅拷贝结构, 深拷贝权重)
        self.input_blocks = copy.deepcopy(sd_unet.input_blocks)
        self.middle_block = copy.deepcopy(sd_unet.middle_block)

        # 关键: 复制后必须替换第一层 conv, 因为 SD UNet 的第一层是 4 通道入,
        # 而 ControlNet 接受 (x_t || lr_latent) 共 8 通道 (官方 ControlNet 用单独
        # 的 condition embedding, 这里为简洁直接 concat 到输入)
        old_conv = self._first_conv(self.input_blocks)
        new_conv = nn.Conv2d(
            latent_channels * 2, old_conv.out_channels,
            kernel_size=old_conv.kernel_size, padding=old_conv.padding,
        )
        with torch.no_grad():
            new_conv.weight[:, :latent_channels] = old_conv.weight
            new_conv.weight[:, latent_channels:] = 0      # 让多出的通道初始无效
            new_conv.bias[:] = old_conv.bias
        self._replace_first_conv(self.input_blocks, new_conv)

        # Zero convs 接每个 input_block 输出
        self.zero_convs = nn.ModuleList()
        for block in self.input_blocks:
            ch = self._get_block_out_ch(block)
            self.zero_convs.append(ZeroConv(ch, ch))
        # Mid block 也加 zero conv
        self.zero_conv_mid = ZeroConv(
            self._get_block_out_ch(self.middle_block),
            self._get_block_out_ch(self.middle_block),
        )

        # LR 在喂给 ControlNet 前的预处理 (RGB -> latent 大小)
        self.cond_pre = nn.Sequential(
            nn.Conv2d(lr_input_channels, 16, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, latent_channels, 3, padding=1),
        )

    @staticmethod
    def _first_conv(input_blocks):
        for m in input_blocks[0].modules():
            if isinstance(m, nn.Conv2d):
                return m
        raise RuntimeError("no conv found in first input block")

    @staticmethod
    def _replace_first_conv(input_blocks, new_conv):
        # 简化: 真实 SD UNet 第一个 input_block 通常是单独的 input conv,
        # 这里用搜索-替换示意。生产实现请按具体 UNet 结构精确替换。
        for parent in input_blocks.modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Conv2d) and child.in_channels in (4, 8):
                    setattr(parent, name, new_conv)
                    return

    def forward(self, x_t, lr_img, t, context):
        """
        x_t: (B, 4, h, w) noisy latent
        lr_img: (B, 3, H, W) LR input (RGB)
        t: (B,) time
        context: (B, T, D) text/image tokens
        """
        # 把 LR 处理到 latent 大小
        lr_latent = self.cond_pre(lr_img)

        # ControlNet 的输入 = 噪声 latent + LR latent (concat 成 8 通道)
        h = torch.cat([x_t, lr_latent], dim=1)

        outs = []
        for block, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = block(h, t, context)
            outs.append(zero_conv(h))

        h = self.middle_block(h, t, context)
        outs.append(self.zero_conv_mid(h))

        # 这些 outs 在主 UNet forward 时, 加到对应的 skip 上
        return outs

    @staticmethod
    def _get_block_out_ch(block):
        # 简化, 实际要根据 SD UNet 具体结构推
        for m in block.modules():
            if isinstance(m, nn.Conv2d):
                return m.out_channels
        return None
```

主 UNet 的 forward 要改成把 ControlNet 的输出加到对应 skip：

```python
def forward_with_controlnet(unet, controlnet, x_t, lr_img, t, context):
    """主 UNet forward + ControlNet 注入。"""
    # 1. ControlNet 计算条件信号
    control_outs = controlnet(x_t, lr_img, t, context)

    # 2. 主 UNet encoder, 把 control_outs 加进 skip
    skips = []
    h = x_t
    for i, block in enumerate(unet.input_blocks):
        h = block(h, t, context)
        skips.append(h + control_outs[i])           # 加和

    # 3. 主 UNet mid + control mid
    h = unet.middle_block(h, t, context)
    h = h + control_outs[-1]

    # 4. 主 UNet decoder
    for block, skip in zip(unet.output_blocks, reversed(skips)):
        h = block(torch.cat([h, skip], dim=1), t, context)

    return unet.out(h)
```

### ControlNet 训练数据要求

要 (LR, HR) 配对（HR 用来做 noisy latent 加噪 target，LR 是 ControlNet 输入）。

数据规模：ControlNet 论文用了几十万到几百万张图。增强任务的 ControlNet 通常用第 5 章的合成 pipeline 生成训练数据。

## 9.7 SUPIR（2024）—— 当前 SR SOTA 的设计

SUPIR 把多个工程技巧叠加，达到 2024 年 real-world SR 的 SOTA。值得详细看一下它的组合逻辑。

### 组件 1：SDXL 作为基础

SDXL 是 SD 的更大版本（2.6B 参数 UNet），生成能力比 SD 1.5 强一个数量级。SUPIR 用 SDXL 作为基础保证生成质量。

### 组件 2：ControlNet 注入 LR

类似上面讲的 ControlNet，把 LR 通过 ControlNet 注入。但 SUPIR 用了一个变体——**ZeroSFT**（Zero Spatial Feature Transform）：

ZeroSFT 是一种特征调制——在 ControlNet 输出加到主 UNet 之前，做一个空间相关的仿射变换：

$$
h' = h \odot (1 + \gamma) + \beta
$$

其中 $\gamma, \beta$ 是从 ControlNet 输出预测的空间特征图。这样控制比简单加和更灵活。

### 组件 3：LLaVA prompt

SUPIR 用 LLaVA（一个 VLM）给 LR 自动生成文字描述，作为 SDXL 的文本条件。这让模型有"语义先验"——知道这是猫还是狗，能生成对应的细节。

### 组件 4：自适应噪声

不从纯噪声开始采样，而是从一个**带 LR 信息的噪声**开始：

$$
x_T^{\text{init}} = \sqrt{\bar{\alpha}_T'} \cdot \text{Encode}(y) + \sqrt{1 - \bar{\alpha}_T'} \cdot \epsilon
$$

其中 $T' < T$。这相当于"跳过最高时间步"，从中间开始。优势：减少推理步数 + 保留更多 LR 结构。

### 组件 5：Restoration-Guided Sampling

每一步采样后，用一个 restoration loss（比如 LPIPS to LR-upsampled）把 $\hat{x}_0$ 拉回靠近 LR。这是个推理时的技巧，不需要重新训练。

### SUPIR 综合效果

- 在严重退化的真实老照片上视觉效果远超所有判别式 SR
- LPIPS 比 ESRGAN 低 30%+
- 但 PSNR 比 HAT 低 7-8 dB（perception-distortion trade-off 选了 perception 端）
- 速度慢（30-50 步推理），单张 1K 图需要 5-10 秒（A100）

## 9.8 StableSR（2023）—— 简化版本

Wang et al. 的 StableSR 是 SUPIR 之前的代表，思路简化但工程实践友好。

### 关键设计：CFW（Controllable Feature Warping）

让用户在推理时调节"质量 vs 保真"。具体：在解码 latent 到像素之前，加一层 warping：

$$
\hat{x}_0 = (1 - w) \cdot \text{Decode}(z) + w \cdot \text{Decode}(\text{warp}(z, y))
$$

$w \in [0, 1]$ 是用户可调的参数：

- $w = 0$：纯生成（高质量但低保真）
- $w = 1$：完全保真（接近 LR）
- $w = 0.5$：平衡

这种用户可调设计在生产环境是 plus——同一个模型可以服务不同需求的用户。

### Time-aware Condition

StableSR 还有一个细节：条件注入的强度和时间步相关。早期（高 $t$）注入弱（让模型自由生成），后期（低 $t$）注入强（让模型对齐 LR）。这是对扩散动力学的精确利用。

## 9.9 Tile 推理：处理大图

扩散模型训练时通常在 $256 \times 256$ 或 $512 \times 512$ 的 patch 上。但实际增强任务可能要处理 4K 甚至 8K 图。**直接全图推理会爆显存**——而且模型从没在那么大尺寸上见过，效果可能崩。

解决：**tile-based 推理**。

### 朴素 tile

把大图切成多块，每块独立推理，再拼起来。问题：**块边界不连续**。

### Overlap + blend

让 tile 之间有重叠区域（比如 50%），然后用渐变 mask 融合：

```python
import torch
import torch.nn.functional as F

def tile_diffusion_inference(
    pipeline, hr_img_tensor, tile_size=512, overlap=128, **pipe_kwargs
):
    """
    Tile-based 扩散推理。
    pipeline: 一个 diffusion pipeline
    hr_img_tensor: 输入 LR 图 tensor (1, 3, H, W) - 已经经过 latent 编码或 RGB
    """
    _, _, H, W = hr_img_tensor.shape
    stride = tile_size - overlap

    # 创建累加器和权重
    output = torch.zeros_like(hr_img_tensor)
    weight = torch.zeros_like(hr_img_tensor)

    # 创建渐变权重 mask (中心权重最高, 边缘渐变到 0)
    blend_mask = torch.ones((1, 1, tile_size, tile_size))
    for i in range(overlap):
        v = (i + 1) / (overlap + 1)
        blend_mask[:, :, i, :]  *= v
        blend_mask[:, :, -i-1, :] *= v
        blend_mask[:, :, :, i]  *= v
        blend_mask[:, :, :, -i-1] *= v

    blend_mask = blend_mask.to(hr_img_tensor.device)

    # 滑动窗口推理 - 关键: 用 anchored ranges 保证最后一个 tile 落在 H-tile_size,
    # 否则当 (H - tile_size) 不是 stride 整数倍时, 右/下边缘会缺失覆盖。
    def anchored_starts(total: int, tile: int, step: int):
        if total <= tile:
            return [0]
        starts = list(range(0, total - tile, step))
        if starts[-1] + tile < total:
            starts.append(total - tile)
        return starts

    for top in anchored_starts(H, tile_size, stride):
        for left in anchored_starts(W, tile_size, stride):
            tile = hr_img_tensor[:, :, top:top+tile_size, left:left+tile_size]
            tile_out = pipeline(tile, **pipe_kwargs)

            output[:, :, top:top+tile_size, left:left+tile_size] += tile_out * blend_mask
            weight[:, :, top:top+tile_size, left:left+tile_size] += blend_mask

    return output / (weight + 1e-8)
```

### Shared Noise（共享噪声）

更高级的 trick：所有 tile **共享同一个噪声起点**——把全图的噪声 latent 先生成出来，每个 tile 推理时用对应位置的 noise。这样 tile 之间的"随机性方向"一致，边界更连续。

这是 Multi-Diffusion / SyncDiffusion 等工作的思路。

### ControlNet Tile 模型

专门为 tile 推理训练的 ControlNet 模型——在训练时就用各种 tile 配对训练（包括小尺寸和大尺寸的混合），让模型对 tile 边界更鲁棒。

工程实践：4K+ 图增强**几乎全部用 tile + blend**，没有更好的方案。

## 9.10 Negative Prompt 与质量控制

文生图里 negative prompt 用来排除不想要的（"blurry, low quality, deformed"）。增强任务里也能用：

```python
# 推理时
positive_prompt = "high quality, sharp, detailed photograph"
negative_prompt = "blurry, low quality, jpeg artifacts, oversmooth, plastic skin"
```

通过 CFG 让生成结果**远离** negative prompt 描述的特征。实测影响：

- 不写 negative prompt：~5% 概率出现轻度伪影
- 加合理 negative prompt：伪影概率降到 ~1%

成本：几乎为零（推理时多一次 UNet 前向）。

## 9.11 推理参数调优

扩散增强模型有很多推理参数，对最终效果影响巨大：

| 参数 | 典型范围 | 调高的影响 |
|------|---------|-----------|
| `num_inference_steps` | 20 - 50 | 质量提升、速度变慢 |
| `guidance_scale` | 1.0 - 3.0 | 更"听话"，但可能 oversaturated |
| `controlnet_conditioning_scale` | 0.5 - 1.5 | 更靠 LR，但可能糊 |
| `start_noise_level` | 0.5 - 1.0 | 越小保留更多 LR 结构 |
| `tile_size` | 512, 1024 | 更大块更连贯但显存爆 |
| `tile_overlap` | 64 - 256 | 更平滑但慢 |

工程实践（增强任务的起始配置）：

```python
{
    'num_inference_steps': 25,
    'guidance_scale': 2.0,
    'controlnet_conditioning_scale': 1.0,
    'start_noise_level': 0.7,
    'tile_size': 1024,
    'tile_overlap': 256,
    'positive_prompt': 'high quality, sharp, detailed',
    'negative_prompt': 'blurry, low quality, oversmooth',
}
```

这些参数最好让用户可调——同一个模型不同用户对 fidelity 偏好不同。

## 9.12 训练 vs 推理：关键差异

扩散增强模型的训练和推理差异比 CNN/Transformer 模型大得多。

### 训练时

- 输入大小固定（典型 $512 \times 512$）
- 单步反向（不是迭代采样）
- 不需要采样器，只需 noise scheduler 加噪
- 不需要 tile（直接全图）

### 推理时

- 输入大小可变（任意分辨率）
- 多步迭代采样
- 选择采样器（DDIM/DPM-Solver/UniPC）
- 大图必须 tile
- CFG / negative prompt
- ControlNet conditioning scale 可调

工程含义：**实验时的训练表现和生产时的推理表现可能不一致**——很多问题（tile artifact、CFG 失稳、长序列采样误差累积）只在推理时暴露。这是扩散增强工程的特殊难点。

## 9.13 选型决策表

按场景给推荐：

| 场景 | 推荐范式 | 代表方法 |
|------|---------|---------|
| 严重退化、强生成 | SDXL ControlNet + LLaVA prompt | SUPIR |
| 中等退化、可调节 | SD 1.5 ControlNet + CFW | StableSR |
| 语义保持优先 | CLIP image cross-attn + ControlNet | DiffBIR |
| 内容保持（不变身份） | IP-Adapter + ControlNet | 自定义组合 |
| 快速推理 | LCM 蒸馏 + ControlNet | LCM-LoRA + Tile |
| 端侧 | **不推荐扩散**（用 CNN） | — |
| 4K+ 大图 | ControlNet Tile + Multi-Diffusion | Tile workflow |
| 视频 | 还在研究中 | 第 13 章 |

## 9.14 小结

1. **条件控制是扩散增强的工程核心** —— 比基础扩散重要得多
2. **五种范式**：concat、cross-attention、ControlNet、IP-Adapter、Tile + ControlNet
3. **ControlNet 是事实标准** —— 复制 encoder + zero conv，保留预训练权重
4. **Zero conv 让训练初期模型行为不变**——这是 ControlNet 训练稳定的关键
5. **SUPIR 的多组件叠加**：SDXL + ControlNet + ZeroSFT + LLaVA prompt + 自适应噪声
6. **Tile + Blend 是大图的唯一方案**，用 shared noise + ControlNet Tile 减少边界伪影
7. **推理参数对最终效果影响巨大** —— guidance_scale、conditioning_scale、num_steps、start_noise 都要调
8. **训练和推理差异大**——很多问题只在推理时暴露，必须做生产级测试

到这里 Part II 走过 CNN → Transformer → 扩散基础 → 扩散控制四章。下一章是这部分的最后一章——任务特化模型，讲人脸、文档、医疗等不同任务用了哪些归纳偏置。

---

> 下一章 [任务特化模型](10-task-specific.md) → 通用增强 vs 特化增强：人脸用 GAN inversion、文档用 CRNN 引导、医疗用物理先验。
