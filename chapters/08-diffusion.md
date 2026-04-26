# 第 8 章 · 扩散模型基础

> 这是这本书最核心的范式转变。
>
> 第 6-7 章的所有模型都是**判别式**——给定 $y$ 输出唯一 $\hat{x}$。
> 这一章开始的扩散模型是**生成式**——给定 $y$ 输出 $p(x | y)$，可以采样出多个合理的 $\hat{x}$。
>
> 这个差别在 ill-posed 严重的场景下是质变的。

## 8.1 为什么扩散在影像增强里重要

回到第 4 章 4.8 节的 perception-distortion trade-off：

> ill-posed 问题上 distortion（PSNR）和 perception（FID/LPIPS）不可同时最优。

判别式模型在 distortion 端走到了极致——HAT 在 Set5 4× 上 33.4 dB。但它们在 perception 端有天花板：

- 给定一张糊得厉害的人脸 LR，"最可能"的 $\hat{x}$ 是平均脸（多个合理 $x$ 的均值）
- 判别式模型必须输出**一个**确定的 $\hat{x}$，所以输出平均脸
- 平均脸**视觉上糊**——它在 PSNR 上最优，在感知上不真实

扩散模型直接攻击这个问题：

> 我不输出**一个** $\hat{x}$，我学习 $p(x | y)$ 的整个分布，然后从这个分布**采样**一个具体的 $\hat{x}$。

每个采样出的 $\hat{x}$ 都是分布上的一个具体点——是一张**具体的脸**而不是平均脸，视觉上真实但和真值未必像素级一致。

这就是为什么 SUPIR 在严重退化的老照片上效果惊艳，但 PSNR 比 HAT 低 5+ dB——它走的是 perception-distortion 曲线的另一端。

这一章讲清扩散模型怎么工作、为什么它适合增强任务、以及具体怎么用。

## 8.2 扩散模型的直觉

扩散模型的核心思想可以一句话概括：

> **学一个去噪过程**——从纯噪声开始，一步步去除噪声，最后得到一张图。

具体过程：

```
前向 (固定, 不学习):
  x_0 (干净图)
    ↓ 加一点噪声
  x_1
    ↓ 加一点噪声
  x_2
    ↓ ...
  x_T (纯噪声, T = 1000)

反向 (神经网络学习):
  x_T (纯噪声)
    ↓ 去一点噪声 (用神经网络预测)
  x_{T-1}
    ↓ 去一点噪声
  x_{T-2}
    ↓ ...
  x_0 (干净图)
```

训练目标：**给定任意时间步的加噪图 $x_t$，预测加进去的噪声**。

这看起来很怪——为什么这样能生成图？关键在两点：

1. **任意 $t$ 的加噪图可以一步采样**——不需要从 $x_0$ 一步步加 $T$ 次，有解析公式
2. **学会了预测噪声 = 学会了 $p(x_0)$ 的 score function**——可以从噪声反推回图

下面把数学讲清。

## 8.3 前向过程的数学

定义噪声调度 $\beta_1, \beta_2, \dots, \beta_T$（$T = 1000$, $\beta_t$ 从 $10^{-4}$ 线性增到 $0.02$）。

每一步加噪：

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} \cdot x_{t-1}, \beta_t \mathbf{I})
$$

含义：$x_t$ 是 $x_{t-1}$ 缩小一点（乘 $\sqrt{1-\beta_t}$）再加一点高斯噪声（方差 $\beta_t$）。

定义 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$。**关键性质**：

$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \cdot x_0, (1 - \bar{\alpha}_t) \mathbf{I})
$$

这意味着**给定 $x_0$，可以一步采样到任意 $x_t$**：

$$
x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})
$$

这是训练能高效进行的关键——不需要真的一步步加 1000 次噪声。

```python
import torch


class NoiseScheduler:
    """DDPM 风格的线性 noise schedule。"""

    def __init__(self, num_steps: int = 1000,
                 beta_start: float = 1e-4, beta_end: float = 0.02,
                 device: str = 'cuda'):
        self.num_steps = num_steps
        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # 训练时用到的常用量
        self.sqrt_alphas_cumprod        = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor,
                  noise: torch.Tensor = None) -> tuple:
        """一步采样 x_t。
        x0: (B, C, H, W) 干净图
        t:  (B,) 时间步整数
        noise: 可选, 不传则随机采样
        返回: (x_t, noise)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x_t = sqrt_alpha * x0 + sqrt_one_minus * noise
        return x_t, noise
```

## 8.4 反向过程：训练目标

理论上反向过程是 $p(x_{t-1} | x_t)$，要学的是这个条件分布。但 DDPM 论文证明了一个简化的等价目标：

**直接训一个网络 $\epsilon_\theta(x_t, t)$ 预测加进去的噪声 $\epsilon$**。

训练损失：

$$
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$

这就是第 3 章 3.6 节讲的 simple loss。

### 三种等价预测目标

模型可以预测三种量之一，互相等价：

- **$\epsilon$-prediction**：预测加进去的噪声（DDPM 标准）
- **$x_0$-prediction**：预测原图
- **$v$-prediction**：$v_t = \alpha_t \epsilon - \sigma_t x_0$（更稳定）

它们之间的转换：

```python
# 给定模型输出和当前 (x_t, t), 三种之间互转

def eps_to_x0(x_t, eps_pred, alpha_cumprod_t):
    """从预测的 epsilon 推 x_0。"""
    sqrt_alpha_t = alpha_cumprod_t.sqrt()
    sqrt_one_minus = (1 - alpha_cumprod_t).sqrt()
    return (x_t - sqrt_one_minus * eps_pred) / sqrt_alpha_t


def v_to_x0(x_t, v_pred, alpha_cumprod_t):
    """从预测的 v 推 x_0。"""
    sqrt_alpha_t = alpha_cumprod_t.sqrt()
    sqrt_one_minus = (1 - alpha_cumprod_t).sqrt()
    return sqrt_alpha_t * x_t - sqrt_one_minus * v_pred


def x0_to_eps(x_t, x0_pred, alpha_cumprod_t):
    """从预测的 x_0 推 epsilon。"""
    sqrt_alpha_t = alpha_cumprod_t.sqrt()
    sqrt_one_minus = (1 - alpha_cumprod_t).sqrt()
    return (x_t - sqrt_alpha_t * x0_pred) / sqrt_one_minus
```

训练采样目标的选择：

- **$\epsilon$-pred**：通用文生图标配（SD 1.x）
- **$v$-pred**：高分辨率训练更稳（SD 2.x, SDXL refiner）
- **$x_0$-pred**：增强/恢复任务直观，关心的就是 $x_0$ 质量

## 8.5 一个最小的 DDPM 训练循环

```python
import torch
import torch.nn.functional as F
from torch.optim import AdamW

def train_ddpm(model, train_loader, scheduler, epochs=100,
               lr=1e-4, device='cuda'):
    """最小的 DDPM 训练循环。
    model: UNet, 输入 (x_t, t), 输出预测的噪声 (B, C, H, W)
    """
    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        for x0 in train_loader:                    # x0: (B, C, H, W)
            x0 = x0.to(device)
            B = x0.shape[0]

            # 1. 随机采时间步
            t = torch.randint(0, scheduler.num_steps, (B,), device=device)

            # 2. 一步采样 x_t
            x_t, noise = scheduler.add_noise(x0, t)

            # 3. 模型预测噪声
            pred_noise = model(x_t, t)

            # 4. MSE 损失
            loss = F.mse_loss(pred_noise, noise)

            # 5. 反传
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
```

这个循环看起来过于简单。但它能 work 的根本原因：

> 学习"从带任意噪声的图预测加进去的噪声"= 学习 $\nabla \log p(x)$ 的近似（score function）。
>
> 一旦学到 score，就可以用 Langevin 动力学/反向 SDE 从噪声采样到 $p(x)$。

## 8.6 采样：DDPM, DDIM, DPM-Solver

训练完后怎么从噪声生成图？

### DDPM 采样

按反向链一步步：

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z
$$

其中 $z \sim \mathcal{N}(0, \mathbf{I})$，$\sigma_t$ 是噪声项。

**问题**：要走 1000 步，每步一次 UNet 前向，**慢**。

### DDIM 采样

Song et al. 发现可以用确定性反向：

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1}} \cdot \epsilon_\theta(x_t, t)
$$

其中 $\hat{x}_0$ 是从 $x_t$ 和预测噪声反推的 $x_0$。这个采样**可以跳步**——不需要走完 1000 步，可以挑 50 个时间步采样。

DDIM 50 步 ≈ DDPM 1000 步质量，**20× 加速**。

### DPM-Solver 系列

把反向过程看作 ODE，用高阶数值方法求解。

- **DPM-Solver-2**：二阶，约 20 步达到 DDIM 100 步质量
- **DPM-Solver++**：处理 SDE 形式
- **UniPC**：进一步优化的统一预测-校正方法

```python
# 调用 diffusers 的采样器
from diffusers import DPMSolverMultistepScheduler

scheduler = DPMSolverMultistepScheduler.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    subfolder="scheduler",
    algorithm_type="dpmsolver++",
    solver_order=2,
)
scheduler.set_timesteps(num_inference_steps=20)
```

工程实践（2026 年的 SOTA 配置）：

- **学术 benchmark / 高质量推理**：UniPC，30-50 步
- **生产部署**：DPM-Solver++ 2M，20 步
- **极致速度**：LCM (Latent Consistency Models) 蒸馏后，4 步

## 8.7 LDM：扩散搬到潜空间

第 2 章 2.4 节已经介绍过。这里补充工程细节。

LDM (Rombach et al. 2022) 的关键观察：

> 像素空间的扩散在大量计算"高频细节"上浪费——这些细节可以用一个 VAE 解码器后处理生成。
>
> 让 UNet 只在**潜空间**做扩散，把高频细节交给 VAE 解码器。

### Stable Diffusion 的具体配置

| 组件 | 作用 | 参数量 |
|------|------|--------|
| VAE encoder | RGB → latent (8× 下采样, 4 通道) | ~50M |
| VAE decoder | latent → RGB | ~50M |
| UNet | 潜空间扩散 | ~860M |
| Text encoder (CLIP) | 文本 → embedding | ~120M |
| 总计 | | ~1.1B |

UNet 是大头，VAE 相对小。这就是为什么 SD finetune 主要训 UNet（保持 VAE 不变）。

### LDM 的工程影响

- **计算量降到 1/48**：原本 $512 \times 512 \times 3$ 的扩散，现在在 $64 \times 64 \times 4$ 上
- **可以处理大图**：$1024 \times 1024$ 不再是显存灾难
- **VAE 是个独立可换的组件**：可以微调 VAE（比如增强场景下用更好的 decoder）

## 8.8 SD UNet 内部结构

UNet 是扩散模型的"主网络"。它的结构对增强任务很重要——后面 ControlNet 等扩展都建立在这个结构上。

```
Input (B, 4, 64, 64) latent
  ↓ Conv (4 → 320 channels)
  ↓ DownBlock × 4 (encoder)
    每个 DownBlock:
      ResBlock × 2 + Spatial Transformer × 2 (cross-attention to text)
      Downsample (×2)
  ↓ MidBlock
    ResBlock + Spatial Transformer + ResBlock
  ↓ UpBlock × 4 (decoder, 带 skip connection)
    每个 UpBlock:
      ResBlock × 3 + Spatial Transformer × 3
      Upsample (×2)
  ↓ Conv (320 → 4 channels)
Output (B, 4, 64, 64) 预测的噪声/v
```

### ResBlock：核心计算单元

```python
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """SD UNet 的 ResBlock, 带时间嵌入注入。"""

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
        # 注入时间嵌入 (broadcast)
        h = h + self.time_proj(F.silu(t_emb)).view(*t_emb.shape[:1], -1, 1, 1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)
```

时间嵌入注入是扩散模型独有的——同一个网络要处理 1000 个不同时间步，需要知道当前是哪一步。

### Spatial Transformer：cross-attention 入口

```python
class SpatialTransformer(nn.Module):
    """SD UNet 的 attention block。
    self-attention + cross-attention to text。
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

cross-attention 是文生图的关键——文本通过 cross-attention 影响每个空间位置的特征。这个机制在第 9 章会被复用到增强任务的条件控制。

## 8.9 增强任务里的扩散用法

Text-to-image 是从纯噪声 + 文本条件采样到图。**增强任务**是从纯噪声 + 退化图 $y$ 条件采样到 $\hat{x}$。

几种条件注入范式：

### 范式一：concat 到输入（最简单）

```python
# UNet 输入从 (B, 4, h, w) 变成 (B, 8, h, w)
# 后 4 通道是 LR 的 latent
def prepare_input(noisy_latent, lr_latent):
    return torch.cat([noisy_latent, lr_latent], dim=1)
```

代表：StableSR (2023) 用这种简单形式。

### 范式二：cross-attention（语义条件）

把 LR 的某种 embedding 通过 cross-attention 注入。代表：DiffBIR 用 CLIP image embedding。

### 范式三：ControlNet（结构条件）

复制一份 UNet encoder，专门处理条件输入，输出加到主 UNet 的对应层。代表：StableSR 的 ControlNet 变体、SUPIR。

第 9 章会详谈。

## 8.10 扩散为什么能"无中生有"

回到这一章开头的核心问题：**扩散为什么能生成出真实的高频细节，而判别式不能？**

### 1. 多步采样 = 多次 random refinement

判别式：一次前向，输出确定的 $\hat{x}$。
扩散：$T$ 次前向，每一次都从一个随机分布采样。每一次采样都是"这一步该往哪个方向走"的随机选择。

这个随机性意味着同样的 LR 输入可以生成不同的 $\hat{x}$——每个都是合理的。

### 2. 学到了 score function

理论上扩散模型学到的是：

$$
\epsilon_\theta(x_t, t) \approx -\sqrt{1-\bar{\alpha}_t} \cdot \nabla_{x_t} \log p(x_t)
$$

这个梯度场告诉模型在 $x_t$ 这一点应该往哪个方向走才能更接近自然图像分布。score function 直接编码了**自然图像流形的几何结构**。

### 3. 反向链是 SDE 求解器

可以严格证明 DDPM 的反向链等价于求解一个 SDE（随机微分方程）：

$$
dx = [f(x, t) - g(t)^2 \nabla_x \log p_t(x)] dt + g(t) dw
$$

其中 $w$ 是 Wiener 过程（布朗运动）。这个 SDE 的稳态分布就是自然图像分布。

### 这三个性质合起来

扩散模型不是在"逼近真值"，而是在"沿着自然图像分布的几何结构走 T 步"。每一步都被分布的几何引导，每一步都有随机扰动让生成结果不重复。

最终输出是分布上的一个点——具体的、合理的、随机的。

## 8.11 扩散派 vs 判别式 trade-off

把 Part II 前两章和这一章对比起来：

| 方面 | 判别式 (CNN/Transformer) | 扩散 |
|------|------------------------|------|
| 推理速度 | **1×（baseline）** | 20-1000× 慢 |
| PSNR | **高** | 低 5-10 dB |
| LPIPS / FID | 低 | **更好** |
| 视觉真实感 | 中 | **高** |
| 对严重退化的处理 | 输出"安全模糊" | **能恢复细节** |
| 多样性 | 唯一输出 | **可采样多个** |
| 控制 | 只能改训练数据 | **可加 prompt / control** |
| 显存占用 | 低 | 高 (2-4×) |

**没有谁全胜**——这两类模型在不同应用场景各有优势：

- **法医证据增强**：判别式（不允许编造）
- **学术 benchmark PSNR 比赛**：判别式
- **照片放大、老照片修复**：扩散
- **极轻量端侧部署**：判别式
- **创意性强的应用**（艺术风格、4K 直播创意增强）：扩散

## 8.12 训练扩散模型的工程细节

### Min-SNR 加权（第 3 章 3.6 节提过）

```python
def min_snr_weight(t, alphas_cumprod, gamma=5.0):
    """Min-SNR 加权, 让不同时间步的 loss 量级一致。"""
    snr = alphas_cumprod[t] / (1 - alphas_cumprod[t])
    return torch.minimum(snr, torch.full_like(snr, gamma)) / snr
```

### EMA（指数移动平均）

扩散模型的最终权重通常是 EMA 而不是 raw weights。EMA 衰减常用 0.9999：

```python
class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {n: p.clone().detach() for n, p in model.named_parameters()}

    def update(self, model):
        for n, p in model.named_parameters():
            self.shadow[n] = self.decay * self.shadow[n] + (1 - self.decay) * p.detach()

    def apply_to(self, model):
        """把 EMA 权重 copy 到 model（用于 inference）。"""
        for n, p in model.named_parameters():
            p.data.copy_(self.shadow[n])
```

### 时间步 importance 采样

不要 uniformly 采样 $t$。某些 $t$ 范围对最终质量贡献更大（通常是中间区域 $t \in [200, 800]$），可以提高这些区域的采样概率。

### Classifier-Free Guidance (CFG)

训练时 10% 概率丢弃条件（无条件训练 + 有条件训练混合）。推理时：

$$
\hat{\epsilon} = \epsilon_\theta(x_t, t, \emptyset) + w \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset))
$$

$w > 1$ 让生成更靠近条件，但太大会过拟合。增强任务里 $w$ 通常 1.5-3.0。

```python
def classifier_free_guidance(model, x_t, t, condition, guidance_scale=2.0):
    """CFG 推理。"""
    # 有条件预测
    eps_cond = model(x_t, t, condition)
    # 无条件预测 (用 null/empty condition)
    eps_uncond = model(x_t, t, None)
    # 加权
    return eps_uncond + guidance_scale * (eps_cond - eps_uncond)
```

## 8.13 一个增强任务的扩散训练流程

把上面的概念串成一个增强任务的训练循环。这是 SUPIR/StableSR 的简化版：

```python
def train_diffusion_enhancement(
    unet, vae, scheduler,
    train_loader,                  # 输出 (lr, hr) 配对
    text_encoder=None,             # 可选, 加 prompt 条件
    epochs=50, lr=1e-4,
    device='cuda',
):
    optimizer = AdamW(unet.parameters(), lr=lr)
    ema = EMA(unet)

    # 冻结 VAE 和 text encoder
    vae.eval(); 
    for p in vae.parameters():
        p.requires_grad_(False)

    for epoch in range(epochs):
        for lr_img, hr_img in train_loader:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            # 1. 编码到潜空间
            with torch.no_grad():
                hr_latent = vae.encode(hr_img).latent_dist.sample() * 0.18215
                lr_latent = vae.encode(lr_img).latent_dist.mean * 0.18215

            # 2. 加噪到 hr_latent
            B = hr_latent.shape[0]
            t = torch.randint(0, scheduler.num_steps, (B,), device=device)
            x_t, noise = scheduler.add_noise(hr_latent, t)

            # 3. UNet 输入 = (x_t || lr_latent), 条件 = lr_latent (或 text)
            unet_input = torch.cat([x_t, lr_latent], dim=1)
            pred_noise = unet(unet_input, t)

            # 4. 加权 MSE 损失
            weights = min_snr_weight(t, scheduler.alphas_cumprod)
            loss = (weights.view(-1, 1, 1, 1) * (pred_noise - noise) ** 2).mean()

            # 5. 反传
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()

            # 6. EMA 更新
            ema.update(unet)

    # 训练完后用 EMA 权重做推理
    ema.apply_to(unet)
    return unet
```

## 8.14 推理：从 LR 到 HR 的完整流程

```python
@torch.no_grad()
def diffusion_enhance(unet, vae, scheduler, lr_img,
                       num_inference_steps=20, guidance_scale=2.0):
    """
    给定 LR 图, 生成 HR 估计。
    """
    device = lr_img.device

    # 1. LR 编码到 latent
    lr_latent = vae.encode(lr_img).latent_dist.mean * 0.18215

    # 2. 从纯噪声开始 (latent 大小 = LR latent 大小)
    x_t = torch.randn_like(lr_latent)

    # 3. 设置 inference 时间步 (DPM-Solver / DDIM)
    scheduler.set_timesteps(num_inference_steps)

    # 4. 反向去噪
    for t in scheduler.timesteps:
        unet_input = torch.cat([x_t, lr_latent], dim=1)
        pred_noise = unet(unet_input, t)
        # CFG (省略, 实际生产用)
        x_t = scheduler.step(pred_noise, t, x_t).prev_sample

    # 5. 解码到像素
    hr_img = vae.decode(x_t / 0.18215).sample
    return hr_img.clamp(0, 1)
```

注意输出是 latent 解码后的图，分辨率取决于 VAE 配置（通常 LR 的 8× = 上采样 8×）。如果要 4× SR，可以让 LR 一开始就是 2× 上采样过的，最终输出就是 4×。

## 8.15 小结

1. **扩散是从判别式到生成式的范式转变**——输出不是唯一 $\hat{x}$，是 $p(x|y)$ 的采样
2. **前向加噪 + 反向去噪**：训练目标是预测加进去的噪声
3. **任意时间步可以一步采样**：不需要 1000 次加噪，有解析公式
4. **三种预测目标等价**（$\epsilon$ / $x_0$ / $v$），选哪个看任务
5. **DDPM 1000 步 → DDIM 50 步 → DPM-Solver 20 步 → LCM 4 步**：采样器演进
6. **LDM 是工程关键**：把扩散搬到 VAE 潜空间，计算降到 1/48
7. **Stable Diffusion = LDM + text condition + 大规模训练**
8. **增强任务的扩散用法**：把 LR 作为条件注入 UNet
9. **扩散的"无中生有"= 学到 score function + 多步随机采样**
10. **trade-off**：扩散视觉真实感强但 PSNR 低、慢、显存高

下一章讲条件控制——把 LR 注入扩散 UNet 有几种范式（concat、cross-attention、ControlNet、IP-Adapter），各自适合什么场景。

---

> 下一章 [扩散的条件控制](09-control.md) → 从 concat 到 ControlNet 到 IP-Adapter，扩散增强模型的真正工程战场。
