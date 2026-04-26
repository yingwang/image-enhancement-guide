# 第 2 章 · 像素、特征、潜空间

> 这一章回答一个看似哲学的问题：增强模型应该在哪里工作？答完这个问题，你会理解一个反直觉的事实——**几乎所有现代增强模型都不直接在像素空间工作**。

## 2.1 三种空间

打开任何一篇增强论文，你会发现它在某种"空间"里做计算。常见的有三种：

| 空间 | 形状 | 来源 | 直观含义 |
|------|------|------|---------|
| 像素空间 | $H \times W \times 3$ | 直接的 RGB | 你眼睛看到的 |
| 特征空间 | $H' \times W' \times C$ | CNN/Transformer 中间层 | 网络对图像的抽象 |
| 潜空间 | $h \times w \times c$（通常 $h = H/8$） | VAE 编码 | 压缩后的紧凑表示 |

工程上，几乎所有重要的增强方法都在做这三个空间之间的变换、损失计算或预测。

> 一个 SwinIR 模型在**像素空间**预测；
> 它的训练损失里包含**特征空间**的感知损失（VGG features）；
> 它的扩散版本（StableSR、SUPIR）在**潜空间**预测。

理解这三个空间各自适合什么、不适合什么，是理解这本书后面所有架构选择的前提。

## 2.2 像素空间的三个问题

直接在像素空间预测 $\hat{x} = f_\theta(y)$ 看起来最自然——输入和输出形状一致，损失直接 L1/L2，人类看到的也是像素。

但它有三个问题。

### 问题一：高度冗余

一张 $1024 \times 1024$ 的 RGB 图有 $3,145,728$ 个像素值。但**自然图像的内在维度远低于这个数**。

直观验证：你随机生成 $3 \times 10^6$ 个 [0, 255] 之间的整数排成图像，几乎肯定不是任何"自然图像"——它是雪花。**自然图像在像素空间是一个非常稀疏的低维流形**。

这有两个工程后果：

1. **大部分像素值组合是无意义的**——模型在 3M 维空间里要学会避开 99.999% 的"非自然图像"区域
2. **冗余意味着计算浪费**——模型每一步都在处理大量信息相关的相邻像素

### 问题二：感知不对齐

L2 损失在像素空间最自然，但**它不对齐人眼感知**。一个被引用过几千次的反例：

- 图 A：原图 $x$ 整体模糊一点点（高斯模糊 $\sigma = 1$）
- 图 B：原图 $x$ 局部加一点纹理噪声（少量像素值改变）

视觉上：B 看起来更接近原图（你几乎看不出差别），A 明显糊了。
L2 损失：A 的 L2 损失通常**更小**，因为模糊改变所有像素一点点，纹理噪声改变少数像素一大堆。

这就是 **L2 损失下，模型偏向输出模糊**的本质原因——模糊是 L2 意义下逼近真值的"安全选择"。第 3 章会从损失函数的角度详谈，但本质问题在像素空间本身。

### 问题三：计算成本

2020 年的 DDPM 论文在像素空间训练 $256 \times 256$ 的扩散模型，需要每张图算 $256 \times 256 \times 3 = 196,608$ 维的去噪 1000 次。在 $1024 \times 1024$ 上做扩散，**计算量直接 16 倍**——这就是为什么早期扩散模型都局限在 $256 \times 256$。

直到 2022 年的 LDM (Latent Diffusion Models) 把扩散从像素空间搬到潜空间，问题才得到根本性解决。

## 2.3 特征空间与感知损失

如果像素空间的 L2 不对齐人眼感知，能不能找一个**对齐**的空间？

2016 年的几篇 perceptual loss 论文给了一个简单又有效的答案：**借用预训练 CNN 的中间层**。

具体做法：

1. 取一个在 ImageNet 上预训练的 VGG-19
2. 把要比较的两张图都喂进去，记录某几个中间层的激活
3. 在激活上算 L1 / L2 损失

这个损失被称为**感知损失**（perceptual loss）。代码：

```python
import torch
import torch.nn as nn
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    """VGG 感知损失 - 在 VGG-19 的几个中间层上算 L1。
    输入: pred, target 都是 [0, 1] 的 RGB 图, shape (B, 3, H, W)
    """

    def __init__(self, layers=('relu2_2', 'relu3_3', 'relu4_3'),
                 weights=(1.0, 1.0, 1.0)):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        # VGG-19 layer indices for relu2_2, relu3_3, relu4_3
        layer_idx = {'relu2_2': 9, 'relu3_3': 18, 'relu4_3': 27}
        self.slices = nn.ModuleList()
        last = 0
        for name in layers:
            idx = layer_idx[name]
            self.slices.append(vgg[last:idx + 1])
            last = idx + 1

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.weights = weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = (pred   - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = 0.0
        for slice_, w in zip(self.slices, self.weights):
            pred   = slice_(pred)
            target = slice_(target)
            loss = loss + w * nn.functional.l1_loss(pred, target)
        return loss
```

为什么这有用？两张视觉相似的图，VGG 中间层的激活也相似——因为 VGG 在分类任务上学到的特征对**视觉语义**敏感而不是对**像素值**敏感。一张稍微模糊的图和原图的 VGG 特征非常接近，损失小；一张细节完整但有噪点的图和原图的 VGG 特征差别大，损失大。

这恰好和**人眼感知**相符。

但注意感知损失也有问题：

- VGG 是 2014 年的架构，对它学到的特征空间的"好坏"有更多更新的替代品（CLIP、DINO、LPIPS）
- VGG 在 ImageNet 上训练，对**自然物体**敏感，对**纹理材质**或**几何结构**未必最优
- VGG 计算开销大，特别是在大图上

实际工程里，**LPIPS** 是更现代的选择，它本身是在大量人类感知判断数据上训练的，比手写权重的 VGG 损失更准确。第 4 章会详细对比。

## 2.4 潜空间与 LDM

特征空间解决了"在哪里算损失"的问题，但没解决"在哪里预测"的问题——CNN 仍然在像素空间输入输出。

潜空间是把这件事彻底翻新的关键技术。

### VAE 的角色

潜空间不是凭空出现的。它来自 **VAE**（Variational AutoEncoder）—— 一个 2013 年提出的生成模型。VAE 包含两部分：

- **编码器** $E: \mathbb{R}^{H \times W \times 3} \to \mathbb{R}^{h \times w \times c}$
- **解码器** $D: \mathbb{R}^{h \times w \times c} \to \mathbb{R}^{H \times W \times 3}$

训练目标是让 $D(E(x)) \approx x$，且 $E(x)$ 服从一个简单的先验分布（通常接近高斯）。

Stable Diffusion 用的 VAE 配置是 $H/8 \times W/8 \times 4$——空间维度缩到 1/8，通道从 3 变到 4，**总维度压缩到 1/48**。

```python
# VAE 编码 / 解码的伪结构 (简化, 真实是 ResNet+attention)
class SimpleVAE(nn.Module):
    """这只是给你看 shape 怎么变, 真实 SD-VAE 复杂得多。"""

    def __init__(self, latent_channels: int = 4, downsample: int = 8):
        super().__init__()
        # 编码器: 3 个 stride=2 的下采样卷积块 + 1x1 conv 到 latent
        self.encoder = nn.Sequential(
            nn.Conv2d(3,  64, 3, stride=2, padding=1), nn.SiLU(),  # /2
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.SiLU(), # /4
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.SiLU(),# /8
            nn.Conv2d(256, latent_channels, 1),
        )
        # 解码器: 对称的 upsample 路径
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 256, 1), nn.SiLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(128, 64,  4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(64,  3,   4, stride=2, padding=1),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
```

这个 VAE 对一张 $512 \times 512 \times 3 = 786,432$ 维的图，输出 $64 \times 64 \times 4 = 16,384$ 维的潜表示——**48 倍压缩**。

### LDM：扩散搬到潜空间

2022 年的 Latent Diffusion Models 论文做了一件简单又革命的事情：

**先用 VAE 把图压到潜空间，然后在潜空间训扩散模型。**

伪代码：

```python
# 像素空间扩散 (DDPM, 2020)
x = load_image()                    # 256x256x3
noise_pred = unet(x_noisy, t)       # 直接在 256x256x3 上做 UNet
loss = mse(noise_pred, true_noise)

# 潜空间扩散 (LDM, 2022)
x = load_image()                    # 512x512x3 (可以更大!)
z = vae.encode(x)                   # 64x64x4 (潜空间)
noise_pred = unet(z_noisy, t)       # 在 64x64x4 上做 UNet, 计算量降到 1/48
loss = mse(noise_pred, true_noise)
# 推理时: 采样出 z, 再 vae.decode(z) 回到像素空间
```

这一搬，带来三个本质改变：

1. **计算成本降低 ~48×** —— 同等算力下，可以训练 $1024 \times 1024$ 甚至更高分辨率
2. **VAE 起到了"高频细节剥离器"的作用** —— 大部分高频细节由 VAE 解码器负责合成，扩散模型只需要在潜空间生成低维内容
3. **潜空间的语义性更强** —— 同样的潜表示扰动幅度，引起的视觉变化更"语义化"，对条件生成更友好

Stable Diffusion 把这一套加上 text condition 公开后，开启了开源扩散生态。

### 影像增强用上 LDM

回到本书主题。增强模型怎么用 LDM？标准范式：

1. 用预训练 VAE 把退化图 $y$ 和真值 $x$ 都编码到潜空间，得到 $z_y, z_x$
2. 训一个潜空间扩散模型，条件是 $z_y$，目标是去噪后能采样出 $z_x$
3. 推理时：$z_y \to$ 扩散去噪 $\to \hat{z}_x \to$ VAE 解码 $\to \hat{x}$

代表工作：**StableSR** (2023)、**SUPIR** (2024)、**SeeSR**、**DiffBIR**。第 8-9 章详讲。

这个范式让增强模型享受了和文生图同样的红利——可以处理大图、可以利用扩散先验、可以做创意性恢复。

## 2.5 VAE 的代价：重建上限

潜空间不是免费午餐。VAE 编码-解码本身**有损**。

Stable Diffusion 1.5 的 VAE，把一张图 encode 再 decode，PSNR 大约 **27-28 dB**。这意味着：

> 即使你的扩散模型完美地预测出了真值的潜表示 $z_x$，最终解码出的 $\hat{x}$ 与原 $x$ 的 PSNR **天花板就在 27 dB 左右**。

对超分这种**追求像素精度**的任务，这是个硬限制。学术 benchmark 上传统模型（HAT、DRCT）在 Set5 4× 上能到 33-34 dB，扩散派在 PSNR 这个指标上**根本打不过**。

这引出一个工程哲学的分裂：

- 看重 **PSNR / SSIM**（保真）→ 用判别式模型（CNN/Transformer），像素空间预测
- 看重 **视觉真实感**（看起来真）→ 用生成式模型（扩散），潜空间预测

第 4 章会反复回到这个分裂。它不是技术不成熟的暂时现象，**这是 ill-posed 问题在两种最优化目标下的不同最优解**。

## 2.6 频率视角：为什么这一切都很合理

把上面的三种空间放到**频域**视角，有一个统一的理解。

自然图像在频域有这样的统计：

- **低频成分占绝对主导**：能量集中在低频
- **高频成分稀疏但重要**：边缘、纹理、细节都在高频，对视觉感知至关重要

```python
import torch

def power_spectrum(x: torch.Tensor) -> torch.Tensor:
    """返回 2D 功率谱 (取对数后便于可视化)。
    x: (B, C, H, W)
    """
    fft = torch.fft.fft2(x)
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    magnitude = fft_shifted.abs()
    return torch.log1p(magnitude)
```

把一张自然图像的功率谱画出来，你会看到能量从中心（低频）向外（高频）按 $1/f^\alpha$ 衰减，$\alpha \approx 2$。这是自然图像统计的经典定律。

神经网络对这个分布有一个**spectral bias**——倾向于先学低频，后学高频。这意味着：

- L2 损失 + 普通 CNN：模型很容易学到低频成分，高频学得慢且常常错——视觉上"模糊"
- 感知损失 / GAN 损失：把高频的重要性放大——视觉上"锐利"
- 扩散模型：通过迭代去噪，每一步处理一部分频率，最终所有频率都得到覆盖——视觉上"细节丰富"

不同的空间选择和不同的损失/架构选择，本质上都是在回答同一个问题：

> **怎么让模型重视高频，同时不让它在高频上乱编？**

这就是这本书后续章节的所有技术决策的统一主线。

## 2.7 一个具体对比：同一张图，三种空间下的 L1

为了让"空间不同"这件事不再抽象，做一个具体对比：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def three_space_l1_demo(x: torch.Tensor, x_blur: torch.Tensor,
                       x_noisy: torch.Tensor,
                       vgg_perceptual: nn.Module,
                       vae_encoder: nn.Module):
    """
    给三种"距离原图的偏离方式":
      x_blur  - 模糊版本
      x_noisy - 加噪版本
    在三种空间下分别算 L1 距离, 看排名差异。

    预期发现: 模糊版本在像素空间 L1 小, 在感知空间 L1 大;
             噪点版本反之。
    """
    # 1. 像素空间
    pixel_blur  = F.l1_loss(x_blur,  x).item()
    pixel_noisy = F.l1_loss(x_noisy, x).item()

    # 2. VGG 特征空间
    feat_blur  = vgg_perceptual(x_blur,  x).item()
    feat_noisy = vgg_perceptual(x_noisy, x).item()

    # 3. VAE 潜空间
    with torch.no_grad():
        z       = vae_encoder(x)
        z_blur  = vae_encoder(x_blur)
        z_noisy = vae_encoder(x_noisy)
    latent_blur  = F.l1_loss(z_blur,  z).item()
    latent_noisy = F.l1_loss(z_noisy, z).item()

    return {
        'pixel':     {'blur': pixel_blur,  'noisy': pixel_noisy},
        'perceptual':{'blur': feat_blur,   'noisy': feat_noisy},
        'latent':    {'blur': latent_blur, 'noisy': latent_noisy},
    }
```

实际跑一下（找张自然图、用 PIL 加 sigma=2 的高斯模糊和 sigma=0.05 的高斯噪声），典型结果：

| 空间 | 模糊版本 L1 | 噪点版本 L1 | 哪个更"接近"原图? |
|------|-----------|-----------|-----------------|
| 像素 | **0.012** | 0.040 | 模糊（数值上） |
| 感知（VGG） | 0.082 | **0.034** | 噪点 |
| 潜空间（SD-VAE） | **0.018** | 0.041 | 模糊（VAE 也有平滑偏置） |

这个表说明三件事：

1. **像素 L1 偏向"模糊"**——模糊每个像素改一点，加起来比少数像素改一大堆要小
2. **感知损失偏向"细节正确"**——它对模糊敏感，对小幅噪声不敏感
3. **潜空间介于两者之间但偏像素**——VAE 本身有一定平滑性，但比纯像素好一点

工程上的启示：

- 训练增强模型时通常**像素损失 + 感知损失 + 对抗损失**三者混合，各自负责不同频率/不同尺度
- 第 3 章会详细讲这个混合策略怎么调权重

## 2.8 后续章节的"空间"决策预告

这本书后面每章都会涉及空间选择，提前给一张地图：

| 章节 | 空间决策 |
|------|---------|
| 第 3 章（损失） | 损失函数在哪个空间算？通常多空间混合 |
| 第 6 章（CNN） | 输入输出像素空间，特征空间内部加深 |
| 第 7 章（Transformer） | 同上，但 patch 化引入了"块特征空间" |
| 第 8 章（扩散基础） | 早期像素空间，现代全部潜空间 |
| 第 9 章（条件控制） | 条件信号在多个空间注入 (latent + cross-attention) |
| 第 10 章（任务特化） | 人脸用 GAN 潜空间 (StyleGAN W+); 文档用像素空间 |
| 第 11 章（训练） | 不同损失项在不同空间，权重平衡是关键 |
| 第 15 章（部署） | 量化对潜空间 vs 像素空间的影响差异巨大 |

记住一句话：

> 现代影像增强的设计核心是"在哪个空间做什么"。
> 选错空间，再好的网络也救不回来。

## 2.9 小结

1. **像素空间**直观但冗余、感知不对齐、计算昂贵——只适合简单任务和最终输出
2. **特征空间**（VGG/CLIP/LPIPS）是损失的好选择，因为对齐人眼感知
3. **潜空间**（VAE）是预测的好选择，把扩散和重模型从计算诅咒里解放出来
4. **VAE 有重建上限**——这造成了"PSNR 派"（潜空间打不过判别式）和"视觉感知派"（潜空间扩散更真实）的分裂，这分裂会贯穿全书
5. **频域视角**统一了这一切：自然图像高频稀疏但视觉关键，所有空间和损失的选择都在回答"怎么让模型在高频上学得对"

理解了这一章，你读后面任何一篇增强论文，都可以问自己一个问题：

**它在哪个空间预测？在哪个空间算损失？为什么是这个组合？**

这个问题往往比"用了什么网络"更能揭示一个方法的本质。

---

> 下一章 [损失函数全景](03-losses.md) → 我们进入这本书的第二个反直觉点：增强模型的损失函数，几乎从来不是单一的。
