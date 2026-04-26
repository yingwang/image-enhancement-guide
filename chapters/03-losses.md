# 第 3 章 · 损失函数全景

> 增强模型的损失，几乎从来不是单一的。理解为什么——以及怎么混合——是这一章要回答的问题。

## 3.1 为什么需要这一章

LLM 训练里损失函数基本是一个：next-token cross-entropy。分类任务里也基本是一个：cross-entropy。

影像增强不一样。打开任何一篇主流增强论文，损失函数那一节通常长这样：

$$
\mathcal{L} = \lambda_1 \mathcal{L}_{\text{pixel}} + \lambda_2 \mathcal{L}_{\text{perceptual}} + \lambda_3 \mathcal{L}_{\text{adversarial}} + \lambda_4 \mathcal{L}_{\text{...}}
$$

四五个损失项加权混合。为什么？因为第 1 章和第 2 章已经回答过：

- ill-posed 问题需要先验，**每个损失项就是一种先验的编码方式**
- 不同损失偏好不同的频率成分（L2 偏低频，对抗偏高频，感知偏中高频）
- 单一损失会过度优化某个维度（L2 → 模糊，纯对抗 → 假细节，纯感知 → 颜色漂移）

> 增强模型的训练，不是"让模型逼近真值"，而是"在多个相互制衡的指标上找一个 Pareto 最优点"。
>
> 损失函数加权 = 你告诉模型，这些指标之间你最关心哪一个。

这一章把工程上常用的损失项分门别类讲清楚，最后给一张混合策略的决策表。

## 3.2 像素空间损失

最基础的一类。直接在像素差上算。

### L2（MSE）—— 数学好用，工程不好用

$$
\mathcal{L}_2 = \frac{1}{N} \sum_i (\hat{x}_i - x_i)^2
$$

数学性质：

- 对应高斯噪声假设的最大似然
- 处处可导，凸
- 直接对应 PSNR 指标

工程问题：

- **对异常值敏感**——一个错得很离谱的像素贡献 $error^2$，会主导整个梯度
- **偏向输出均值**——给定 $y$，最小化 $\mathbb{E}[||\hat{x} - x||^2]$ 的解是 $\mathbb{E}[x | y]$，多个合理 $x$ 的均值视觉上是模糊的
- **不对齐感知**——第 2 章 2.7 节已经验证过

实际工程里 L2 现在基本只用在两个地方：

1. **扩散模型的去噪损失**（数学上必须 L2，因为 score matching 推导出来就是这个）
2. **早期消融实验**（作为基线）

### L1（MAE）—— 现代默认

$$
\mathcal{L}_1 = \frac{1}{N} \sum_i |\hat{x}_i - x_i|
$$

性质：

- 对应拉普拉斯噪声假设的最大似然
- 在 0 处不可导（反正实际中梯度是 sign，工程上不是问题）
- **对异常值更鲁棒**——错得离谱的像素贡献 $|error|$，不会主导梯度

视觉效果：L1 训出来的图比 L2 训出来的更锐利一点。原因：L1 不会被均值"拉平"——给定 $y$，最小化 $\mathbb{E}[||\hat{x} - x||_1]$ 的解是中位数 $\text{median}(x | y)$，比均值更倾向于"某一个具体的合理 $x$"。

**几乎所有非扩散的现代增强模型都用 L1（或 Charbonnier）作为像素损失基础**。

### Charbonnier —— L1 的平滑版

$$
\mathcal{L}_{\text{Charb}} = \frac{1}{N} \sum_i \sqrt{(\hat{x}_i - x_i)^2 + \epsilon^2}
$$

$\epsilon$ 通常取 $10^{-3}$ 或 $10^{-6}$。当 $|error| \gg \epsilon$ 时它退化为 L1，当 $|error| \approx 0$ 时它退化为 L2。

为什么用它而不是直接用 L1？

- L1 在 0 附近梯度是 $\pm 1$ 的阶跃，在已经接近真值时**梯度不衰减**，造成训练后期震荡
- Charbonnier 在 0 附近平滑，已经接近真值时梯度自动变小，训练更稳

```python
import torch

def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor,
                     eps: float = 1e-3) -> torch.Tensor:
    """Charbonnier 损失 (smooth L1 的连续版本)。
    比 L1 数值稳定, 比 L2 鲁棒, 是低层视觉的事实标准。
    """
    diff = pred - target
    return torch.sqrt(diff * diff + eps * eps).mean()
```

工程经验：Restormer、NAFNet、SwinIR 等几乎都在论文里写"L1 损失"，但实际代码里很多用 Charbonnier，因为后者训练更稳。

## 3.3 平滑性 / 梯度损失

像素损失只看每个点对每个点。**梯度损失**看相邻点的差。

### Total Variation

$$
\mathcal{L}_{\text{TV}} = \sum_{i,j} \left( |\hat{x}_{i+1,j} - \hat{x}_{i,j}| + |\hat{x}_{i,j+1} - \hat{x}_{i,j}| \right)
$$

它惩罚相邻像素的差，鼓励**分段平滑**——内部平、边缘锐利。

```python
def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    """TV 损失 (各向异性版本)。
    x: (B, C, H, W)
    """
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw
```

何时用：

- **去噪场景**：TV 是经典先验，能抑制噪点
- **生成模型**：扩散/GAN 模型容易在平坦区域产生纹理伪影，加 TV 能压住
- **工程上权重很小**（通常 $\lambda = 10^{-5}$ 到 $10^{-3}$），权重大了会糊掉细节

### 梯度域损失

更精细的版本：在梯度域算 L1。

$$
\mathcal{L}_{\text{grad}} = ||\nabla \hat{x} - \nabla x||_1
$$

```python
import torch.nn.functional as F

# Sobel kernels
SOBEL_X = torch.tensor([[-1., 0., 1.],
                        [-2., 0., 2.],
                        [-1., 0., 1.]]).view(1, 1, 3, 3)
SOBEL_Y = SOBEL_X.transpose(-1, -2)

def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """梯度域 L1 损失。强迫预测的边缘和真值边缘对齐。
    pred, target: (B, C, H, W), 假设值域 [0, 1]
    """
    sobel_x = SOBEL_X.to(pred).expand(pred.shape[1], 1, 3, 3)
    sobel_y = SOBEL_Y.to(pred).expand(pred.shape[1], 1, 3, 3)

    pred_gx = F.conv2d(pred, sobel_x, padding=1, groups=pred.shape[1])
    pred_gy = F.conv2d(pred, sobel_y, padding=1, groups=pred.shape[1])
    targ_gx = F.conv2d(target, sobel_x, padding=1, groups=target.shape[1])
    targ_gy = F.conv2d(target, sobel_y, padding=1, groups=target.shape[1])

    return (pred_gx - targ_gx).abs().mean() + (pred_gy - targ_gy).abs().mean()
```

何时用：

- 边缘非常重要的任务（去模糊、文档增强）
- 作为感知损失的轻量替代

## 3.4 感知损失

第 2 章已经介绍过 VGG perceptual loss。这里补充另外几个变体。

### LPIPS

LPIPS（Learned Perceptual Image Patch Similarity）是 2018 年的工作，本质是**手工权重的感知损失换成数据驱动**：

1. 用 AlexNet/VGG/SqueezeNet 提特征
2. **对每层特征加一个学习出来的线性权重**（在大量人类感知判断的数据上训）
3. 把加权后的距离当作感知距离

它比 VGG perceptual loss 更对齐人眼。但**作为损失训练**时有几个坑：

- 计算量比 VGG 大
- 在某些纹理上有偏差
- 直接用作主损失容易让模型生成"游戏化"输出（针对 LPIPS 优化但视觉差）

实际工程里 LPIPS 更多作为**评估指标**用，第 4 章详谈。作为损失时仍以 VGG 为主。

### DISTS

DISTS（Deep Image Structure and Texture Similarity）是 2020 年的工作，把感知损失分成结构和纹理两部分。它的特点是**对位置不敏感**——同样的纹理在不同位置不会被惩罚。

何时用：纹理生成任务（皮肤、毛发、布料），DISTS 比 VGG 更合适。

### CLIP loss

把图过 CLIP image encoder，在 CLIP 特征空间算距离。

性质：

- 对**语义内容**敏感，对低层细节不敏感
- 适合做"内容保持"约束，不适合做主损失
- 在生成式增强（diffusion）里用得多，判别式增强里少

工程经验：单独用 CLIP loss 训不出好的增强模型，但**把它作为正则项**（权重 0.01）能保证模型不"漂移"——比如修复一张猫照片不会变成狗。

## 3.5 对抗损失

GAN 损失是这本书最复杂、最容易踩坑、也最关键的损失类型之一。它的角色：**让模型生成真实细节**，不再"安全地输出模糊"。

### Vanilla GAN —— 不要直接用

原始 GAN 损失：

$$
\min_G \max_D \mathbb{E}_{x \sim p_{\text{data}}} [\log D(x)] + \mathbb{E}_{z \sim p_z} [\log(1 - D(G(z)))]
$$

理论好但工程上**极不稳定**：

- 梯度消失：当 $D$ 训得太好时，$\log(1 - D(G(z)))$ 在 $D(G(z)) \to 0$ 时梯度饱和
- 模式崩溃（mode collapse）：$G$ 输出趋向单一模式
- 训练发散：$D$ 和 $G$ 无法达到 Nash 均衡

实际工程不用 vanilla GAN，用下面的变体。

### LSGAN —— 简单稳定

把 sigmoid + log 替换成 MSE：

$$
\mathcal{L}_D = \frac{1}{2}\mathbb{E}[(D(x) - 1)^2] + \frac{1}{2}\mathbb{E}[(D(G(z)))^2]
$$
$$
\mathcal{L}_G = \frac{1}{2}\mathbb{E}[(D(G(z)) - 1)^2]
$$

性质：梯度不饱和、训练稳定、超参数好调。

### Hinge loss —— 现代 GAN 标准

$$
\mathcal{L}_D = -\mathbb{E}[\min(0, D(x) - 1)] - \mathbb{E}[\min(0, -D(G(z)) - 1)]
$$
$$
\mathcal{L}_G = -\mathbb{E}[D(G(z))]
$$

性质：当 $D$ 已经分得很开时不再继续推（margin），训练更稳。BigGAN、StyleGAN 等都用这个。

```python
def hinge_d_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """Hinge 损失 (判别器一侧)。
    d_real, d_fake: 判别器对真/假图的输出 logits, shape (B,) 或 (B, 1, h', w')
    """
    loss_real = F.relu(1.0 - d_real).mean()
    loss_fake = F.relu(1.0 + d_fake).mean()
    return loss_real + loss_fake


def hinge_g_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """Hinge 损失 (生成器一侧)。"""
    return -d_fake.mean()
```

### Relativistic GAN —— ESRGAN 之选

ESRGAN 用的是 Relativistic average GAN（RaGAN）。它的核心思想：

> 判别器不应该判断"这是真的吗？"
> 应该判断"这比那个假的更像真的吗？"

形式上：

$$
D_{\text{Ra}}(x_r, x_f) = \sigma(D(x_r) - \mathbb{E}[D(x_f)])
$$

$$
D_{\text{Ra}}(x_f, x_r) = \sigma(D(x_f) - \mathbb{E}[D(x_r)])
$$

判别器损失同时关心"真的更像真的"和"假的不那么像真的"，让 $D$ 的训练信号对 $G$ 更友好。

```python
def relativistic_d_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """RaGAN 判别器损失 (ESRGAN 风格)。"""
    real_logits = d_real - d_fake.mean()
    fake_logits = d_fake - d_real.mean()
    return (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
          + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)))


def relativistic_g_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """RaGAN 生成器损失。"""
    real_logits = d_real - d_fake.mean()
    fake_logits = d_fake - d_real.mean()
    return (F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
          + F.binary_cross_entropy_with_logits(real_logits, torch.zeros_like(real_logits)))
```

实测：在超分任务上 RaGAN 比 LSGAN 提升约 0.05 LPIPS，是 ESRGAN-Real-ESRGAN 这条线长期的标配。

### Patch GAN

不输出一个标量判别值，而是一张特征图，每个位置判断对应感受野的真假。

性质：

- **局部判别**：模型不能在某个区域偷懒
- **天然支持任意分辨率**输入
- **更稳定**：每个 patch 独立提供梯度，不会被全局信号淹没

代码（第 6 章会展开判别器架构）：

```python
class PatchDiscriminator(nn.Module):
    """70x70 receptive field PatchGAN, Pix2Pix/Real-ESRGAN 风格。"""
    def __init__(self, in_ch: int = 3, base_ch: int = 64):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]
        for i in range(len(chs) - 1):
            stride = 2 if i < 2 else 1
            layers += [
                nn.Conv2d(chs[i], chs[i+1], 4, stride=stride, padding=1),
                nn.GroupNorm(8, chs[i+1]),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        layers.append(nn.Conv2d(chs[-1], 1, 4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # 输出 (B, 1, h', w'), 不做 sigmoid (logits)
```

### 谱归一化 + 加 dropout

工程上还有几个稳定 GAN 训练的技巧，作为损失之外的辅助：

- **Spectral Normalization**：在判别器每层做谱归一化，控制 Lipschitz 常数，效果显著
- **R1 / R2 正则**：在判别器对真/假输入加梯度惩罚
- **Two-Time-Scale Update Rule (TTUR)**：$D$ 用更高学习率（4×）

第 11 章会把这些工程细节集中讲。

## 3.6 扩散损失

扩散模型有自己一套损失，**和上面的判别式损失体系不在同一个框架里**。

### Simple loss（epsilon prediction）

DDPM 的标准目标：

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ ||\epsilon - \epsilon_\theta(x_t, t)||^2 \right]
$$

模型直接预测加进去的噪声。这是大多数扩散教程里看到的形式。

```python
def diffusion_simple_loss(model, x0, t, noise_scheduler):
    """DDPM 标准的 epsilon prediction 损失。
    model: UNet, 输入 (x_t, t) 输出预测噪声
    x0: 原始图像 (B, C, H, W) or 潜空间 (B, c, h, w)
    t: 时间步 (B,)
    noise_scheduler: 提供 alpha_cumprod
    """
    noise = torch.randn_like(x0)
    sqrt_alpha = noise_scheduler.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = noise_scheduler.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    x_t = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
    pred_noise = model(x_t, t)
    return F.mse_loss(pred_noise, noise)
```

### v-prediction —— 数值更好

原始 DDPM 的问题：当 $t$ 很小（接近 $x_0$）时，$\epsilon$ 已经几乎被全部"挤出去"，预测目标信号弱、信噪比差。

v-prediction（Salimans & Ho 2022）改成预测：

$$
v_t = \sqrt{\bar{\alpha}_t} \cdot \epsilon - \sqrt{1 - \bar{\alpha}_t} \cdot x_0
$$

其中 $\sqrt{\bar{\alpha}_t}$ 是信号缩放系数、$\sqrt{1 - \bar{\alpha}_t}$ 是噪声缩放系数（与 8.3 节同一定义）。这个量在所有 $t$ 上数值范围相对均匀，训练更稳，特别是在低 $t$ 区域。Stable Diffusion 2.x、Imagen 用 v-prediction；SDXL base 模型则仍用 epsilon-prediction（具体看 checkpoint 的 `prediction_type` 配置）。

### x0-prediction

直接预测原图 $x_0$。在某些任务（特别是恢复任务）上比 epsilon 更直观——因为我们关心的就是 $x_0$ 的质量。

三种预测目标可以互相转换，但训练动力学不同。**经验上**：

- 通用文生图：v 或 epsilon
- 增强/恢复：x0 或 v
- 极低 SNR 区段：x0 更稳

### Min-SNR 加权

不同时间步的 loss 量级差异巨大，直接平均会让模型过度关注某些 $t$。Min-SNR weighting（Hang et al. 2023）：

$$
w(t) = \min\left(\text{SNR}(t), \gamma\right) / \text{SNR}(t)
$$

其中 $\gamma$ 通常取 5。这是 SDXL 等现代扩散训练的标配。

## 3.7 频域损失

直接在 FFT 域算损失，强调高频成分。

### Focal Frequency Loss

```python
import torch

def focal_frequency_loss(pred: torch.Tensor, target: torch.Tensor,
                         alpha: float = 1.0) -> torch.Tensor:
    """Focal Frequency Loss (FFL, Jiang et al. 2021)。
    把预测和真值都做 FFT, 在频域算加权 L2,
    高频差异权重更大。
    pred, target: (B, C, H, W)
    """
    pred_fft   = torch.fft.fft2(pred,   norm='ortho')
    target_fft = torch.fft.fft2(target, norm='ortho')

    diff = pred_fft - target_fft
    distance = (diff.real ** 2 + diff.imag ** 2)  # |.|^2

    # focal 权重 (难训的频率分量权重大)
    weight = distance.detach() ** alpha
    weight = weight / (weight.max() + 1e-8)

    return (weight * distance).mean()
```

何时用：

- 模型输出明显糊（像素损失收敛但 PSNR 不再涨）
- 某个频段一直恢复不好（实测 FFT power spectrum 看出哪段缺）
- 超分高倍率（4×、8×），高频权重特别值钱

工程经验：FFL 单独用容易让模型生成"格点状"伪影，**当辅助损失加上权重 0.05-0.1**比较合理。

## 3.8 任务特化损失

针对特定任务的额外约束。

### 颜色一致性损失

在 Lab 颜色空间或 YCbCr 空间的 ab/CbCr 通道上算 L1，约束色彩不漂移。

```python
def color_consistency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """对预测和真值都做大幅模糊后比较, 只关心颜色不关心细节。
    简单有效, 用在去模糊/超分等任务防止色调漂移。
    """
    pred_blur   = F.avg_pool2d(pred,   kernel_size=11, stride=1, padding=5)
    target_blur = F.avg_pool2d(target, kernel_size=11, stride=1, padding=5)
    return F.l1_loss(pred_blur, target_blur)
```

### 身份保留损失（人脸专用）

GFPGAN/CodeFormer 用的：把预测和真值都过 ArcFace 人脸识别模型，比较 embedding。

```python
class IdentityLoss(nn.Module):
    """人脸增强专用 - 用预训练 ArcFace 提 ID embedding, 算余弦距离。"""
    def __init__(self, arcface_model: nn.Module):
        super().__init__()
        self.arcface = arcface_model.eval()
        for p in self.arcface.parameters():
            p.requires_grad_(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        emb_pred   = self.arcface(pred)    # (B, 512)
        emb_target = self.arcface(target)
        return 1.0 - F.cosine_similarity(emb_pred, emb_target).mean()
```

这是人脸修复"修不变身份"的关键。第 10 章详谈。

### 时序一致性损失（视频专用）

视频任务里，相邻帧之间应当满足光流约束。第 13 章详谈。

## 3.9 混合策略 —— 这一章的核心

把上面所有损失项放在一起，怎么调权重？

### 经典 ESRGAN 配方

$$
\mathcal{L}_G = \lambda_1 \mathcal{L}_1 + \lambda_2 \mathcal{L}_{\text{percep}} + \lambda_3 \mathcal{L}_{\text{adv}}
$$

权重：$\lambda_1 = 1.0$, $\lambda_2 = 1.0$, $\lambda_3 = 5 \times 10^{-3}$

注意对抗损失权重**很小**（5/1000）。原因：对抗损失梯度大、有噪声，权重大了会破坏像素一致性。

### Real-ESRGAN 配方

加上一个二阶退化合成（不是损失改的，是数据改的，第 5 章详讲），损失项几乎一样：

$$
\mathcal{L}_G = \mathcal{L}_1 + \mathcal{L}_{\text{percep}} + 0.1 \cdot \mathcal{L}_{\text{adv}}
$$

对抗权重提到 0.1 而不是 0.005，因为数据更"难"（更接近真实退化），需要对抗损失贡献更多。

### 现代扩散派配方（SUPIR）

扩散模型主损失就是 simple loss / v-loss / x0-loss，外加：

- **Latent perceptual loss**（在潜空间算 LPIPS）权重 0.1
- **解码后的像素 L1**（防止 VAE 重建漂移）权重 0.1
- **CLIP loss**（语义不漂移）权重 0.01

具体见第 8-9 章。

### 权重调参的工程经验

新手最常踩的坑：**所有损失从一开始就加上**。这容易让训练前期被对抗损失或感知损失带偏，模型不收敛。

推荐两阶段策略：

**阶段一（pretrain）**：只用像素损失（L1 + Charbonnier），训到 PSNR 收敛。这个阶段模型学到"基本恢复"，输出虽然糊但稳定。

**阶段二（finetune）**：加入感知损失和对抗损失，权重从小到大。模型学会"恢复 + 锐化"，PSNR 会下降一点但视觉质量大幅提升。

ESRGAN/Real-ESRGAN 的官方训练流程都是这两阶段。

### 损失曲线诊断

训练时同时记录每个损失项的值，看出问题：

| 症状 | 可能原因 | 调整方向 |
|------|---------|---------|
| 像素损失下降但感知损失不下降 | 模型陷入"安全模糊" | 增大感知损失权重 |
| 对抗损失剧烈震荡 | $D$ 训得过强或过弱 | 调 D/G 学习率比、加 spectral norm |
| 早期对抗损失突然爆炸 | 没有 pretrain 直接上对抗 | 先 pretrain 再加对抗 |
| 感知损失先降后升 | 过拟合 VGG 的对抗样本 | 加 LPIPS / 减 VGG 权重 |
| 颜色漂移 | 缺少色彩约束 | 加颜色一致性损失 |
| FID 不降但 LPIPS 降 | 多样性不足 | 检查数据增广、增大对抗权重 |

## 3.10 损失选择决策表

按任务给一个起点配方（具体权重要根据数据/网络微调）：

| 任务 | 主损失 | 辅助损失 | 备注 |
|------|--------|---------|------|
| 经典 SR (PSNR 导向) | Charbonnier | — | 学术 benchmark 用 |
| 真实 SR (视觉导向) | L1 + VGG + RaGAN | + 0.05 FFL | Real-ESRGAN 风格 |
| 去噪 | Charbonnier | + 0.001 TV | NAFNet 风格 |
| 去模糊 | Charbonnier + gradient | + VGG | Restormer 风格 |
| 人脸修复 | L1 + VGG + Adv + Identity | — | GFPGAN/CodeFormer 风格 |
| 扩散增强 | v-prediction MSE | + 潜空间 LPIPS + CLIP | SUPIR 风格 |
| 视频超分 | Charbonnier + VGG | + 时序一致 | BasicVSR++ 风格 |
| 帧插值 | Charbonnier + VGG + LapPyr | — | RIFE 风格 |

记住一个工程直觉：

> **想要锐 → 加对抗损失**
> **想要保真 → 加重像素损失**
> **想要颜色对 → 加颜色一致性**
> **想要边缘对 → 加梯度损失**
> **想要细节真 → 加感知损失**
> **想要不漂移内容 → 加 CLIP 损失**

## 3.11 小结

1. **增强损失几乎从来不是单一的**——四五项混合是常态
2. **像素空间用 L1 / Charbonnier 而不是 L2**——更鲁棒、更锐利
3. **感知损失（VGG/LPIPS）让模型重视高频和语义** —— 但不能单独用
4. **对抗损失（RaGAN/Hinge）让模型生成真实细节** —— 权重要小、要稳定
5. **扩散损失自成一系**（simple/v/x0），与判别式损失体系不混用
6. **频域和任务特化损失**作为补丁，针对性强但权重要克制
7. **混合策略 = pretrain 像素 → finetune 加感知和对抗**
8. **损失曲线诊断**比单看最终指标更能告诉你训练在哪里出问题

后面架构章（第 6-10 章）讲到具体模型时，会反复回到本章——每个模型的"训练方案"小节，本质都是损失加权的具体方案。

---

> 下一章 [评估的陷阱](04-metrics.md) → 我们将看到，影像增强领域的指标比损失函数更容易骗人。
