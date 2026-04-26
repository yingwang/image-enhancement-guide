# 第 5 章 · 数据与退化合成

> 这是这本书最"工程"的一章。
>
> 影像增强领域有一句话流传很久：**Real-ESRGAN 的核心贡献不是网络，是数据**。
>
> 这一章把这句话拆解清楚。

## 5.1 数据 > 网络

第 1 章提到过，2014-2020 年这个领域有一个反复出现的模式：

- 用 bicubic 下采样合成训练数据
- 在合成数据上训出 SOTA 网络
- 实际部署时模型在真实图像上几乎不工作

2021 年的 Real-ESRGAN 论文做了一个对比实验：

| 网络 | 训练数据 | 真实图像 PSNR | 真实图像视觉 |
|------|---------|--------------|------------|
| ESRGAN（RRDB 网络） | bicubic 退化 | 18.2 dB | 几乎不工作 |
| **Real-ESRGAN（同样的 RRDB 网络）** | **复杂退化合成** | **23.8 dB** | **可用** |

**网络没变，数据变了，性能从"不工作"到"可用"**。这是这个领域最重要的一次方法论修正。

类似的故事在去噪、去模糊、视频增强里反复出现：

- DnCNN 在 BSD68（标准基准）上无敌，在真实手机图上不行——用 SIDD 真实数据微调后大幅提升
- SUPIR 的关键之一是用了亿级互联网图像 + 复杂退化合成
- 视频去抖在合成抖动数据上完美，在真实手机视频上需要额外的真实数据

**工程哲学**：

> 在影像增强里，**数据 pipeline 决定了模型能力上限**。
> 同样的网络在不同数据上能差出一个数量级，反过来不成立。

这一章讲数据 pipeline 怎么设计。

## 5.2 训练数据的两种来源

### 合成数据

**做法**：拿一批高质量图（HR），用代码模拟退化得到（HR, LR）配对。

```python
# 伪流程
for hr in high_quality_images:
    lr = degradation_pipeline(hr)
    yield (lr, hr)
```

优点：

- **无限**：只要有 HR，就能合成无限多 (LR, HR) 对
- **便宜**：不需要拍照设备
- **可控**：能精确控制退化的每个参数

缺点：

- **不真实**：合成出的 LR 不一定像真实拍到的烂图
- **偏差**：所有"已知不真实"的部分会被模型学到

### 真实数据

**做法**：用相机/扫描仪在物理世界采集 (HR, LR) 对。

实现方式：

- **不同焦段**（RealSR、DRealSR）：同一相机用不同焦段拍同一场景，远焦图当作 HR、近焦图当作 LR
- **不同设备**（DPED）：低端手机和单反同时拍同一场景
- **降质模拟**（少见）：拿 HR 走特定流程（打印 + 扫描）得到 LR

优点：**真实**——所有退化都是物理过程，不是合成假设

缺点：

- **少**：几百到几千张配对，对数据驱动方法不够
- **对齐难**：两次拍摄的物体位置、光线、白平衡都会偏，需要复杂的对齐流程
- **退化模式有限**：你拍的是哪种相机的哪种退化，模型就只对这种退化好

**工程实践**：合成数据为主 + 真实数据微调。

## 5.3 经典 bicubic 流程的问题（再深入一点）

第 1 章已经提过 bicubic 退化的问题。这里再深入：

**真实低分辨率图的复杂性**：

```
用户拍照流程:
  传感器 (光子噪声 + 读出噪声)
    → demosaicing (RGB 重建)
    → ISP (降噪 + 锐化 + 白平衡 + tone mapping + 色彩矩阵)
    → JPEG 压缩 (8-bit, quality 70-95)
    → 上传到微信/微博 (再压缩, quality 50-70)
    → 别人下载 (可能再压缩一次)
```

每一步都有变化，最终的 LR 是这些变化的复合。

**bicubic 退化只模拟了哪一步？** 严格说，**一步都没真正模拟**——它假设 LR 是 HR 的 ideal anti-aliasing 下采样，**与真实物理过程没有对应关系**。

ESRGAN 在 bicubic 数据上学到的是"反 bicubic 函数"，恰好这个函数对真实退化几乎零迁移性。

**修正这个问题的方向**：

让训练数据的退化分布**覆盖**真实世界的退化分布。Real-ESRGAN 的设计就是朝这个方向。

## 5.4 Real-ESRGAN 退化 pipeline 详解

Real-ESRGAN 的 pipeline 由两个关键设计组成：

### 设计一：每个组件参数化 + 随机化

每一种退化（模糊、下采样、噪声、JPEG）都不是固定的，而是**从一个分布里随机采**。模型在训练时见到了广泛的退化分布。

### 设计二：二阶退化（high-order degradation modeling）

把整个退化流程跑两遍。这是 Real-ESRGAN 的核心创新。

```
一阶退化 (模拟相机内部):
  HR → blur₁ → downsample₁ → noise₁ → jpeg₁ → 中间产物

二阶退化 (模拟传输/二次压缩):
  中间产物 → blur₂ → downsample₂ → noise₂ → jpeg₂ → LR
```

这样合成的 LR 就接近"图片被相机处理后又被互联网压缩多次"的真实状态。

第二阶的关键参数稍微不同：

- 第一阶 blur 偏大（模拟实际相机/传输模糊）
- 第二阶 blur 偏小（避免训练数据糊到不能学）
- 第二阶有可能加 **sinc 滤波**——模拟过度锐化导致的振铃伪影（这是 LCD 显示器/某些图像处理软件特有的伪影）

### 一个简化版的 Real-ESRGAN pipeline

```python
import random
import torch
import torch.nn.functional as F
from typing import Optional


class RealESRGANDegradation:
    """Real-ESRGAN 风格的退化合成 (简化但完整)。
    输入: HR clean image (B, 3, H, W) in [0, 1]
    输出: LR degraded image (B, 3, H/scale, W/scale) in [0, 1]
    
    省略部分: 真实代码里 blur kernel 类型更多, sinc 滤波, 
    GPU 上的可微 JPEG 等; 这里给出主流程。
    """

    def __init__(self, scale: int = 4):
        self.scale = scale

        # 一阶退化参数范围
        self.blur1_sigma_range = (0.2, 3.0)
        self.noise1_range = (1, 30)         # std on [0, 255]
        self.jpeg1_range = (30, 95)
        self.resize1_range = (0.15, 1.5)    # 相对 final size

        # 二阶退化参数范围 (整体偏弱)
        self.blur2_sigma_range = (0.2, 1.5)
        self.noise2_range = (1, 25)
        self.jpeg2_range = (30, 95)
        self.resize2_range = (0.3, 1.2)

    # ============= 模糊 =============
    def random_gaussian_blur(self, x: torch.Tensor,
                             sigma_range: tuple) -> torch.Tensor:
        sigma = random.uniform(*sigma_range)
        ksize = max(3, 2 * int(3 * sigma) + 1)
        kernel = self._gaussian_kernel_2d(ksize, sigma).to(x)
        kernel = kernel.expand(x.shape[1], 1, ksize, ksize)
        pad = ksize // 2
        return F.conv2d(F.pad(x, [pad]*4, mode='reflect'),
                        kernel, groups=x.shape[1])

    @staticmethod
    def _gaussian_kernel_2d(ksize: int, sigma: float) -> torch.Tensor:
        ax = torch.arange(ksize).float() - ksize // 2
        gauss = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
        kernel = (gauss[:, None] * gauss[None, :])
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)

    # ============= 下采样 =============
    def random_resize(self, x: torch.Tensor, target_size: tuple,
                      scale_range: tuple) -> torch.Tensor:
        """随机选择目标尺寸 (相对 target_size 缩放), 随机选插值。"""
        s = random.uniform(*scale_range)
        h, w = target_size
        new_h = max(1, int(h * s))
        new_w = max(1, int(w * s))
        mode = random.choice(['bilinear', 'bicubic', 'area'])
        kw = {'mode': mode}
        if mode in ('bilinear', 'bicubic'):
            kw['align_corners'] = False
        return F.interpolate(x, size=(new_h, new_w), **kw)

    # ============= 噪声 =============
    def random_noise(self, x: torch.Tensor,
                     sigma_range_255: tuple) -> torch.Tensor:
        """三种噪声等概率: 灰度高斯, 彩色高斯, 泊松。"""
        choice = random.random()
        sigma_max = random.uniform(*sigma_range_255) / 255.0

        if choice < 0.4:
            # 彩色高斯
            return (x + torch.randn_like(x) * sigma_max).clamp(0, 1)
        elif choice < 0.7:
            # 灰度高斯 (所有通道相同噪声)
            n = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
            return (x + n * sigma_max).clamp(0, 1)
        else:
            # 泊松 (signal-dependent)
            scale = random.uniform(80, 1000)
            photons = (x * scale).clamp(min=0)
            return (torch.poisson(photons) / scale).clamp(0, 1)

    # ============= JPEG (简化, 真实需 diffjpeg) =============
    def random_jpeg(self, x: torch.Tensor, quality_range: tuple) -> torch.Tensor:
        """占位符。真实代码:
            from diffjpeg import DiffJPEG
            quality = random.randint(*quality_range)
            return DiffJPEG(differentiable=False)(x, quality)
        """
        # 用一个 8x8 box filter 近似一下 JPEG 的块状效应 (示意, 不用于训练)
        return x

    # ============= 一阶退化 =============
    def first_order(self, hr: torch.Tensor, target_size: tuple) -> torch.Tensor:
        x = self.random_gaussian_blur(hr, self.blur1_sigma_range)
        x = self.random_resize(x, target_size, self.resize1_range)
        x = self.random_noise(x, self.noise1_range)
        x = self.random_jpeg(x, self.jpeg1_range)
        return x

    # ============= 二阶退化 =============
    def second_order(self, x: torch.Tensor, target_size: tuple) -> torch.Tensor:
        x = self.random_gaussian_blur(x, self.blur2_sigma_range)
        x = self.random_resize(x, target_size, self.resize2_range)
        x = self.random_noise(x, self.noise2_range)
        x = self.random_jpeg(x, self.jpeg2_range)
        return x

    # ============= 主流程 =============
    def __call__(self, hr: torch.Tensor) -> torch.Tensor:
        """完整 Real-ESRGAN 风格退化。
        hr: (B, 3, H, W) in [0, 1]
        return: lr (B, 3, H/scale, W/scale)
        """
        b, c, h, w = hr.shape
        out_h, out_w = h // self.scale, w // self.scale

        # 一阶: 缩到 [0.15*out, 1.5*out] 之间的中间尺寸
        intermediate = self.first_order(hr, target_size=(out_h, out_w))

        # 二阶: 最终缩到 (out_h, out_w), 注意先 resize 到目标
        lr = self.second_order(intermediate, target_size=(out_h, out_w))

        # 强制最终尺寸 (二阶 resize 可能让尺寸偏离)
        lr = F.interpolate(lr, size=(out_h, out_w),
                          mode='bicubic', align_corners=False)
        return lr.clamp(0, 1)
```

实战中的版本会比这个复杂：完整的 Real-ESRGAN 代码大约 700 行。补的内容包括：

- 多种模糊核（各向异性、广义高斯、plateau）
- sinc 滤波（模拟过锐化伪影）
- 可微 JPEG（直接在 GPU 上做）
- 灰度图/彩色图的概率分配
- USM sharpening 反向（模拟手机后处理）

但骨架就是上面这个。

## 5.5 模糊核家族

不同退化场景需要不同的模糊核。

### 各向同性高斯（最基础）

$$
k(u, v) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{u^2 + v^2}{2\sigma^2}\right)
$$

只有一个参数 $\sigma$。模拟散焦、大气模糊。

### 各向异性高斯

$$
k(u, v) \propto \exp\left(-\frac{1}{2}\begin{pmatrix}u\\v\end{pmatrix}^T \Sigma^{-1} \begin{pmatrix}u\\v\end{pmatrix}\right)
$$

协方差矩阵 $\Sigma$ 控制 x/y 方向不同的模糊量。模拟相机抖动、镜头像差。

### 广义高斯（Generalized Gaussian）

$$
k(u, v) \propto \exp\left(-\left(\frac{u^2}{\sigma_x^2} + \frac{v^2}{\sigma_y^2}\right)^\beta\right)
$$

多一个形状参数 $\beta$：$\beta < 1$ 像 plateau 模糊（中心平坦边缘陡峭），$\beta = 1$ 是标准高斯，$\beta > 1$ 边缘更尖。

### 运动模糊核

线段形：

```python
import numpy as np
import torch

def motion_blur_kernel(length: int, angle_deg: float) -> torch.Tensor:
    """长度为 length 的线段形运动模糊核, 方向 angle_deg 度。"""
    kernel = np.zeros((length, length), dtype=np.float32)
    angle = np.deg2rad(angle_deg)
    cx, cy = length // 2, length // 2
    for i in range(length):
        dx = int(round(cx + (i - cx) * np.cos(angle)))
        dy = int(round(cy + (i - cx) * np.sin(angle)))
        if 0 <= dx < length and 0 <= dy < length:
            kernel[dy, dx] = 1.0
    kernel /= kernel.sum()
    return torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)
```

### Sinc 滤波

$$
k(u, v) = \frac{\omega_c}{2\pi r} J_1(\omega_c r), \quad r = \sqrt{u^2 + v^2}
$$

其中 $J_1$ 是一阶 Bessel 函数。Sinc 在频域是理想低通滤波（rect 形），但截断的 sinc 在像素域有振荡——会产生**振铃**伪影。

为什么 Real-ESRGAN 用 sinc：模拟某些图像处理软件（如 Adobe 系产品）的锐化算法在过度处理后产生的振铃。这种伪影在真实"被处理过"的图像里很常见。

## 5.6 下采样的多种策略

不同的插值核在频域行为差别巨大：

| 算法 | 频域行为 | 视觉特点 | 何时用 |
|------|---------|---------|-------|
| nearest | 矩形（高频泄漏严重） | 锯齿 | 模拟极差的处理 |
| bilinear | 三角形 | 略糊 | 均衡选择 |
| bicubic | 接近 sinc | 锐利但有 ringing | 学术标准 |
| lanczos | 截断 sinc | 最锐利、振铃明显 | 模拟某些软件输出 |
| area / box | 矩形（空间） | 柔和无锯齿 | 大幅缩小时 |

Real-ESRGAN 的 pipeline 在每次 resize 时**随机选**这些之一，让模型见过各种插值伪影。

## 5.7 噪声的实现细节

第 1 章已经讲过噪声的物理模型。这里补充几个工程细节。

### 灰度噪声 vs 彩色噪声

真实噪声的颜色相关性是个微妙问题：

- 高端相机：噪声大致独立同分布（每通道独立）
- 手机摄像头：经过 demosaicing 后，相邻像素的噪声相关；通道间也有相关
- ISP 降噪后：剩余噪声常常是"灰色"（三通道相关）

工程实践：训练数据里**两种都给**——50% 时间用三通道独立的彩色噪声，50% 时间用灰度噪声（所有通道用同一个噪声 map）。

### 噪声强度分布

噪声 sigma 不要均匀采样——真实场景里**小噪声更常见，大噪声较少见**。

工程上常用：

```python
# 偏小噪声居多的分布
sigma = torch.rand(()) ** 2 * sigma_max  # 平方让分布偏小
```

或者从一个明确的混合分布采：

```python
if random.random() < 0.5:
    sigma = random.uniform(0.005, 0.02)   # 小噪声 (常见)
else:
    sigma = random.uniform(0.02, 0.08)    # 大噪声 (少见但要见过)
```

### 真实传感器噪声模型

如果想做更精确的模拟，可以用论文 "A Physics-based Noise Formation Model for Extreme Low-light Raw Denoising"（CVPR 2020）提出的物理模型。

简化版：

```python
def realistic_sensor_noise(
    x: torch.Tensor,
    iso: int = 1600,
    quantum_efficiency: float = 0.5,
    read_noise_sigma: float = 0.005,
    dark_current: float = 0.001,
    quant_step: float = 1/255.0,
) -> torch.Tensor:
    """模拟真实传感器噪声链。
    iso 越大, 各类噪声越强。
    """
    # 模拟"相机增益"对噪声的影响
    gain = iso / 100.0

    # 1. 光子噪声 (Poisson, signal-dependent)
    photons = x * 1000 / gain  # 标定光子数
    photons_noisy = torch.poisson(photons.clamp(min=0))
    shot = photons_noisy / 1000 * gain

    # 2. 暗电流噪声
    dark = torch.poisson(torch.full_like(x, dark_current * gain)) / 1000

    # 3. 读出噪声 (Gaussian, signal-independent)
    read = torch.randn_like(x) * read_noise_sigma * gain

    # 4. 量化误差
    out = shot + dark + read
    out = (out / quant_step).round() * quant_step

    return out.clamp(0, 1)
```

这个模型对**手机暗光去噪**的训练数据合成至关重要。

## 5.8 JPEG 压缩

JPEG 在合成 pipeline 里需要满足：

1. **可以在 GPU 上做**（不要每张图存盘读盘）
2. **理想情况下可微**（虽然增强训练里梯度不会传过 JPEG，但可微版本性能也不错）

推荐用 **DiffJPEG** 库（[github.com/mlomnitz/DiffJPEG](https://github.com/mlomnitz/DiffJPEG)），它实现了可微的 JPEG 编解码：

```python
from DiffJPEG import DiffJPEG

# 不需要可微 (推理用)
jpeg_module = DiffJPEG(differentiable=False, quality=80).to(device)

# 可微 (训练时如果想反传)
jpeg_module = DiffJPEG(differentiable=True, quality=80).to(device)

def random_jpeg_diffjpeg(x: torch.Tensor, quality_range=(30, 95)) -> torch.Tensor:
    quality = random.randint(*quality_range)
    return DiffJPEG(differentiable=False, quality=quality).to(x.device)(x)
```

### 第二阶 JPEG 的特殊性

Real-ESRGAN 的二阶 JPEG 模拟"已经压缩过的图被再次压缩"。这种情况下：

- 第一阶 JPEG 的块状伪影会被第二阶 JPEG 进一步混乱
- 块边界对不齐导致复杂的复合伪影
- 真实"网络重传"图像里这种现象普遍存在

直接两次调用 JPEG 就能模拟这种现象。

## 5.9 数据集选择

按用途列出常用数据集：

### 通用 SR / 去噪 / 去模糊

| 数据集 | 张数 | 特点 | 用途 |
|-------|------|------|------|
| **DIV2K** | 800 + 100 val | 高质量自然图，2K 分辨率 | 学术标准 |
| **Flickr2K** | 2650 | DIV2K 风格补充 | 与 DIV2K 合并成 DF2K |
| **LSDIR** | 84,991 + 250 val | 2023 大规模数据集 | 现代训练首选 |
| **BSD500** | 500 | 老但经典 | 去噪 |
| **OST300** | 300 | 户外纹理特化 | 真实 SR 辅助 |
| **WED** | ~5000 | Watermark/exif 多样 | 真实场景 |

### 真实退化（无合成）

| 数据集 | 张数 | 特点 |
|-------|------|------|
| **RealSR** | 595 (V3) | Canon + Nikon 不同焦段对 |
| **DRealSR** | ~800 | DSLR 不同焦段，对齐更精 |
| **NTIRE Real-World SR** | 每年增 | 比赛数据，质量高 |
| **DPED** | ~16K | 手机 vs 单反 |
| **SIDD** | 320 (HR) + 30K (LR) | 真实手机噪声 |
| **DND** | 50 测试 | Darmstadt Noise Dataset，真实噪声测试 |

### 任务特化

| 数据集 | 用途 |
|-------|------|
| **FFHQ** | 70K 高质量人脸，人脸增强 |
| **CelebA-HQ** | 30K 人脸，老一点的标准 |
| **REDS** | 视频去模糊/SR |
| **Vimeo-90K** | 视频帧插值 |
| **GoPro** | 视频去模糊 |
| **UDM10** | 视频去噪 |

### 数据集组合策略

最常见的两种：

1. **学术标准**：DF2K（DIV2K + Flickr2K）训练，Set5/14/B100/Urban100/Manga109/DIV2K val 测试
2. **真实导向**：LSDIR + DF2K + OST300 训练，加少量 RealSR 微调，DRealSR/RealSR 测试

## 5.10 数据增广

低层视觉的数据增广和分类任务很不同。**安全**的：

```python
# 安全增广 (不改变退化模型)
import torchvision.transforms.functional as TF

def safe_augment(hr: torch.Tensor, lr: torch.Tensor):
    """对 HR-LR 配对做同步的安全增广。"""
    # 随机水平翻转
    if random.random() < 0.5:
        hr = TF.hflip(hr); lr = TF.hflip(lr)
    # 随机垂直翻转
    if random.random() < 0.5:
        hr = TF.vflip(hr); lr = TF.vflip(lr)
    # 90° 旋转
    if random.random() < 0.5:
        k = random.choice([1, 2, 3])
        hr = torch.rot90(hr, k, dims=[-2, -1])
        lr = torch.rot90(lr, k, dims=[-2, -1])
    return hr, lr
```

**慎用**的：

- **色彩抖动**：会改变退化分布，模型学到错的颜色映射
- **任意角度旋转**：插值会引入额外退化，不在原始 D 里
- **随机缩放**：等同于改 scale factor，增强 SR 时不要
- **mixup / cutmix**：低层视觉用得很少，效果不明确

**裁剪策略**：

```python
def random_crop(hr: torch.Tensor, lr: torch.Tensor,
                lr_size: int, scale: int):
    """随机裁剪 LR + 对应 HR 区域。"""
    _, _, lr_h, lr_w = lr.shape
    top  = random.randint(0, lr_h - lr_size)
    left = random.randint(0, lr_w - lr_size)
    lr_crop = lr[:, :, top:top+lr_size, left:left+lr_size]
    hr_crop = hr[:, :, top*scale:(top+lr_size)*scale,
                       left*scale:(left+lr_size)*scale]
    return hr_crop, lr_crop
```

## 5.11 数据加载 pipeline 优化

退化合成的计算量不小（特别是模糊 + JPEG），如果在 CPU 上做会成为训练瓶颈。两种优化：

### 选项一：CPU 多 worker

```python
from torch.utils.data import DataLoader, Dataset

class SRDataset(Dataset):
    def __init__(self, hr_paths, scale=4, crop_size=256):
        self.hr_paths = hr_paths
        self.degrader = RealESRGANDegradation(scale=scale)
        self.crop_size = crop_size

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr = load_image(self.hr_paths[idx])  # PIL or cv2
        hr = random_crop_hr(hr, self.crop_size)
        hr_tensor = to_tensor(hr).unsqueeze(0)  # (1, 3, H, W)
        lr_tensor = self.degrader(hr_tensor)
        return hr_tensor.squeeze(0), lr_tensor.squeeze(0)


loader = DataLoader(
    SRDataset(hr_paths),
    batch_size=16,
    num_workers=8,           # 8 CPU workers
    pin_memory=True,
    persistent_workers=True,
)
```

### 选项二：GPU 上做退化（Real-ESRGAN 推荐）

把退化 pipeline 移到 GPU 上，整个 batch 一次过：

```python
# 训练循环里
for hr_batch in loader:                  # CPU 只读 HR
    hr_batch = hr_batch.cuda()
    lr_batch = degrader(hr_batch)        # GPU 上一次性合成
    pred = model(lr_batch)
    loss = compute_loss(pred, hr_batch)
    ...
```

Real-ESRGAN 的官方代码就用这种模式。优点：

- 大幅减少 CPU 工作量
- 可以用更复杂的退化（多 GPU 并行合成）
- 退化结果可以利用 GPU 的随机性（每个 step 不一样）

缺点：

- 占 GPU 显存（一些临时 tensor）
- 退化代码必须支持 batched + GPU

### LMDB 存储加速大数据集

LSDIR 这种 8 万+ 张图的数据集，每个 epoch 解码 PNG 是显著开销。用 LMDB 把数据预编码后存为 key-value 数据库，加载快 5-10×：

```python
import lmdb
import pickle

# 写入
env = lmdb.open('lsdir.lmdb', map_size=int(1e12))
with env.begin(write=True) as txn:
    for i, path in enumerate(hr_paths):
        img_bytes = open(path, 'rb').read()  # 或者预 decode 成 numpy
        txn.put(f"{i:08d}".encode(), pickle.dumps(img_bytes))

# 读取
class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, length):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        self.length = length

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            data = pickle.loads(txn.get(f"{idx:08d}".encode()))
        # decode bytes to image
        ...
```

## 5.12 真实数据微调

通用最佳实践：

```
阶段 1: pretrain (合成数据)
  - 数据: DF2K + LSDIR
  - 退化: Real-ESRGAN pipeline
  - 训练步数: 500K - 1M
  - 目标: 学到通用的退化恢复能力

阶段 2: fine-tune (真实数据)
  - 数据: RealSR / DRealSR (几百张)
  - 训练步数: 10K - 50K (远少于 pretrain)
  - 学习率: 比 pretrain 小 10×
  - 目标: 把通用模型 calibrate 到真实退化分布
```

不直接在真实数据上从头训的原因：数据太少（几百张）—— 模型容易过拟合到这几百张的具体物体和场景。

## 5.13 数据质量审计

在跑训练前，**审计你的训练数据本身**：

### HR 数据是否真"高质量"

DIV2K 的 HR 看起来高质量，但仍有：

- 轻微 JPEG 压缩痕迹
- 偶尔的 noise pattern
- 局部过曝 / 欠曝

如果 HR 本身有压缩伪影，模型会学到"输出带轻微压缩伪影是正常的"——你的"HR 真值"不是真正的 ground truth。

工程检查：

```python
# 审计 HR 数据集的 JPEG 压缩痕迹
def audit_jpeg_traces(hr_path: str) -> float:
    """通过分析 8x8 块边界处的不连续性, 估计 HR 数据有多"被压缩过"。
    返回值越大, JPEG 压缩痕迹越明显。
    """
    img = cv2.imread(hr_path)
    img = img.astype(np.float32)
    # 块边界 (8 的倍数列/行) 处的差异
    h, w = img.shape[:2]
    block_diff = 0
    for i in range(7, h-1, 8):
        block_diff += np.abs(img[i] - img[i+1]).mean()
    for j in range(7, w-1, 8):
        block_diff += np.abs(img[:, j] - img[:, j+1]).mean()
    return block_diff
```

### 数据多样性审计

模型在某类内容上失败，常常是数据偏差：

- 如果训练数据 90% 是城市/自然风景，模型在文档/截图上失败
- 如果训练数据全是日光照片，模型在夜景上失败
- 如果训练数据全是西方人脸，模型在亚洲/非洲人脸上失败

这种偏差不会被指标显示出来——平均 PSNR 看起来正常，但失败模式集中。第 17 章会专门讲。

最简单的审计：

```python
# 用 CLIP 把训练集每张图嵌入, 然后聚类看分布
import open_clip

clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32')
embeddings = []
for path in hr_paths:
    img = preprocess(Image.open(path)).unsqueeze(0)
    with torch.no_grad():
        emb = clip_model.encode_image(img).cpu().numpy()
    embeddings.append(emb)

# K-means 聚 20 类, 看每类是什么内容、占比多少
from sklearn.cluster import KMeans
clusters = KMeans(n_clusters=20).fit_predict(np.vstack(embeddings))
```

如果某一类占比极少（< 1%），考虑专门补这部分数据。

## 5.14 小结

1. **数据 > 网络**：低层视觉里同样的网络在不同数据上能差一个数量级
2. **bicubic 退化是这个领域 5 年的方法论错误**，Real-ESRGAN 系统性纠正了它
3. **退化 pipeline 三个核心**：参数化 + 随机化 + 二阶
4. **每个组件都是一个家族**：模糊核多种、噪声多种、JPEG 多 quality、resize 多算法
5. **数据集的选择决定模型能见到什么世界**：DF2K + LSDIR 是合成数据通用基础，RealSR/DRealSR 用于真实退化微调
6. **数据加载 pipeline 优化**：GPU 上做退化是 Real-ESRGAN 风格的事实标准
7. **数据增广要克制**：低层视觉只用翻转 + 90° 旋转，色彩抖动等会破坏退化模型
8. **审计你的 HR 数据**：HR 本身的 JPEG 痕迹和数据偏差会被模型学到
9. **真实数据微调**是把合成训练模型 calibrate 到真实分布的关键步骤

到这里 Part I 五章全部完成。读完这五章，你应当能：

- 看到一张烂图，知道它经历了哪些 D
- 看到一个增强模型，知道它在哪种空间预测、用什么损失
- 看到一组指标，知道哪些可信、哪些有偏
- 看到一个数据 pipeline，知道它能 cover 真实分布的哪部分

Part II 开始我们进入具体的网络架构。第一站是 CNN 时代——这条线从 2014 年的 SRCNN 一直走到 2022 年的 NAFNet，是这个领域的"传统重武器"。

---

> 下一章 [CNN 时代](06-cnn.md) → 从 SRCNN 三层网络到 NAFNet 移除所有非线性激活，CNN 在低层视觉走过的 8 年。
