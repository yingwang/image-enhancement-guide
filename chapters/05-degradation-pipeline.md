# 第 5 章 · 数据与退化合成

> 这是全书最偏工程实践的一章。
>
> 影像增强领域有一句流传很久的话：Real-ESRGAN 的核心贡献不是网络，而是数据。
>
> 这一章把这句话拆解清楚。

## 5.0 阅读须知

本章是 Part I 的最后一章，也是全书最贴近工程实现的一章。前四章建立的概念在这里全部落到 Python 代码、目录结构和训练循环里。读完之后，你应当能直接动手写出一份可喂给 ESRGAN、Real-ESRGAN 或 Restormer 的训练 dataloader。

本章预设你已经掌握：

- 第 1 章里"退化算子 $D$ 是一串随机复合"的图像（§1.5 那张数据流图，本章会扩充为更完整的版本）
- 第 2–3 章里像素空间、特征空间与损失函数的对应关系
- 第 4 章里 PSNR/SSIM/LPIPS 与 NIQE/MANIQA 的区别（决定了什么时候"真值不存在"）

不预设你写过 Real-ESRGAN 的 dataloader，也不预设你读过它七百行的退化代码。

本章首次出现或将反复出现的缩写如下，先集中列出一遍，后面用到具体术语时还会再展开一句话定义：

- **Real-ESRGAN**：2021 年的代表性"真实场景"超分模型，论文核心贡献是退化合成 pipeline 而不是网络结构
- **BSRGAN**（Blind Super-Resolution GAN）：与 Real-ESRGAN 同期、思路相近的 blind SR 方案，由 Zhang et al. 提出，用"随机化退化顺序"训出一个对真实图鲁棒的 SR 模型
- **SRMD**（Super-Resolution with Multiple Degradations）：早期把"退化参数"作为条件喂给网络的工作，属于 non-blind 的折衷方案
- **KernelGAN**：用 GAN 在测试图上估退化核的方法，blind SR 的另一条路线
- **RRDB**（Residual in Residual Dense Block）：ESRGAN 提出的密集残差块，Real-ESRGAN 沿用它作为 backbone（详见第 6 章）
- **DiffJPEG**：可微的 JPEG 编解码实现，让退化 pipeline 能在 GPU 上整批做完 JPEG 步骤
- **DCT**（Discrete Cosine Transform，离散余弦变换）：JPEG 压缩的核心算子（已在 §1.0 介绍过，此处复述）
- **ISP**（Image Signal Processor，图像信号处理器）：相机内部把传感器读数变成 8-bit RGB 的整套流水线（同样已在 §1.0 介绍）
- **LMDB**（Lightning Memory-Mapped Database）：一个轻量级的 key-value 存储，常用来把大量小图像预编码缓存起来加速训练
- **USM**（Unsharp Mask，反掩模锐化）：传统图像处理里的锐化算子，先对图像做高斯模糊再用原图减去模糊版得到高频，再加回原图
- **chroma subsampling**（色度下采样）：JPEG 在 RGB → YCbCr 转换后把 Cb/Cr 两个色度通道按 4:2:0 等方式降分辨率存储，对人眼不敏感、对模型很敏感

后面用到具体术语时还会再展开一句话定义。

## 5.1 数据 > 网络

第 1 章提到过，2014-2020 年这个领域有一个反复出现的模式：

- 用 bicubic 下采样合成训练数据
- 在合成数据上训出 SOTA 网络
- 实际部署时模型在真实图像上几乎不工作

2021 年的 Real-ESRGAN 论文做了一个对比实验：

| 网络 | 训练数据 | 合成退化 holdout 上的 PSNR | 真实图像视觉 |
|------|---------|--------------|------------|
| ESRGAN（RRDB 网络） | bicubic 退化 | 18.2 dB | 几乎不工作 |
| **Real-ESRGAN（同样的 RRDB 网络）** | 复杂退化合成 | 23.8 dB | 可用 |

网络没变，数据变了，性能就从"几乎不工作"变成了"可用"。这是该领域最重要的一次方法论修正。

类似的故事在去噪、去模糊、视频增强里反复出现：

- DnCNN 在 BSD68（标准基准）上表现极强，但在真实手机图像上失效，用 SIDD 真实数据微调后才大幅提升
- SUPIR 的关键之一是使用了约 2000 万（20M）张高质量互联网图像配合复杂退化合成
- 视频去抖在合成抖动数据上表现理想，落到真实手机视频上仍需补充真实数据

工程上的核心经验是：

> 在影像增强里，数据 pipeline 决定了模型的能力上限。
> 同样的网络在不同数据上能差出一个数量级，反过来则不成立。

这一章讲数据 pipeline 该如何设计。

## 5.2 训练数据的两种来源

### 合成数据

做法是拿一批高质量图像（HR），用代码模拟退化得到（HR, LR）配对。

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

做法是用相机或扫描仪在物理世界采集 (HR, LR) 配对。

实现方式：

- **不同焦段**（RealSR、DRealSR）：同一相机用不同焦段拍同一场景，长焦图作 HR、广角图作 LR
- **不同设备**（DPED）：低端手机与单反同时拍摄同一场景
- **降质模拟**（少见）：拿 HR 走特定物理流程（打印再扫描）得到 LR

优点是真实，所有退化都来自物理过程而非合成假设。

缺点：

- **数量少**：通常只有几百到几千张配对，对数据驱动方法远远不够
- **对齐难**：两次拍摄的物体位置、光线、白平衡都会偏移，需要复杂的配准流程
- **退化模式有限**：采集自哪种相机的哪种退化，模型就只对这一类退化有效

工程实践中的主流做法是：以合成数据预训练为主，再用真实数据微调。

## 5.3 经典 bicubic 流程的问题（再深入一点）

第 1 章已经提过 bicubic 退化的问题，这里再往深处看一层。

一张被用户从相册里随手传上来的低分辨率图像，实际经历过的处理链路大致如下：

```
用户拍照流程:
  传感器 (光子噪声 + 读出噪声)
    → demosaicing (RGB 重建)
    → ISP (降噪 + 锐化 + 白平衡 + tone mapping + 色彩矩阵)
    → JPEG 压缩 (8-bit, quality 70-95)
    → 上传到微信/微博 (再压缩, quality 50-70)
    → 别人下载 (可能再压缩一次)
```

每一步都有自己的退化贡献，最终的 LR 是这些变换的非线性复合。其中 demosaicing 是把传感器 Bayer 阵列上每个像素仅采到单一颜色（R、G 或 B）的数据重建为三通道彩色图的步骤；几乎所有相机内部都做这件事，但不同厂家的插值算法会留下不同的相邻像素相关性。

那么 bicubic 退化模拟了其中哪一步？严格说，一步都没有真正模拟。bicubic 假设 LR 是 HR 经过理想抗混叠滤波后再降采样的结果，整个过程线性、可微、与场景内容无关，与上面这条物理链路没有任何对应关系。

ESRGAN 在 bicubic 数据上学到的恰恰是"反 bicubic 函数"。这个函数在 bicubic 测试集上表现优异，但只要测试图换成手机拍出的真实低分辨率图，模型就会照常执行"反 bicubic"操作：把本不该锐化的噪点也当成高频信号放大，输出一张布满伪影的图像。这种迁移失败不是偶然，而是因为两者的函数空间根本不重叠。

修正这个问题的方向很明确：让训练数据的退化分布覆盖真实世界的退化分布。Real-ESRGAN 的设计正是沿着这个方向展开的，下一节具体拆解。

为了让 §5.4 的代码读起来不抽象，先把这一节的主张画成一张图：左边是真实世界用户拿到一张退化图的物理过程，右边是 Real-ESRGAN 用代码模拟的合成过程，中间用虚线箭头标出合成步骤与物理步骤的对应关系。

```mermaid
graph LR
    subgraph Real["真实世界 (物理链, 一次成像)"]
        S1[光子分布]
        S2[传感器读数<br/>+ shot + read noise]
        S3[demosaicing<br/>+ ISP]
        S4[JPEG quality 70-95]
        S5[网络重压缩<br/>quality 50-70]
        S1 --> S2 --> S3 --> S4 --> S5
    end

    subgraph Synth["合成世界 (二阶退化, 可参数化)"]
        D1[blur 1<br/>各向同性/异性高斯]
        D2[resize 1<br/>area / bilinear / bicubic]
        D3[noise 1<br/>Gaussian + Poisson]
        D4[JPEG 1<br/>quality 30-95]
        D5[blur 2<br/>偏小]
        D6[resize 2]
        D7[noise 2]
        D8[JPEG 2 / sinc]
        D1 --> D2 --> D3 --> D4 --> D5 --> D6 --> D7 --> D8
    end

    S2 -. 模拟 .-> D3
    S3 -. 模拟 .-> D1
    S3 -. 模拟 .-> D2
    S4 -. 模拟 .-> D4
    S5 -. 模拟 .-> D8

    style Real fill:#e3f2fd
    style Synth fill:#fff8e1
```

图里有几条规律值得记住。第一，物理链是一次性的、顺序固定的，合成链是有意打乱顺序、参数随机化的，因为模型要见过的不是某一台相机的退化，而是"所有可能的相机+所有可能的传输链"的退化分布。第二，物理链的某些步骤（如 ISP 降噪和锐化）会在 LR 上留下细节耦合，合成链很难精确复刻，所以 §5.12 才会强调"用真实数据微调"作为最后一公里。第三，合成链里两阶 JPEG 之间的中间产物是带块状伪影的，第二阶 JPEG 会进一步打乱这些块边界，这是 §5.4 里要强调"二阶"的根本原因。

## 5.4 Real-ESRGAN 退化 pipeline 详解

Real-ESRGAN 的 pipeline 由两个核心设计组成：

### 设计一：每个组件参数化与随机化

每一种退化（模糊、下采样、噪声、JPEG）都不是固定的，而是从一个参数分布中随机采样。模型在训练时因此能够见到广泛的退化组合。

### 设计二：二阶退化（high-order degradation modeling）

把整个退化流程串联执行两遍。这是 Real-ESRGAN 的核心创新，也是论文标题中"high-order"的含义。

```
一阶退化 (模拟相机内部):
  HR → blur₁ → downsample₁ → noise₁ → jpeg₁ → 中间产物

二阶退化 (模拟传输/二次压缩):
  中间产物 → blur₂ → downsample₂ → noise₂ → jpeg₂ → LR
```

这样合成出的 LR 就更接近"图片经相机处理后又在互联网上被多次转存压缩"的真实状态。

第二阶的关键参数与第一阶有意拉开差异：

- 第一阶 blur 偏大（模拟实际镜头与传输模糊）
- 第二阶 blur 偏小（避免训练数据糊到无法恢复）
- 第二阶按一定概率加入 **sinc 滤波**（频域为矩形低通、空域产生振荡的滤波器），用来模拟过度锐化引发的振铃伪影；这种伪影在部分图像处理软件或显示链路的输出里十分常见
- 二阶末尾阶段将 **resize + sinc 与 JPEG 的先后顺序**随机化（每个训练 step 从两种顺序中随机选一），而非固定顺序，从而覆盖更多退化组合

把"二阶退化 + 顺序随机化"画成状态图如下：

```mermaid
graph TD
    HR[HR clean image<br/>x in 0,1]
    HR --> B1[blur 1<br/>sigma 0.2-3.0]
    B1 --> R1[resize 1<br/>scale 0.15-1.5]
    R1 --> N1[noise 1<br/>std 1-30/255]
    N1 --> J1[JPEG 1<br/>quality 30-95]
    J1 --> Mid[中间产物]
    Mid --> B2[blur 2<br/>sigma 0.2-1.5]
    B2 --> Choice{随机顺序}
    Choice -->|分支 A| A1[resize + sinc 后 JPEG]
    Choice -->|分支 B| A2[JPEG 后 resize + sinc]
    A1 --> LR[LR degraded]
    A2 --> LR

    style HR fill:#e8f5e9
    style Mid fill:#fff3e0
    style LR fill:#ffebee
```

这张图里需要强调两个工程结论。第一，每个 step 的参数都是从一个明确的区间里随机采样的，所以"训练时模型见到的 $D$ 分布"被显式地写成了代码，而不是模糊地"希望它见过各种情况"。第二，最后阶段两种顺序随机化的意义是让模型见到"先 resize 再 JPEG"和"先 JPEG 再 resize"两种结果，前者更像相机本机处理，后者更像微博转发链。模型如果只见过一种顺序，它会过拟合到该顺序留下的伪影模式。

### 一个简化版的 Real-ESRGAN pipeline

```python
import random
import torch
import torch.nn.functional as F
from typing import Optional


class RealESRGANDegradation:
    """Real-ESRGAN 风格退化的主流程示意 (非完整实现)。
    输入: HR clean image (B, 3, H, W) in [0, 1]
    输出: LR degraded image (B, 3, H/scale, W/scale) in [0, 1]

    与官方 Real-ESRGAN 的差异:
    - JPEG 用占位符 (官方用 diffjpeg)
    - 没有 sinc 滤波 (官方用来模拟过锐化伪影)
    - 模糊核只有各向同性高斯 (官方还有各向异性、广义高斯、plateau)
    - 退化顺序固定 (官方的最终阶段 sinc/resize/JPEG 顺序随机化)
    本节后面会逐项展开这些缺失部分。
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
        """三种噪声按概率选择: 彩色高斯 0.4, 灰度高斯 0.3, 泊松 0.3。"""
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
            from basicsr.utils.diffjpeg import DiffJPEG
            jpeger = DiffJPEG(differentiable=False).to(x.device)
            q = random.randint(*quality_range)
            quality = torch.full((x.shape[0],), q, device=x.device)
            return jpeger(x, quality=quality)
        """
        # 直接返回，不做任何处理（占位符）
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
- USM sharpening（Unsharp Mask，反掩模锐化）反向（模拟手机后处理）

USM 的具体做法是：先对图像做高斯模糊得到 $x_b$，然后用 $x_{usm} = x + \lambda (x - x_b)$ 增强高频，其中 $\lambda$ 控制锐化强度。手机厂商的 ISP 几乎都会内置 USM 风格的锐化，这导致用户拍到的图本身就带轻微"过锐化"特征。Real-ESRGAN 在合成时也加入 USM 步骤，让模型能"识别"这类已经被锐化过一遍的输入，避免在 SR 时再次叠加锐化导致看起来油腻。

但骨架就是上面这个，扩展只是把"采核"这一步从单一函数换成一族函数，把退化顺序从固定换成随机选择。读完 §5.5 到 §5.8 各组件之后，你应该能自己把上面的骨架补全到 production 级别。

## 5.5 模糊核家族

不同退化场景需要不同的模糊核。

### 各向同性高斯（最基础）

$$
k(u, v) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{u^2 + v^2}{2\sigma^2}\right)
$$

只有一个参数 $\sigma$。模拟散焦、大气模糊。$\sigma$ 越大模糊越严重，且模糊在所有方向上都一样。

### 各向异性高斯

$$
k(u, v) \propto \exp\left(-\frac{1}{2}\begin{pmatrix}u\\v\end{pmatrix}^T \Sigma^{-1} \begin{pmatrix}u\\v\end{pmatrix}\right)
$$

协方差矩阵 $\Sigma$ 控制 x/y 方向不同的模糊量。模拟相机抖动、镜头像差。把 $\Sigma$ 写开是：

$$
\Sigma = R(\theta)\begin{pmatrix}\sigma_1^2 & 0 \\ 0 & \sigma_2^2\end{pmatrix} R(\theta)^T
$$

其中 $R(\theta)$ 是旋转矩阵，$\theta$ 是模糊方向，$\sigma_1, \sigma_2$ 是两个主轴上的模糊尺度。三个参数共同决定核的形状（一个角度 + 两个尺度），比各向同性多两个自由度，能模拟"水平模糊比垂直模糊明显"这类常见的相机抖动残留。

### 广义高斯（Generalized Gaussian）

$$
k(u, v) \propto \exp\left(-\left(\frac{u^2}{\sigma_x^2} + \frac{v^2}{\sigma_y^2}\right)^\beta\right)
$$

多一个形状参数 $\beta$：$\beta = 1$ 是标准高斯，$\beta > 1$ 边缘更尖（更接近 box）、$\beta < 1$ 边缘更软。

### Plateau-shaped 核

Real-ESRGAN 还引入了一族 **plateau 核**：中心是平坦的"平台"区域，边缘陡降。它与广义高斯是两个独立的核家族，并非简单地取 $\beta < 1$。Plateau 核更接近散焦圆盘（circle of confusion）的几何形态，常出现在浅景深散焦场景中，普通高斯核无法准确逼近。Real-ESRGAN 在采样训练模糊核时会按概率混合这几族，具体比例可在官方代码的 `degradations.py` 中查到：各向同性高斯 0.45、各向异性高斯 0.25、广义各向同性高斯 0.12、广义各向异性高斯 0.03、plateau 各向同性 0.12、plateau 各向异性 0.03，另有约 0.1 的概率直接改用 sinc 核。

### 运动模糊核

模拟拍摄时相机或被摄物体的直线运动。在曝光时间 $T$ 内，如果传感器相对场景以速度 $v$ 作匀速直线运动，场景中的每个点都会沿运动方向在像面上扫出一条长度为 $vT$ 的线段。把这条线段离散化到像素网格并归一化，就得到最简形式的运动模糊核。它在频域上表现为沿运动方向的 sinc 函数，因此不仅会让边缘模糊，还会在该方向上形成周期性的高频零点。

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

更真实的运动核会让线段带轻微曲率（模拟手抖时的非匀速轨迹），或者把多条不同方向的短线段叠加（模拟相机抖动与物体局部运动的复合）。Real-ESRGAN 官方实现中运动核占比不高，主要作为低概率难样本混入训练分布，避免模型对单一形态的模糊过拟合。

### Sinc 滤波

$$
k(u, v) = \frac{\omega_c}{2\pi r} J_1(\omega_c r), \quad r = \sqrt{u^2 + v^2}
$$

其中 $J_1$ 是一阶 Bessel 函数。Sinc 在频域是理想矩形低通滤波器，但空间域截断后的 sinc 核带有旁瓣振荡，会在强边缘两侧产生振铃（ringing）伪影。

Real-ESRGAN 引入 sinc 滤波，是为了模拟部分图像处理软件（如早期修图工具或社交平台转码链路）过度锐化或插值后留下的振铃。这类伪影在互联网二次传播的图像里很常见，肉眼看似锐利，实则边缘伴随明暗交替的假轮廓。如果模型在训练中没见过 sinc 振铃，遇到这类输入时就会把振铃当成真实高频细节保留甚至继续放大，让输出显得更加生硬。

为了让"模糊核家族"这件事有个直观印象，下面这张图比较各种核在像素域的剖面形状：

```mermaid
graph LR
    A[各向同性高斯<br/>圆形钟形] --> Aa[1 参数 sigma]
    B[各向异性高斯<br/>椭圆钟形] --> Bb[3 参数 sigma1 sigma2 theta]
    C[广义高斯<br/>钟形可调尖度] --> Cc[加形状参数 beta]
    D[Plateau<br/>中心平台 边缘陡降] --> Dd[模拟散焦圆盘]
    E[运动核<br/>线段] --> Ee[长度和方向]
    F[Sinc<br/>振荡环] --> Ff[模拟过锐化振铃]

    style A fill:#e3f2fd
    style B fill:#e3f2fd
    style C fill:#fff8e1
    style D fill:#fff8e1
    style E fill:#ffebee
    style F fill:#ffebee
```

颜色分组反映"难度"：左侧三族是高斯家族，参数少、采样便宜；中间两族需要额外几何参数，但仍可批量在 GPU 上生成；右侧两族（运动、sinc）形状特殊、对模型来说最难，需要专门采样且不能用单一高斯近似。

## 5.6 下采样的多种策略

不同的插值核在频域行为差别巨大：

| 算法 | 频域行为 | 视觉特点 | 何时用 |
|------|---------|---------|-------|
| nearest | 矩形（高频泄漏严重） | 锯齿 | 模拟极差的处理 |
| bilinear | 三角形 | 略糊 | 均衡选择 |
| bicubic | 接近 sinc | 锐利但有 ringing | 学术标准 |
| lanczos | 截断 sinc | 最锐利、振铃明显 | 模拟某些软件输出 |
| area / box | sinc（空间域是矩形均值） | 柔和无锯齿 | 大幅缩小时 |

Real-ESRGAN 的 pipeline 在每次 resize 时从上述插值方式中随机选一种，让模型覆盖不同插值核留下的频域特征。

从频域角度看，这一点尤为关键：bicubic 接近理想低通，会把高频干净地滤除；nearest 完全不做低通滤波，留下大量混叠；area 将每个 LR 像素取为对应 HR 块的均值，等价于空域 box filter，频域响应为 sinc，会抹平细碎纹理，但对粗大边缘保留较多。如果模型在训练时只见过 bicubic 生成的"干净低频图"，面对真实场景中经 area 或 bilinear 缩放的图像时，就会把残留混叠误判为真实高频信号并加以放大，结果把锯齿做得更刺眼，而不是恢复出自然细节。

工程上还有一个容易踩坑的细节：`torch.nn.functional.interpolate` 在 `align_corners=False` 时与 OpenCV `cv2.resize` 的默认设置基本对齐；若设为 `True`，则对应早期 TensorFlow 的角点对齐方式，两者在边缘 1–2 个像素处会有可测量的偏差。线上推理所用的缩放实现应当与训练时保持一致，否则容易出现"训练用 PyTorch bicubic、部署调 ImageMagick catmull-rom"之类的隐蔽分布偏移。

## 5.7 噪声的实现细节

第 1 章已经讲过噪声的物理模型，这里补充几个工程实现上的细节。

### 灰度噪声 vs 彩色噪声

真实噪声的通道相关性需要从传感器与 ISP 两个层面理解：

- 高端相机（全画幅单反、中画幅）：像元尺寸大、信噪比高，单帧噪声接近各通道独立同分布，RGB 三通道之间相关性很弱
- 手机摄像头：像元小、感光度常拉得很高，经过 demosaicing（用邻近像素插值补齐缺失颜色）后，相邻像素的噪声会产生空间相关；随后的色彩校正矩阵（CCM）又把不同通道的噪声线性混合，引入跨通道相关
- ISP 降噪后：机内降噪算法通常会优先抹掉刺眼的彩色噪点（chroma noise），残留下来往往是三通道同步起伏的亮度噪声（luma noise），视觉上像一层细灰雾

工程实践上，训练数据通常将两者混用：约一半概率加三通道独立的彩色噪声（模拟轻度 ISP 处理或近 RAW 状态），另一半概率加灰度噪声（三通道共享同一张噪声图，模拟 ISP 降噪后的残留）。更精细的实现还会按一定比例混入空间相关噪声（先对白噪声做小核高斯模糊再叠加到图像上），用来逼近 demosaicing 之后的空间相关形态。

### 噪声强度分布

噪声标准差 $\sigma$ 不宜均匀采样，因为真实场景里轻度噪声远比极端高噪常见。

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

这个物理噪声模型对手机暗光去噪的数据合成尤为关键。

## 5.8 JPEG 压缩

JPEG 是退化合成 pipeline 里最容易被忽略、却又至关重要的一步。如果训练数据里没有包含 JPEG 压缩，模型在面对任何经过互联网传输的图像（几乎占真实输入的绝大多数）时都会失效，因为 8×8 块效应与振铃对它而言是完全陌生的分布。

JPEG 编解码的核心流程如下，有助于理解为什么"DCT 量化步"不可微：

1. RGB 转 YCbCr，将亮度通道 Y 与色度通道 Cb/Cr 分离
2. 对 Cb/Cr 做色度下采样（最常见的是 4:2:0，水平与垂直分辨率各减半）
3. 每个 8×8 像素块独立做 DCT，得到 64 个频域系数
4. 用由 quality 决定的量化表对频域系数做除法并取整
5. 按 zig-zag 顺序扫描并做熵编码落盘

第 4 步的"取整"将连续实数映射为离散整数，导数处处为零或无定义。若要在端到端训练中让梯度穿过 JPEG，就必须用可微近似替代硬取整（例如 straight-through estimator 或分段多项式近似），这正是 DiffJPEG 库所做的事情。

合成 pipeline 中的 JPEG 模块需要满足两点：

1. **能在 GPU 上批量执行**（不能每张图走 CPU 编码存盘再读回），否则磁盘与 CPU 编解码会直接卡死训练吞吐
2. **可选支持可微反传**（虽然常规退化合成在 `no_grad` 下运行即可，但若后续要在压缩后图像上计算损失并反传梯度，就需要可微模式）

工程上推荐使用 **DiffJPEG**（`basicsr` 已内置可直接调用的实现，原始仓库见 [github.com/mlomnitz/DiffJPEG](https://github.com/mlomnitz/DiffJPEG)）：

```python
from basicsr.utils.diffjpeg import DiffJPEG

# 构造一次, 不在构造时传 quality; 是否可微在构造时决定
jpeger = DiffJPEG(differentiable=False).to(device)    # 推理用
# jpeger = DiffJPEG(differentiable=True).to(device)    # 训练时想反传就换这一行

def random_jpeg_diffjpeg(x: torch.Tensor, quality_range=(30, 95)) -> torch.Tensor:
    # quality 在 forward 时传入, basicsr 版接受一个 (B,) 的 quality 张量
    q = random.randint(*quality_range)
    quality = torch.full((x.shape[0],), q, device=x.device)
    return jpeger(x, quality=quality)
```

### 第二阶 JPEG 的特殊性

Real-ESRGAN 的二阶 JPEG 模拟"已经压缩过的图被再次压缩"。这种情况下：

- 第一阶 JPEG 的块状伪影会被第二阶 JPEG 进一步混乱
- 两次 JPEG 的 8×8 块边界一般不重合，因为中间夹了一次 resize，导致复杂的非对齐复合伪影
- 真实"网络重传"图像里这种现象普遍存在，你保存的微博图被别人重转后，块边界已经"漂移"过一次

直接串联调用两次 JPEG 就能复现这种现象。需要注意的细节是：第二阶 JPEG 的 quality 未必比第一阶更低，反过来在真实链路里也很常见（例如手机原图经社交软件以 quality 70 压缩，用户保存时又以 quality 90 重新封装）；关键在于两次 8×8 量化网格因中间隔了一次缩放而不再对齐，而不在于 quality 数值本身的高低。

### 关于 chroma subsampling 的额外提醒

JPEG 的 4:2:0 色度下采样会把 Cb/Cr 通道的分辨率各砍一半。人眼对色度分辨率下降不敏感，但神经网络会敏锐捕捉到"亮度边缘比色度边缘更锐利"的统计规律。如果目标是修复经过社交网络压缩的图像，训练时的 JPEG 模拟必须保留 chroma subsampling；如果目标是修复高保真专业摄影的轻度压缩痕迹，则可按需关闭。DiffJPEG 默认启用 4:2:0。

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

学术标准的好处是结果可比，缺点是 benchmark 都很小（Set5 真的只有 5 张），结论的统计意义有限。真实导向的好处是反映生产分布，缺点是没有标准化测试集让你和别人的论文做横向比较，你只能拿自己的真实 holdout 集做内部对比。生产环境里两套都跑一遍是常见做法：学术 benchmark 用来确认"没把基础能力训差"，真实测试集用来确认"对得起这次部署"。

## 5.10 数据增广

低层视觉的数据增广与高层分类任务差别很大。下面是不会破坏退化分布的安全增广操作：

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

需要慎用甚至避免的增广包括：

- **色彩抖动**：会改变退化分布，让模型学到错误的颜色映射；对于真实退化模型而言，白平衡与色彩偏移本身属于退化算子 $D$ 的一部分，不应作为随意扰动的增广维度
- **任意角度旋转**：双线性或双三次插值会引入额外的重采样模糊，不在设定的退化算子 $D$ 之内，容易让模型学到带插值伪影的输出
- **随机缩放**：等价于改变超分倍率（scale factor），训练超分模型时不宜对配对数据同步做非整数缩放，以免混淆任务定义中的尺度关系
- **mixup / cutmix**：低层视觉里极少使用；这类增广依赖"语义标签对线性混合满足光滑过渡"的假设，而在逐像素回归里混合两张自然图像只会制造非物理的半透明叠影

裁剪策略如下：

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

把退化 pipeline 移到 GPU 上，整个 batch 一次过。这个方案乍看违反直觉：通常我们把数据预处理放在 CPU 上让 GPU 专心做矩阵乘。但是退化合成里 blur、JPEG、resize 这几步本身就是大量卷积和频域变换，跑在 GPU 上比 CPU 快 20× 以上，把它们留在 CPU 反而拖垮整个训练。

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

为了让"CPU 读 + GPU 退化"两边都不空转，整个数据流应该长成下面这样：

```mermaid
graph LR
    Disk[HR PNG / LMDB] --> CPU[CPU workers<br/>read + crop + to_tensor]
    CPU --> Queue[pinned memory queue]
    Queue --> GPU1[GPU: degradation<br/>blur/resize/noise/JPEG]
    GPU1 --> GPU2[GPU: model forward]
    GPU2 --> GPU3[GPU: loss + backward]
    GPU3 --> GPU4[GPU: optimizer step]
    GPU4 -.next batch.-> Queue

    style Disk fill:#e3f2fd
    style CPU fill:#fff8e1
    style GPU1 fill:#ffebee
    style GPU2 fill:#ffebee
    style GPU3 fill:#ffebee
    style GPU4 fill:#ffebee
```

这张图里有两个工程窍门容易被忽视。第一，pinned memory queue 是 PyTorch 的 `pin_memory=True` 开启的固定内存缓冲区，让 CPU 到 GPU 的拷贝可以走异步 DMA，不阻塞主进程；第二，退化步骤要写成"batch 级 + 完全 GPU 可执行"，任何不小心写成 Python for-loop 或者 `.cpu()` 的代码都会让训练 throughput 掉一个数量级。

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

不直接在真实数据上从头训的原因：数据太少（几百张），模型容易过拟合到这几百张的具体物体和场景。这是一个典型的 catastrophic specialization 陷阱：模型在训练集物体上 PSNR 涨得飞快，但一旦换一类场景，表现就急剧变差。两阶段训练法本质上是用合成数据先把模型的"先验"打牢，让真实数据只负责微小的分布对齐，而不必从零学习"自然图像应该长什么样"。

微调阶段还有几个工程细节需要注意。学习率不仅要比 pretrain 小一个数量级，最好也用 cosine schedule 让它在最后几千步降到几乎为零，避免模型在小数据上反复震荡。EMA（Exponential Moving Average，对权重做指数移动平均的影子副本）在微调阶段尤其重要，它能进一步把训练抖动平滑掉。最后，验证集要用与微调集**不同来源**的真实图（比如微调用 RealSR，验证用 DRealSR），否则你看到的提升完全可能是过拟合。

## 5.13 数据质量审计

在启动正式训练前，务必先审计训练数据本身的质量：

### HR 数据是否真正高清

DIV2K 等标准数据集的 HR 图像整体质量虽高，但仔细检查仍能发现：

- 轻微的 JPEG 压缩痕迹
- 暗部残余噪点
- 局部过曝或欠曝

如果 HR 真值本身带有压缩伪影，模型就会把"输出带轻微块效应"当成优化目标，所谓的 ground truth 也就不再干净。

可以用下面的简单脚本筛查 HR 数据集中的 8×8 块边界伪影：

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

这种偏差不会被指标显示出来：平均 PSNR 看起来正常，但失败模式集中。第 17 章会专门讲。

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

### 不平衡退化分布的隐患

合成数据的另一个隐藏问题是退化参数本身的分布。假设你的 blur sigma 范围是 [0.2, 3.0]，看似合理，但如果你用均匀采样，那么 [2.5, 3.0] 区间的样本和 [0.2, 0.7] 区间的样本数量相同，而实际真实世界的模糊几乎都集中在小 sigma 端。这会导致模型在"轻微模糊"输入上反应过度（把本来清晰的小细节当成需要去模糊的对象），在"严重模糊"输入上反而表现不错。

工程上常见的修正是把均匀采样换成对数采样或者 beta 分布采样：

```python
import math
import random

def log_uniform(low: float, high: float) -> float:
    """对数均匀分布, 让小值更常被采到。"""
    return math.exp(random.uniform(math.log(low), math.log(high)))

# 用法
sigma = log_uniform(0.2, 3.0)
```

类似的修正适用于噪声 sigma、JPEG quality 等所有"退化强度"参数。在生产中观察到模型表现异常时，第一步排查就是看训练参数分布是否和你 deploy 时遇到的真实分布对得上。

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

到这里 Part I 的五章就全部结束了。读完这五章，你应当能够：

- 看到一张退化图像，拆解出它经历了哪些退化算子 $D$
- 看到一个增强模型，判断它在什么表征空间做预测、由哪些损失函数驱动
- 看到一组评估指标，分辨哪些可信、哪些存在盲区
- 看到一条数据合成 pipeline，判断它能覆盖真实退化分布的哪一部分

从 Part II 开始，我们将进入具体的神经网络架构演进。第一站是 CNN 时代：从 2014 年的三层 SRCNN 一路走到 2022 年去繁就简的 NAFNet，这是低层视觉最扎实的工程基本功。

---

> 下一章 [CNN 时代](06-cnn.md) → 从 SRCNN 三层网络到 NAFNet 移除所有非线性激活，CNN 在低层视觉走过的 8 年。
