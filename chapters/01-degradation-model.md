# 第 1 章 · 退化模型与逆问题

> "增强"是一种说法。本书用更准确的词：**估计**。

## 1.1 一个具体的场景

晚上九点，你在街上举起手机拍了一张霓虹灯下的招牌。回家一看，画面有这些问题：

- 整体偏暗，亮部过曝、暗部漆黑
- 字的边缘糊成一团
- 阴影里有一颗颗紫绿色的小点（噪点）
- 招牌的细线纹有奇怪的彩色波纹（摩尔纹 + JPEG 块）

你在修图 App 里点了一下"AI 增强"，三秒后画面看起来好多了。

这一节回答一个问题：**这三秒里，模型到底在做什么？**

它不是"找回了原本的画面"——那张原本的画面，从信息论上说，已经永远丢失了。它是在**根据这张烂图，估计一张最可能产生这张烂图的高质量图**。这是这本书的第一个反直觉的点：

> 增强模型不在"恢复"，它在"猜"。差别在于猜得有多合理。

理解这一点，后面所有架构、损失、评估、数据合成的设计选择就都有了同一个出发点。

## 1.2 中心方程

把上面这件事写成数学：

$$
y = D(x) + n
$$

- $x$：理想的高质量图像（你晚上**应该**拍到的那张）
- $D$：退化算子（光线、相机、镜头、传感器、压缩共同作用的"破坏机器"）
- $n$：噪声（随机扰动，主要来自传感器）
- $y$：你手里这张实际拍到的烂图

整本书做的事情，就是**给定 $y$，估计 $x$**。

记作 $\hat{x} = f_\theta(y)$，其中 $f_\theta$ 是我们要训的模型，参数是 $\theta$。

这个看起来简单的方程，藏着这个领域所有的工程难题：

1. $D$ 通常**未知**（你拍照时不知道当时镜头多脏、传感器多热、JPEG quality 多少）
2. $D$ 是**多对一**的（无穷多张 $x$ 经过 $D$ 都能产生这张 $y$）
3. $n$ 是**随机的**（同一张 $x$ 加不同的 $n$ 得到不同的 $y$）
4. 数据集里我们通常**只有 $y$**，没有 $x$（真实低质量图配真实高质量图的成对数据极少）

第 1、2 点决定了这是一个**逆问题**（inverse problem），第 3 点说明它是一个**统计逆问题**，第 4 点决定了我们的训练范式必须**自己合成 $(x, y)$ 配对**——这就是第 5 章要详细讲的退化合成。

## 1.3 为什么这是 ill-posed 问题

数学家用三个词描述一个良好的（well-posed）问题：

1. **解存在**
2. **解唯一**
3. **解对输入连续依赖**（输入小扰动导致输出小扰动）

逆问题往往三条都不满足，称作 **ill-posed**。影像增强是典型的 ill-posed 问题。

举一个最直观的例子：4 倍下采样的超分辨率。

你有一张 $512 \times 512$ 的低分辨率图 $y$，要恢复成 $2048 \times 2048$ 的高分辨率 $x$。也就是说，原图里的 $4 \times 4 = 16$ 个像素，被某种方式压成了 $y$ 里的 1 个像素。

具体压成什么？最简单的 bicubic 下采样，每个低分辨率像素是 16 个高分辨率像素的加权平均（权重由 bicubic 核决定）。这是一个 16 维空间到 1 维空间的映射。**16 维降到 1 维，丢掉 15 维信息**。

反过来想：给你这一个像素值 $y_{ij}$，问哪些 $4 \times 4$ 的高分辨率块能产生它？答案是一个 15 维的解空间——里面有无穷多个合理的高分辨率块。

考虑一个具体的低分辨率像素 $y_{ij} = 128$（中等灰度）：

- 它可能是一块均匀灰色（高分辨率块全是 128）
- 也可能是黑白条纹（一半 0 一半 255，平均仍是 128）
- 也可能是一段斜坡（从 100 渐变到 156）
- 也可能是中间有一个小亮点（15 个 100 加 1 个 548... 不对，亮点不能超过 255，但可以是 15 个 120 加 1 个 248）

所有这些不同的高分辨率块经过下采样都给你同一个 128。模型必须**选一个**输出——选哪一个？

这就是先验（prior）发挥作用的地方。

## 1.4 三种"先验"

先验 = **关于"自然图像应该长什么样"的假设**。一个好的增强模型，本质上是把好的先验编码进了网络的归纳偏置或者训练数据里。

历史上有三种来源：

### 解析先验（手工设计）

人类工程师根据观察总结出"自然图像的统计规律"，写成约束。最经典的几个：

- **平滑性**：相邻像素值相近（用梯度的 L2 范数惩罚）
- **稀疏梯度**（Total Variation）：自然图像的梯度大多数地方接近零、少数地方很大（边缘）。用梯度的 L1 范数惩罚。
- **小波/DCT 稀疏性**：自然图像在小波/DCT 域稀疏

这一类方法的代表是 1990s-2010s 主导这个领域的**变分方法**和**稀疏编码**。它们**简洁、可解释、不需要训练**——但效果上限非常低。原因：手工先验太弱，无法编码"这个区域是人脸还是猫毛"这种内容相关的知识。

我们现在不用这些方法做主力，但它们的思想还活着。比如第 3 章会讲到的 TV 损失：

```python
def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    """Total Variation：惩罚相邻像素差，鼓励分段平滑。
    x: (B, C, H, W)
    """
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw
```

### 数据驱动先验（CNN/Transformer 学到的）

2014 年 SRCNN 之后，深度学习取代了变分方法。一个 CNN 端到端从 $y$ 映射到 $\hat{x}$，先验隐式地编码在权重里。模型在大量自然图像 $(x, y)$ 配对上训练，学到的是"自然图像的分布在权重空间里的低维流形"。

这一类的特点：**判别式（discriminative）**——给一个 $y$ 输出一个 $\hat{x}$。模型不显式建模 $p(x)$，但在大数据集上训练后，输出的 $\hat{x}$ 自然落在自然图像流形上。

代表：SRCNN、EDSR、RCAN、SwinIR、Restormer、NAFNet（第 6-7 章详讲）。

### 生成式先验（扩散模型）

2020 年 DDPM 之后，生成模型本身就显式地建模了 $p(x)$。给定 $y$，可以做条件生成 $p(x | y)$，从这个条件分布里采样出 $\hat{x}$。

这是**生成式（generative）**路线。和判别式的本质区别：

- 判别式回答"最可能的 $\hat{x}$ 是什么"——产出**一个**确定答案
- 生成式回答"$x | y$ 的分布是什么"——可以采样出**多个**合理答案

生成式的优势：在 ill-posed 严重的场景（比如 8 倍超分、人脸高度模糊），判别式只能输出一个"平均脸"，生成式能给一个高频细节合理的具体脸。

劣势：它会**编造**。这会引出本书一个反复强调的工程哲学（第 10、17 章详谈）：

> 增强模型在生成细节，不在恢复细节。
>
> 用扩散模型修老照片，效果惊艳；用扩散模型修法医证物照片，是事故。

## 1.5 退化算子 D 的解剖

回到中心方程 $y = D(x) + n$。$D$ 在真实世界里不是一个简单算子，而是一串复合：

$$
D = \text{JPEG} \circ \text{Quantization} \circ \text{Downsample} \circ \text{Blur} \circ \text{ColorShift} \circ \text{LensDistortion} \circ \dots
$$

而且这串复合算子本身是**随机的**——同一台手机不同时刻拍同一个场景，得到的 $y$ 都不完全一样。

工程上我们关心的主要 $D$ 组件：

### 模糊（Blur）

模糊在数学上是和一个"模糊核" $k$ 做卷积：$y = x * k$。

不同的物理来源对应不同的模糊核：

- **散焦模糊**（defocus）：核接近圆盘形，半径取决于失焦程度
- **运动模糊**（motion blur）：核是一条线段，长度和方向取决于相机或物体的运动
- **大气模糊**（atmospheric）：核接近高斯，方差取决于大气湍流强度
- **镜头衍射 / 像差**：核接近 Airy disk 或更复杂的形状

实际拍到的图通常是几种模糊的复合，再加上空间变化（图像中心和边角的模糊核可能不同）。这个复杂性是**盲去模糊**（blind deblur）这个子方向独立存在的原因——核未知。

### 下采样（Downsample）

把高分辨率图变成低分辨率。常见算法：

- **Nearest**：直接取最近的像素。最快、最差，会产生锯齿
- **Bilinear**：双线性插值，2×2 邻域加权平均
- **Bicubic**：双三次插值，4×4 邻域加权平均，是这个领域**事实标准的训练用下采样**
- **Lanczos**：基于 sinc 函数的截断滤波，频域响应最接近理想低通
- **Box / Average**：每个低分辨率像素是对应高分辨率块的简单平均

不同算法的频域行为差别很大。Bicubic 是个**理论上不错但工程上有问题**的选择——它假设原图无混叠（aliasing-free），但真实拍到的图大多带混叠。这就是 1.6 节要讲的"训练-推理失配"。

```python
import torch
import torch.nn.functional as F

def downsample_compare(x: torch.Tensor, scale: int = 4):
    """对同一张图用不同方法下采样,看差异。
    x: (1, C, H, W), 值域 [0, 1]
    """
    h, w = x.shape[-2:]
    new_h, new_w = h // scale, w // scale

    nearest  = F.interpolate(x, size=(new_h, new_w), mode='nearest')
    bilinear = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
    bicubic  = F.interpolate(x, size=(new_h, new_w), mode='bicubic',  align_corners=False)
    area     = F.interpolate(x, size=(new_h, new_w), mode='area')  # 等价于 box filter

    return {
        'nearest':  nearest,
        'bilinear': bilinear,
        'bicubic':  bicubic,
        'area':     area,
    }
```

实测一下你会发现：bicubic 在视觉上"最锐利"，area（box）在视觉上"最柔和"，nearest 满是锯齿。**没有哪个是"对的"**——它们各自模拟了不同的物理过程。

### 噪声（Noise）

噪声是这一节最容易被简化、也最值得讲透的部分。1.7 节会单独深挖。

### 压缩伪影（Compression Artifacts）

JPEG 是图像领域最常见的压缩。它的工作流程：

1. RGB → YCbCr，对色度通道下采样（chroma subsampling）
2. 分成 8×8 块
3. 每块做 DCT 变换
4. 量化（quality 越低，量化越粗）
5. 熵编码

第 4 步的量化是有损的，重建后的图会有：

- **块状伪影**（blocking artifacts）：8×8 块边界可见
- **振铃**（ringing）：高对比边缘附近的振荡
- **彩色失真**：色度下采样造成的彩色边缘

视频压缩（H.264/H.265/AV1）类似但更复杂——还有运动补偿带来的拖影。

### 颜色失真

白平衡漂移、颜色衰减、低光导致的色温偏移。这一类退化在数学上不是空间维度的问题，而是值域上的非线性变换。

## 1.6 真实世界退化与训练-推理失配

把 1.5 的所有组件串起来，一张你晚上拍的图大致经过了：

$$
y = \text{NetworkRecompression}(\text{JPEG}(\text{Quantize}(\text{ISP}(\text{Sensor}(x_{\text{photon}})))))
$$

里面：

- $x_{\text{photon}}$ 是落到传感器上的光子分布（这才是真正的"原始信号"）
- $\text{Sensor}$ 把光子变成电信号，加进光子噪声、读出噪声、暗电流
- $\text{ISP}$ 是相机里的图像信号处理器，做去马赛克、降噪、白平衡、tone mapping、色彩矩阵
- $\text{Quantize}$ 是 8 位量化（高端相机 10-12 位）
- $\text{JPEG}$ 是相机存储时的有损压缩
- $\text{NetworkRecompression}$ 是图片在微信/微博/Twitter 等平台传输时**又一次**的有损压缩

这个长链条是**真实世界的 D**。

而过去十年绝大多数学术论文，包括 SRCNN、EDSR、RCAN、ESRGAN、SwinIR，训练数据都是这样合成的：

```python
# 学术论文的"经典"训练对合成
y = bicubic_downsample(x, scale=4)   # 完了
```

这就是这个领域 2014-2020 年最大的方法论问题，叫做 **train-test mismatch**：

- **训练**：$y$ 是干净的高分辨率图直接 bicubic 下采样
- **推理**：$y$ 是真实手机拍的、被传感器噪声+ISP+JPEG+网络重压缩的图

模型在训练时见过的"低质量图"完全不是真实世界的低质量图。结果就是：

- ESRGAN 在 Set5/Set14（学术 benchmark）上 PSNR 30+，效果惊艳
- ESRGAN 在你手机里的真实照片上，几乎不工作——甚至把噪点放大成糖纸

这个失配持续了五六年，直到 2021 年的 **Real-ESRGAN** 才被系统性地解决。它的核心贡献**不是网络结构**（用的还是 RRDB），而是**退化合成 pipeline**：

```python
# Real-ESRGAN 风格的退化(简化版,真实代码在第 5 章详讲)
y = x.clone()
y = apply_blur(y, kernel=random_blur_kernel())      # 模糊
y = downsample(y, scale=random_scale(), mode=random_mode())  # 多种下采样
y = add_noise(y, type=random_noise_type())          # 高斯+泊松+真实传感器噪声
y = jpeg_compress(y, quality=random.randint(40, 95))  # JPEG
# 关键: 整个流程做两遍 (second-order degradation)
y = apply_blur(y, ...); y = downsample(y, ...); y = add_noise(y, ...); y = jpeg_compress(y, ...)
```

这个 pipeline 的每一步都在"扮演"真实世界 D 的某个组件。**训出来的模型在真实图像上的表现是质变级别的**——这是 2021 年这个领域的分水岭事件。

记住这一点：

> 影像增强领域的"模型架构"和"数据合成"，长期以来后者更重要。
>
> 同样的网络，bicubic 数据训出来不能用，Real-ESRGAN pipeline 数据训出来好用。

第 5 章会完整展开退化合成的工程细节。

## 1.7 噪声深挖：为什么不是高斯

90% 的论文用高斯噪声做训练加噪：

```python
y = x + torch.randn_like(x) * sigma
```

这是**错的**——准确说，是**对真实场景错了的简化**。

真实传感器噪声有两个主要来源：

### 光子噪声（shot noise）

光子到达传感器是一个**泊松过程**。如果某个像素位置在曝光时间内平均接收 $N$ 个光子，那实际接收数量服从 $\text{Poisson}(N)$，方差也是 $N$。

关键性质：**方差等于均值**。亮的地方光子多，绝对噪声大；暗的地方光子少，绝对噪声小。但**信噪比** SNR $= N / \sqrt{N} = \sqrt{N}$，亮的地方 SNR 更高。

这就是为什么晚上拍的暗部噪点严重——光子少，相对噪声大。

### 读出噪声（read noise）

传感器把电荷转成电压、再 ADC 量化，每一步引入电子噪声。这部分**近似高斯**，与信号无关，是个常数 $\sigma_r$。

### 真实模型：信号相关高斯

把两者合起来近似为一个信号相关的高斯：

$$
y = x + \mathcal{N}(0, a \cdot x + b)
$$

其中 $a$ 控制光子噪声强度（与信号相关），$b$ 控制读出噪声（与信号无关）。$a, b$ 是相机的物理参数，不同 ISO 下不同。

```python
import torch

def heteroscedastic_noise(
    x: torch.Tensor,
    a: float = 0.01,
    b: float = 0.001,
) -> torch.Tensor:
    """信号相关的高斯噪声(近似真实传感器)。
    x: (B, C, H, W), 值域 [0, 1]
    a: 光子噪声系数 (signal-dependent)
    b: 读出噪声方差 (signal-independent)
    返回: 加噪后的 y, 仍在 [0, 1]
    """
    variance = a * x + b
    sigma = variance.clamp(min=1e-8).sqrt()
    noise = torch.randn_like(x) * sigma
    return (x + noise).clamp(0.0, 1.0)
```

更精确的模型是**直接采样泊松分布**：

```python
def poisson_gaussian_noise(
    x: torch.Tensor,
    photon_scale: float = 1000.0,  # 越小越暗,噪声相对越大
    read_sigma: float = 0.005,
) -> torch.Tensor:
    """泊松+高斯,真实传感器物理模型。
    photon_scale 模拟曝光,值越小代表暗光场景。
    """
    # 把信号缩放到光子数量级,采样泊松,再缩回来
    photons = x * photon_scale
    noisy_photons = torch.poisson(photons.clamp(min=0))
    shot = noisy_photons / photon_scale
    # 加读出噪声
    read = torch.randn_like(x) * read_sigma
    return (shot + read).clamp(0.0, 1.0)
```

为什么这件事重要？因为：

- 用纯高斯噪声训出来的去噪模型，在真实手机暗光场景下偏弱
- 它学到的是"均匀噪声"的统计，但真实噪声**亮处大暗处小**
- 真实图像在暗部需要更激进的去噪，亮部需要更保守的去噪——纯高斯训出来的模型做不到

这个细节是 SIDD、DND 这些"真实噪声数据集"出现的动因。它们直接用真实相机在控制环境下拍噪声-干净对，绕过合成。

但真实数据贵且稀少，工程上的妥协是用**精确的合成噪声模型**，再用少量真实数据微调。

## 1.8 简化的 Degradation 类（不含 JPEG）

最后把这一章的概念串成一个 Real-ESRGAN 风格的退化类骨架。**这是预热版**——为了保持可读性，**省略了 JPEG 步骤**（真实 JPEG 需要 `diffjpeg` 库，第 5 章详谈），只保留 blur / downsample / noise 三步。完整版本在第 5 章。

```python
import random
import torch
import torch.nn.functional as F


class SimpleDegradation:
    """Real-ESRGAN 风格的简化退化合成。
    输入高分辨率干净图 x, 输出低分辨率烂图 y。

    完整的 D = JPEG ∘ Noise ∘ Downsample ∘ Blur
    每个组件参数都随机化, 模拟真实世界 D 的不确定性。
    """

    def __init__(self, scale: int = 4):
        self.scale = scale

    # --- 1. 模糊 ---
    def random_blur(self, x: torch.Tensor) -> torch.Tensor:
        """随机选择高斯模糊核。"""
        ksize = random.choice([7, 9, 11, 13, 15])
        sigma = random.uniform(0.2, 3.0)
        kernel = self._gaussian_kernel(ksize, sigma).to(x)
        kernel = kernel.expand(x.shape[1], 1, ksize, ksize)
        pad = ksize // 2
        return F.conv2d(F.pad(x, [pad]*4, mode='reflect'),
                        kernel, groups=x.shape[1])

    @staticmethod
    def _gaussian_kernel(ksize: int, sigma: float) -> torch.Tensor:
        ax = torch.arange(ksize) - ksize // 2
        gauss = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
        kernel = gauss[:, None] * gauss[None, :]
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)

    # --- 2. 下采样 ---
    def random_downsample(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        new_h, new_w = h // self.scale, w // self.scale
        mode = random.choice(['bilinear', 'bicubic', 'area'])
        kw = {'mode': mode}
        if mode in ('bilinear', 'bicubic'):
            kw['align_corners'] = False
        return F.interpolate(x, size=(new_h, new_w), **kw)

    # --- 3. 噪声 ---
    def random_noise(self, x: torch.Tensor) -> torch.Tensor:
        """三种噪声各 1/3 概率, 模拟不同传感器/场景。"""
        p = random.random()
        if p < 0.33:
            sigma = random.uniform(0.005, 0.05)
            return (x + torch.randn_like(x) * sigma).clamp(0, 1)
        elif p < 0.66:
            a = random.uniform(0.005, 0.03)
            b = random.uniform(0.001, 0.005)
            sigma = (a * x + b).clamp(min=1e-8).sqrt()
            return (x + torch.randn_like(x) * sigma).clamp(0, 1)
        else:
            scale = random.uniform(50.0, 1000.0)
            photons = (x * scale).clamp(min=0)
            return (torch.poisson(photons) / scale).clamp(0, 1)

    # --- JPEG 在第 5 章用 diffjpeg 实现, 这里省略 ---

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """简化退化, 顺序: blur -> downsample -> noise (无 JPEG)。"""
        y = self.random_blur(x)
        y = self.random_downsample(y)
        y = self.random_noise(y)
        return y
```

实际工程里还要加：

- **二阶退化**（second-order degradation）：把这个流程跑两遍。Real-ESRGAN 的关键技巧之一，模拟"图被压缩-传输-再压缩"。
- **退化顺序随机化**：blur 和 noise 谁先谁后随机。
- **可微 JPEG**：用 `diffjpeg` 这种库，能在退化阶段反传梯度（虽然增强训练里很少需要）。
- **更复杂的模糊核**：广义高斯、运动核、混合核（generalized Gaussian、motion kernels）。

第 5 章会把这些都补上。

## 1.9 本章小结

把这一章压缩成几条：

1. **影像增强本质是逆问题**：从 $y$ 估计 $x$，给定 $y = D(x) + n$。
2. **它是 ill-posed 的**：$D$ 是多对一映射 + $n$ 是随机的，解空间无穷大。
3. **模型靠先验做选择**：解析先验弱、数据驱动先验强、生成式先验最强但会编造。
4. **真实退化是一个长复合**：模糊 + 下采样 + 噪声 + 压缩 + 网络重压缩。
5. **训练-推理失配是这个领域的核心方法论问题**：bicubic 训出来的模型在真实图像上崩。
6. **Real-ESRGAN 的核心贡献是数据**：复杂退化 pipeline + 二阶退化。
7. **真实噪声不是高斯**：是泊松+高斯的混合，且方差与信号相关。

剩下 17 章都是在回答同一个问题：**如何在 ill-posed 的约束下，估计出最合理的 $\hat{x}$？**

- 第 2 章：在哪个空间做这件事（像素 / 特征 / 潜空间）
- 第 3 章：用什么损失函数衡量"合理"
- 第 4 章：用什么指标评估"合理"
- 第 5 章：如何合成训练数据
- 第 6-10 章：用什么模型架构编码先验
- 第 11-12 章：怎么训得稳、评得准
- 第 13-14 章：视频场景的额外约束（时序一致）
- 第 15-17 章：怎么把模型部署到真实世界
- 第 18 章：现在用谁

在每一章读到具体技术决策时，回到这一章问自己：**这个设计在改进 $D$ 的建模、$x$ 的先验、还是 $\hat{x}$ 的搜索？** 这本书的所有内容都在这个三角形里。

---

> 下一章 [像素、特征、潜空间](02-representation.md) → 我们将看到，为什么所有现代增强方法都不在像素空间直接工作。
