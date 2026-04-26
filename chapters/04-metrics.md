# 第 4 章 · 评估的陷阱

> 影像增强领域有一个公开的秘密：**论文里的指标和肉眼观感经常对不上**。
>
> 这不是某些人造假，是这个领域的指标本身就有系统性偏差。

## 4.1 一个矛盾的现象

把同一张低分辨率猫脸图给三个模型做 4× 超分：

- **A**：直接 bicubic 上采样
- **B**：ESRGAN（经典 GAN-based SR）
- **C**：SUPIR（扩散派 SOTA）

跑下来的指标可能长这样：

| 模型 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | 肉眼观感 |
|------|--------|--------|---------|-------|---------|
| A. bicubic | **28.4** | **0.82** | 0.42 | 78 | 糊到不能看 |
| B. ESRGAN | 26.1 | 0.76 | 0.18 | 22 | 锐利但有些噪点 |
| C. SUPIR | 24.8 | 0.71 | **0.12** | **8** | 细节丰富有创意 |

矛盾在哪里：

- **PSNR 和 SSIM 给 bicubic 最高分**，但 bicubic 在视觉上是最糊的
- **LPIPS 和 FID 给 SUPIR 最高分**，但 SUPIR 在像素一致性上最差（有"创意"修改）
- 三个模型在不同指标上**没有一个全胜**

这一章讲清两件事：

1. 每个指标在量什么、漏什么
2. 为什么这些不一致是**理论必然**而不是工程缺陷

## 4.2 PSNR

最古老的全参考指标。

$$
\text{PSNR}(\hat{x}, x) = 10 \log_{10} \left( \frac{\text{MAX}^2}{\text{MSE}(\hat{x}, x)} \right)
$$

其中 MSE 是均方误差，MAX 是像素最大值（8 位图为 255）。单位是分贝（dB），数值越大越好。

```python
import torch

def psnr(pred: torch.Tensor, target: torch.Tensor,
         data_range: float = 1.0) -> torch.Tensor:
    """PSNR (Peak Signal-to-Noise Ratio)。
    pred, target: (B, C, H, W), 假设 [0, 1]
    返回: (B,) 每张图的 PSNR (dB)
    """
    mse = ((pred - target) ** 2).flatten(1).mean(dim=1)
    return 10 * torch.log10((data_range ** 2) / (mse + 1e-12))
```

### PSNR 在量什么

直接量像素级 L2 误差。等价于"假设图像噪声是高斯独立同分布"下的对数似然。

### PSNR 漏什么

**漏感知**——它不知道：

- 一像素的小色差和一像素的大错位，对人眼差别巨大，但 PSNR 可能给同样的分数
- 整体偏移 1 像素的图，肉眼几乎看不出，PSNR 会暴跌
- 全图加一点高斯模糊，肉眼明显糊了，PSNR 可能反而升了（因为模糊降低了高频差异）

工程意义：

- PSNR 高的图**不一定好看**，PSNR 低的图**不一定难看**
- PSNR 在比较**算法之间**有相对意义（同样的退化、同样的真值），但跨数据集比较意义有限

### 为什么 PSNR 还是事实标准

它有几个工程好处压不住：

- **简单**：一行 MSE 就能算
- **可加性**：可以分通道、分区域统计
- **历史惯性**：1990 年代起所有论文都报，删不掉

实际工程里 PSNR 仍然是必报指标，但**单独用它做模型选择**是一个新手错误。

## 4.3 SSIM 系列

### SSIM

SSIM（Structural Similarity，2004）试图修补 PSNR 的"忽视结构"问题。它把局部相似度分成三块：

$$
\text{SSIM}(x, y) = l(x, y)^\alpha \cdot c(x, y)^\beta \cdot s(x, y)^\gamma
$$

- **luminance** $l$：均值相似度
- **contrast** $c$：方差相似度
- **structure** $s$：归一化后的相关性

实际计算用滑动窗口（11×11 高斯加权），最后取全图平均。

```python
import torch
import torch.nn.functional as F

def gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)
    return window  # (W, W)


def ssim(pred: torch.Tensor, target: torch.Tensor,
         window_size: int = 11, sigma: float = 1.5,
         data_range: float = 1.0) -> torch.Tensor:
    """SSIM, 简化版本。生产环境推荐用 torchmetrics 或 pyiqa。
    pred, target: (B, C, H, W) in [0, data_range]
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    C = pred.shape[1]

    win = gaussian_window(window_size, sigma).to(pred)
    win = win.expand(C, 1, window_size, window_size)

    mu_x = F.conv2d(pred,   win, padding=window_size // 2, groups=C)
    mu_y = F.conv2d(target, win, padding=window_size // 2, groups=C)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred,     win, padding=window_size // 2, groups=C) - mu_x2
    sigma_y2 = F.conv2d(target * target, win, padding=window_size // 2, groups=C) - mu_y2
    sigma_xy = F.conv2d(pred * target,   win, padding=window_size // 2, groups=C) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return (num / den).flatten(1).mean(dim=1)
```

### MS-SSIM

多尺度 SSIM——把图重复下采样几次，每次都算 SSIM，加权平均。优点：能捕捉不同尺度的结构相似度。

### SSIM 漏什么

- **对模糊宽容**：均值不变、方差变化也不大，SSIM 可能只跌 0.05
- **对几何形变敏感**：物体偏移 5 个像素，SSIM 会跌很多，但人眼几乎看不出
- **对纹理不敏感**：两块不同纹理但统计量相近，SSIM 给高分

工程意义：

- SSIM 比 PSNR 好一点，但**仍然不对齐人眼对纹理和细节的感知**
- 在"换了纹理但保留结构"的场景下，SSIM 会高估相似度

## 4.4 LPIPS

LPIPS（Learned Perceptual Image Patch Similarity，2018）是这个领域的范式转变。它放弃手工公式，改用：

1. 用 AlexNet/VGG/SqueezeNet 提多层特征
2. 把每层特征做空间对齐 + 通道归一化
3. 用一个**学习出来的线性权重**加权各层各通道的 L2 距离
4. 这个线性权重是在大量人类感知判断的数据集（BAPPS）上拟合的

数学上：

$$
\text{LPIPS}(x, y) = \sum_l \frac{1}{H_l W_l} \sum_{h, w} ||w_l \odot (\hat{F}_l(x)_{h,w} - \hat{F}_l(y)_{h,w})||_2^2
$$

直接调用：

```python
import lpips  # pip install lpips

# 用 AlexNet backbone, 通常作为损失函数 backbone 选 vgg
loss_fn_alex = lpips.LPIPS(net='alex')   # net='alex' 更接近人眼, 'vgg' 用作训练损失更稳

def lpips_distance(pred: torch.Tensor, target: torch.Tensor,
                   loss_fn: lpips.LPIPS) -> torch.Tensor:
    """计算 LPIPS 距离。
    输入: [-1, 1] 范围的 RGB, shape (B, 3, H, W)
    返回: (B,) 每张图的 LPIPS 距离 (越小越像)
    """
    return loss_fn(pred * 2 - 1, target * 2 - 1).flatten()
```

### LPIPS 在量什么

学到的"两张图在视觉上有多不像"。在大量"哪一张更像参考图"的人工标注上训练。

### LPIPS 比 PSNR/SSIM 强的地方

- 对**模糊**敏感（这是 PSNR/SSIM 的最大盲区）
- 对**纹理替换**敏感（同一物体不同纹理，LPIPS 会区分）
- 对**整体偏移**鲁棒（小位移不会让 LPIPS 暴跌）

### LPIPS 也漏什么

- **可被对抗样本骗**——AlexNet 本身有对抗脆弱性，模型可以学到"针对 LPIPS 的优化伪影"，LPIPS 低但视觉差
- **对色调漂移不敏感**——VGG/AlexNet 都做 ImageNet 归一化，对整体颜色变化迟钝
- **不擅长极端伪影**——块状伪影、彩色振铃可能 LPIPS 不大但视觉惨不忍睹
- **不同 backbone 给不同结果**——alex 版和 vgg 版的 LPIPS 不能直接比

工程经验：LPIPS 是当前最好用的全参考感知指标，**但不是唯一指标**。

## 4.5 DISTS

DISTS（2020）是 LPIPS 的改进，把感知距离分成两部分：

- **结构距离**（structure）：捕捉空间排列
- **纹理距离**（texture）：捕捉统计特性，对位置不敏感

DISTS 的核心创新：**对纹理位置不敏感**。

举例：草地上的草叶位置不一样、整体纹理一致，DISTS 给高分；LPIPS/SSIM 给低分。这在很多增强场景（修复纹理、生成草地/毛发/水波）下，DISTS 比 LPIPS 更对齐人眼。

```python
# 推荐用 pyiqa 库 (统一了所有 IQA 指标)
import pyiqa  # pip install pyiqa

dists_metric = pyiqa.create_metric('dists', as_loss=False)

def dists_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """DISTS 距离, [0, 1] 输入。"""
    return dists_metric(pred, target)
```

何时用 DISTS：

- 任务涉及**纹理生成或恢复**（毛发、草地、皮肤、布料）
- 已经在 LPIPS 上调到极致但效果不再涨
- 发现模型生成的纹理"统计上对但位置错"被 LPIPS 误判

## 4.6 FID / KID

LPIPS 和 DISTS 都是**全参考**——需要真值 $x$ 配对地比较 $\hat{x}$。但生成式模型有一个不一样的需求：

> 我不在乎单张 $\hat{x}$ 是不是接近 $x$，我在乎 $\hat{x}$ 的**整体分布**像不像自然图像分布。

这是 FID（Fréchet Inception Distance）和 KID 的设计目标。

### FID

把所有真实图像和所有生成图像分别过 InceptionV3，得到两组 2048 维特征。假设各自服从高斯分布，算两个高斯之间的 Fréchet 距离：

$$
\text{FID}(R, G) = ||\mu_R - \mu_G||^2 + \text{tr}(\Sigma_R + \Sigma_G - 2(\Sigma_R \Sigma_G)^{1/2})
$$

```python
import pyiqa
fid_metric = pyiqa.create_metric('fid')
# fid 需要批量数据, 用法不同于 PSNR/LPIPS, 通常给两个文件夹路径
# fid = fid_metric(real_image_folder, generated_image_folder)
```

### FID 在量什么

**两组图在 Inception 特征空间的分布差异**。

适合：

- 生成式增强模型（扩散 SR、生成式去噪）
- 任务的输出空间是多模态的（同样的退化输入可以有多个合理输出）

不适合：

- 判别式 SR/去噪——给定 $y$ 输出唯一 $\hat{x}$，比"分布差异"意义不大
- 单张图——FID 至少需要几千张图，单张图的 FID 是噪声

### FID 的常见误用

> "我们模型在 100 张测试集上 FID = 12，他们模型 FID = 18，所以我们更好"

这有两个问题：

1. **样本量太小**：FID 在小样本上方差极大，100 张图可能跑出 ±10 的标准差，差 6 没意义
2. **bias**：FID 对样本量有系统偏差，必须用同样数量的样本比较

工程经验：

- FID 至少需要 **10K 张样本**才稳定
- 真实工程里 FID 主要看**绝对量级和趋势**（FID 50+ vs 10+ 是质变），不要在小数点上较真
- KID（基于 Maximum Mean Discrepancy）在小样本下比 FID 稳，是更好的小数据指标

## 4.7 无参考指标 (NR-IQA)

真实世界场景里我们经常**没有真值**——你只有一张手机拍的烂图，没有"理想原图"做参照。

这种情况要用无参考（No-Reference）指标。代表：

- **NIQE**（2013）：基于自然图像统计的距离
- **BRISQUE**（2012）：基于 spatial 特征的分类器评分
- **MUSIQ**（2021）：Vision Transformer 学的图像质量评分
- **MANIQA**（2022）：注意力 + 多尺度，目前 NR-IQA 的 SOTA
- **CLIPIQA**（2023）：用 CLIP 评估"美感 / 真实感"

```python
import pyiqa

niqe   = pyiqa.create_metric('niqe',    device='cuda')
musiq  = pyiqa.create_metric('musiq',   device='cuda')
maniqa = pyiqa.create_metric('maniqa',  device='cuda')

# NIQE 越小越好, MUSIQ/MANIQA 越大越好
def assess_no_reference(image: torch.Tensor):
    """对单张图给出无参考质量评估。"""
    return {
        'niqe':   niqe(image).item(),
        'musiq':  musiq(image).item(),
        'maniqa': maniqa(image).item(),
    }
```

### 无参考指标的核心问题

它们只能告诉你"这张图看起来质量好不好"，**不能告诉你它是不是 $x$**。

- **真实退化场景必须的**——没有真值时唯一的选择
- **容易被欺骗**——模型可以学到"针对 NIQE/MUSIQ 看起来质量高"但实际上加了奇怪伪影
- **不同指标之间相关性差**——一个模型 NIQE 好但 MUSIQ 差是常见现象

工程经验：

- 真实场景评估**多个 NR 指标一起报**（NIQE + MUSIQ + MANIQA + CLIPIQA），任何一个指标不正常都可能是失败模式
- 无参考指标永远配合**人工评估**使用，不能单独决定模型好坏

## 4.8 保真 vs 感知 —— 一个理论必然

回到开头那张矛盾的表。这不是工程巧合，是**信息论必然**。

Blau & Michaeli 在 2018 年的论文 "The Perception-Distortion Tradeoff" 严格证明了：

> 在 ill-posed 逆问题上，**distortion**（PSNR / SSIM 这类保真指标）和 **perception**（FID / LPIPS 这类感知指标）**不可同时最优**。
>
> 任何优化算法都在这条 trade-off 曲线上找一个点。

直观理解：

- **distortion 最优**意味着输出尽量接近真值的"统计期望"——多个合理 $x$ 的均值，是模糊的
- **perception 最优**意味着输出落在自然图像分布上——必须是某个具体的 $x$，不是均值

这两个目标本质冲突。

```
         ↑ Perception (FID, LPIPS)
       Bad
    *
       *
         *
           *  <- 现实模型在这条曲线上选点
             *
               * 
                 *
                  Best
                   ←————————————→ Distortion (PSNR, SSIM)
                  Good          Bad
```

**这条曲线的存在**有几个工程含义：

1. **PSNR 派和感知派"打架"是常态**，不会有一个模型在所有指标上全胜
2. **看 benchmark 时要看曲线上的位置，不是单点**——HAT 的位置是高 PSNR 低感知，SUPIR 是低 PSNR 高感知，这两个不是同一目标的竞争者
3. **应用场景决定曲线上的最优点**——法医证据要 distortion 端，照片美化要 perception 端

第 8 章会回到这个 trade-off，讲扩散模型为什么主动选择牺牲 PSNR 换感知。

## 4.9 benchmark 数据集的偏差

学术论文几乎都报这几个数据集的指标。但**这些数据集都有系统性偏差**。

### 经典 benchmark

| 数据集 | 图片数量 | 类型 | 偏差 |
|-------|---------|------|------|
| Set5 | 5 | 杂图 | 太少，方差极大 |
| Set14 | 14 | 杂图 | 同上 |
| BSD100 | 100 | 自然图 | 略小，平均仍可信 |
| Urban100 | 100 | 城市建筑 | 偏向规则结构 |
| Manga109 | 109 | 漫画 | 全是黑白线条 |
| DIV2K val | 100 | 高质量自然图 | 标准但偏"漂亮" |
| LSDIR val | 250 | 多样化 | 较新较公平 |

### 所有这些数据集的共同问题

> **它们的"低分辨率版本"几乎都是 bicubic 下采样得到的。**

也就是说：

- 训练 $y$ 用 bicubic
- 测试 $y$ 也用 bicubic
- **测试和训练在同一个退化假设下**

这就是第 1 章讲的"训练-推理失配"在评估上的对应版本。结论：

> "在 Set14 4× 上 30 dB" **不能**告诉你"在你手机暗光照片上好不好"。

### 真实退化数据集

为了测真实场景，2019 年起出现了"真实退化对"数据集：

- **RealSR**（2019）：用同一相机不同焦段拍同一场景，得到 LR-HR 对
- **DRealSR**（2020）：DSLR 相机，更精确对齐
- **DPED**（2017）：低端手机 vs DSLR 配对
- **NTIRE Real-World SR challenge**：每年的竞赛数据

这些数据集小（几百到几千张），但**测试时更接近真实**。

工程实践：

- **学术 benchmark + 真实退化数据集都报**
- 学术 benchmark 上 PSNR 高没意义（除非两个模型都在同退化下比较）
- 真实退化数据集上的指标更说明问题

## 4.10 训练时看哪些指标

每个 epoch / 每 N 步 validation 时，应该跑哪些指标？

**最小集合（必须）**：

- PSNR：监控保真度
- LPIPS：监控感知质量

**推荐集合**：

- PSNR + SSIM + LPIPS + DISTS（全参考全套）
- 模型损失曲线（每个损失项分开记录）

**生成式模型额外**：

- FID（每 N 个 epoch 在足够大测试集上跑一次）
- 一组固定输入的可视化对比（直观看到模型变化）

```python
class ValidationMetrics:
    """训练时的 validation 指标 logger。"""
    def __init__(self, device='cuda'):
        import pyiqa
        self.psnr  = pyiqa.create_metric('psnr',  device=device, as_loss=False)
        self.ssim  = pyiqa.create_metric('ssim',  device=device, as_loss=False)
        self.lpips = pyiqa.create_metric('lpips', device=device, as_loss=False)
        self.dists = pyiqa.create_metric('dists', device=device, as_loss=False)

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        return {
            'psnr':  self.psnr (pred, target).mean().item(),
            'ssim':  self.ssim (pred, target).mean().item(),
            'lpips': self.lpips(pred, target).mean().item(),
            'dists': self.dists(pred, target).mean().item(),
        }
```

## 4.11 论文 / 报告里报什么

主表必须有：

- PSNR / SSIM / LPIPS（基础全参考）
- 至少一个真实退化数据集的结果

视情况加：

- DISTS（如果是纹理生成任务）
- FID（如果是生成式模型）
- NIQE / MUSIQ / MANIQA（真实场景）
- 主观评测（MOS / 2AFC，第 12 章详谈）

**不要做的**：

- 只报 PSNR
- 只在 Set5/Set14 上报指标
- 报 FID 但样本量小于 1000
- 不展示失败案例

## 4.12 一个具体案例：bicubic vs ESRGAN vs SUPIR

回到开头那个表，但展开分析一下为什么每个指标这样：

| 模型 | PSNR | SSIM | LPIPS | FID | 解读 |
|------|------|------|-------|-----|------|
| bicubic | **28.4** | **0.82** | 0.42 | 78 | 输出是真值的"模糊均值"，PSNR/SSIM 自然高；但完全无高频，LPIPS/FID 差 |
| ESRGAN | 26.1 | 0.76 | 0.18 | 22 | GAN 生成高频细节，"细节对但不一定是真细节"，PSNR 跌但感知大幅好 |
| SUPIR | 24.8 | 0.71 | **0.12** | **8** | 扩散先验生成更真实细节，但创意性强，像素级一致性最差 |

**没有一个模型"最好"**——选哪个取决于场景：

- 法医证据增强 → bicubic（不允许编造）
- 照片放大打印 → ESRGAN（保真+清晰平衡）
- 老照片修复 → SUPIR（最大化视觉真实感）

> **"哪个模型最好"是一个错误的问题。**
> **正确的问题是"在 distortion-perception 平面上，哪个点最适合我的应用"。**

## 4.13 小结

1. **PSNR/SSIM 量保真，对感知不敏感**——单独看会误导，但作为基线必报
2. **LPIPS/DISTS 量感知**，比 PSNR/SSIM 对齐人眼，但不能盲信
3. **FID 量分布距离**，适合生成式模型，但要足够大样本量
4. **NR-IQA**（NIQE/MUSIQ/MANIQA）在真实场景必须，但容易被骗
5. **Perception-Distortion 是理论 trade-off**——任何模型都在这条曲线上选点
6. **学术 benchmark 都用 bicubic 退化**，与真实场景失配
7. **训练时至少 PSNR + LPIPS**，论文报告至少 PSNR + SSIM + LPIPS + 真实退化数据集
8. **没有一个指标能单独说明问题**——总是组合报、配合人工评估

这一章和第 3 章一起回答了"什么叫好"——损失定义训练目标，指标定义评估标准。这两章理解清楚后，后面的架构章节才有"评判好坏"的依据。

---

> 下一章 [数据与退化合成](05-degradation-pipeline.md) → Real-ESRGAN 真正的核心贡献是数据合成。我们将看到，同样的网络换不同的数据 pipeline，性能差距能达到一个数量级。
