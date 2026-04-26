# 第 6 章 · CNN 时代

> 2014 年到 2022 年，CNN 在低层视觉走过了一条独特的路：
>
> 从模仿稀疏编码（SRCNN）→ 加深加残差（VDSR/EDSR）→ 引入注意力（RCAN）→ 密集连接（RRDB）→ 反向简化（NAFNet）。
>
> 这八年是这个领域的"经典力学"——理解了它，再看 Transformer 和扩散就能找到对应。

## 6.1 为什么从 CNN 开始

Transformer 在 2021 年起在低层视觉显身手（SwinIR、Restormer、HAT），扩散在 2023 年起占据生成派 SOTA（StableSR、SUPIR）。看起来 CNN 已经是过去式。

**实际不是这样。**

- NAFNet（2022 纯 CNN）至今仍是去噪/去模糊任务的事实标杆
- Real-ESRGAN 用的还是 RRDB（2018 的 CNN 架构）
- 几乎所有 Transformer 模型的 patch embedding、上采样头、bottleneck 仍然是卷积
- 移动端部署的增强模型 99% 是纯 CNN（Transformer attention 在端侧效率太差）

所以 Part II 的第一站必须是 CNN。这一章讲清三件事：

1. CNN 在低层视觉的演进逻辑（哪些设计是渐进改良、哪些是范式转变）
2. 每个时代的代表网络在解决什么具体问题
3. 设计自己的 CNN 增强网络时，选什么、不选什么

## 6.2 SRCNN（2014）—— 起点

第一篇深度学习超分论文。Dong et al. 把传统的稀疏编码超分流程映射成了三层 CNN：

```
Layer 1 (9×9 conv, 64 ch)  ←→ Patch extraction & representation
Layer 2 (1×1 conv, 32 ch)  ←→ Non-linear mapping
Layer 3 (5×5 conv,  3 ch)  ←→ Reconstruction
```

输入：bicubic 上采样到目标尺寸的 LR
输出：HR 估计

```python
import torch.nn as nn

class SRCNN(nn.Module):
    """SRCNN, 2014. 三层 CNN 的开山之作。"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x 是 bicubic 上采样后的 LR, shape 已经等于 HR
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
```

性能：在 Set5 4× 上 ~30.5 dB（bicubic 是 28.4 dB）。

**SRCNN 的价值不在效果**，在于它**证明了端到端学习的可行性**——之前所有超分方法都是"先稀疏字典学习、再块分类、再重建"的多阶段流程，SRCNN 把这一切变成一个 CNN，开启了之后十年的发展。

**SRCNN 的局限**：

- **太浅**：只有 3 层
- **先上采样浪费计算**：所有计算都在 HR 尺寸上做
- **大卷积核（9×9）效率差**：参数多但有效感受野有限

后续的工作都在解决这些问题。

## 6.3 VDSR（2016）—— 深 + 残差

Kim et al. 提出 VDSR（Very Deep SR），核心两点贡献：

### 加深到 20 层

VGG 风格的堆叠 3×3 卷积。20 层的感受野比 SRCNN 大很多，能利用更广的上下文。

### 残差学习

不是直接学 HR，而是学**残差** $r = x - y_{\text{up}}$（HR 减去上采样的 LR）。

为什么残差学习在低层视觉特别重要？

- LR 上采样后 $y_{\text{up}}$ 已经接近 HR 的低频成分
- 模型只需要学**高频补充**，不必重新学整张图
- 残差的方差远小于原图，**优化更容易**
- 残差大部分接近 0（平坦区域），梯度更稀疏，模型更容易收敛

```python
class VDSR(nn.Module):
    """VDSR, 2016. 深 + 残差学习。"""
    def __init__(self, num_layers: int = 20, base_ch: int = 64):
        super().__init__()
        layers = [nn.Conv2d(3, base_ch, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers += [nn.Conv2d(base_ch, base_ch, 3, padding=1),
                       nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(base_ch, 3, 3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        # x 是 bicubic 上采样后的 LR
        return x + self.body(x)  # 残差学习: 输出 = 输入 + 学到的残差
```

性能：Set5 4× 上 ~31.4 dB，比 SRCNN 提升约 0.9 dB。

**残差学习从此成为低层视觉的事实标准**——之后所有网络（EDSR、RCAN、RRDB、NAFNet）都用残差。

**VDSR 的局限**：

- 仍然先 bicubic 上采样再过网络，计算浪费严重（4× 时计算量是 LR 空间的 16 倍）

## 6.4 EDSR（2017）—— 移除 BN，残差块标准化

Lim et al. 提出 EDSR（Enhanced Deep Residual SR），是低层视觉的"工程基线"网络。它的几个决定影响了后续所有 CNN 工作。

### 决定一：移除 Batch Normalization

ResNet 的标准残差块是 `Conv → BN → ReLU → Conv → BN`。EDSR 论文发现：**在低层视觉里去掉 BN 反而更好**。

原因（这一节稍微展开，因为重要）：

- BN 本质是把每个 mini-batch 的特征做归一化
- 高层视觉（分类）：归一化能稳定训练，且分类的目标是 invariant 的
- 低层视觉（重建）：**像素的绝对值就是输出**——归一化破坏了输入和输出之间的尺度关系
- BN 在不同 batch 上行为不同，**测试时切到 EMA 后会有 train-test mismatch**
- BN 也限制了大 batch size 下的内存

**移除 BN 后**：

- 模型在低对比度、纯色区域更准确
- 能用更大的网络（省下来的内存可以增加深度/宽度）
- 训练稳定性反而更好

这是低层视觉和高层视觉一个**很重要的差异点**：高层视觉里 BN/LayerNorm 是必备，低层视觉里它们经常有害。

### 决定二：在 LR 空间计算 + 末尾 PixelShuffle 上采样

VDSR 在 HR 空间计算的浪费被 EDSR 修正了：所有特征提取在 LR 空间做，最后用 **PixelShuffle**（sub-pixel convolution）做上采样。

PixelShuffle 的本质：通道维转空间维。把 $(B, r^2 C, H, W)$ 重新排成 $(B, C, rH, rW)$。

```python
# PyTorch 内置 PixelShuffle, 但理解一下数学:
# 输入 (B, r^2 * C, H, W)
# 输出 (B, C, rH, rW)
# 重排方式: 每 r^2 个通道 → 一个 r×r 的空间块
import torch.nn as nn

class UpsampleBlock(nn.Module):
    """EDSR 风格的上采样模块。
    支持 2/3/4 倍。4 倍 = 两次 2 倍。"""
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

为什么 PixelShuffle 是事实标准（vs nearest/bilinear/transpose conv）：

- **transposed conv**：会产生棋盘伪影（checkerboard），如果 stride 与 kernel size 不匹配
- **nearest + conv** / **bilinear + conv**：可以，但参数效率低
- **PixelShuffle**：通过 `r²` 倍通道的卷积学到的 reorder，参数效率最高，无棋盘伪影

实际工程里 PixelShuffle 是**默认选择**。除非有特殊原因（比如部署平台不支持），不要用 transposed conv。

### 决定三：残差块标准化

EDSR 的残差块就是 `Conv → ReLU → Conv` + 残差，**没有归一化**。这个设计成了之后几年的标准。

```python
class ResidualBlock(nn.Module):
    """EDSR 风格的残差块: 无 BN, 无激活在残差路径末尾。"""
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
    """EDSR, 2017. 低层视觉的工程基线。"""
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
        body = self.body_tail(self.body(feat)) + feat   # 全局残差
        out = self.tail(self.upsample(body))
        return out
```

**res_scale = 0.1** 是个工程小细节：把残差路径的输出缩小 10×，防止深网络的残差累积导致数值发散。这个技巧在 EDSR 之后被广泛采用。

## 6.5 RCAN（2018）—— 引入注意力

EDSR 之后的方向之一是把**注意力机制**引入低层视觉。Zhang et al. 的 RCAN 是代表。

### Channel Attention

不同通道学到不同的特征——有些通道对**纹理**敏感，有些对**边缘**敏感，有些可能是冗余的。Channel Attention 让模型学一个**通道权重**，自动放大重要通道、抑制不重要通道。

```python
class ChannelAttention(nn.Module):
    """RCAN 用的 SE-style channel attention。"""
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
        # 全局平均池化 -> 通道压缩 -> 通道还原 -> sigmoid 归一化
        # 得到 (B, C, 1, 1) 的通道权重, 然后 broadcast 乘回去
        w = self.fc(self.avg_pool(x))
        return x * w
```

Channel Attention 在低层视觉的具体作用：

- **抑制噪声通道**：噪声特征在某些通道上集中，attention 自动降权
- **放大边缘通道**：边缘对重建关键，attention 给它们更大权重
- **自适应输入内容**：纹理多的图和平坦的图，需要的通道权重不同

### 残差中的残差（Residual in Residual）

RCAN 把残差块嵌套：每个 RCAB（带 channel attention 的残差块）外面再包一层残差，多个 RCAB 组成 group，group 外面再包一层残差。这种结构能稳定训练 400 层以上的网络。

```python
class RCAB(nn.Module):
    """Residual Channel Attention Block。RCAN 的基础单元。"""
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

性能：Set5 4× 上 ~32.6 dB，比 EDSR 提升约 0.5 dB。

## 6.6 RRDB（ESRGAN，2018）—— 密集连接

Wang et al. 在 ESRGAN 里提出 RRDB（Residual in Residual Dense Block）。这是**Real-ESRGAN 至今还在用的 backbone**。

### Dense Block

ResNet 用残差连接（add），DenseNet 用密集连接（concat）。在每一层，把前面所有层的输出 concat 起来作为输入：

$$
x_{l+1} = H([x_0, x_1, \dots, x_l])
$$

优点：

- **最大特征复用**：每层都直接看到所有前面的特征
- **缓解梯度消失**：梯度可以通过任何层直接传到输入
- **参数效率**：每层的输出通道少（growth rate），但特征丰富

### RRDB 的具体设计

每个 RRDB 包含 3 个 dense block，每个 dense block 由 5 个 conv + LeakyReLU 层组成。block 内 dense connection、block 间 residual connection、整个 RRDB 外面再一层 residual。

```python
class DenseBlock(nn.Module):
    """ESRGAN 用的 5 层 dense block。"""
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
        return x5 * 0.2 + x   # residual scale + 残差


class RRDB(nn.Module):
    """Residual in Residual Dense Block。"""
    def __init__(self, ch: int = 64, growth: int = 32):
        super().__init__()
        self.db1 = DenseBlock(ch, growth)
        self.db2 = DenseBlock(ch, growth)
        self.db3 = DenseBlock(ch, growth)

    def forward(self, x):
        out = self.db1(x)
        out = self.db2(out)
        out = self.db3(out)
        return out * 0.2 + x   # 又一层残差
```

ESRGAN 整体堆叠 23 个 RRDB，约 1700 万参数。这个网络在 PSNR 指标上比 RCAN 略低，但配合 GAN 训练后**视觉质量**远好于纯 PSNR 优化的 RCAN——这就是 PSNR 派和感知派的分裂（第 4 章 4.8 节）。

Real-ESRGAN 沿用 RRDB 网络，只换数据 pipeline 和训练损失，效果完全脱胎换骨。**这再次验证了第 5 章 5.1 节的结论：数据 > 网络。**

## 6.7 NAFNet（2022）—— 反潮流的简化

2018-2022 年低层视觉的趋势是"加更多花活"：transformer block、各种 attention、复杂归一化。Chen et al. 的 NAFNet 反过来——**移除一切非必要的东西**，反而做到去噪/去模糊 SOTA。

### 移除清单

- ❌ Batch Normalization
- ❌ GELU / Swish（用更简单的 SimpleGate 代替）
- ❌ Self-Attention（保留极简的 channel attention）
- ❌ ReLU（Plain net 完全不用激活函数）

注意：NAFNet **保留了简化的 LayerNorm2d**（每个 block 的 spatial 路径和 channel 路径开头各一次），并不是把所有归一化层都移除。它移除的是非线性 + attention 这两类"花架构"，归一化反而是稳定训练的必需。

### SimpleGate（替代 GELU）

GELU 是 $x \cdot \Phi(x)$（输入乘以高斯 CDF）。NAFNet 注意到这本质是"输入乘以一个门控信号"，把它简化成：

$$
\text{SimpleGate}(x) = x_1 \odot x_2
$$

其中 $x = [x_1, x_2]$ 是输入沿通道维一分为二，元素相乘。

```python
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
```

视觉影响：和 GELU 类似的非线性能力，**没有可学参数 + 比 GELU 便宜**（只是 chunk + 元素级乘法，没有 erf/exp 的近似计算）。

### Simplified Channel Attention（SCA）

```python
class SCA(nn.Module):
    """NAFNet 的简化 channel attention - 只保留 average pool + 1x1 conv。"""
    def __init__(self, ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        return x * self.conv(self.pool(x))
```

注意：

- 没有 reduction（RCAN 是 ch//16 → ch）
- 没有 ReLU
- 没有 sigmoid（直接相乘，不是归一化的权重）

这种"看起来不对"的设计反而效果好——这是 NAFNet 论文的反直觉发现。

### NAFBlock 整体

```python
class NAFBlock(nn.Module):
    """NAFNet 的核心 block。"""
    def __init__(self, ch: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        # Spatial mixing
        self.norm1 = LayerNorm2d(ch)
        self.conv1 = nn.Conv2d(ch, ch * dw_expand, 1)
        # Depthwise
        self.dwconv = nn.Conv2d(ch * dw_expand, ch * dw_expand, 3,
                                padding=1, groups=ch * dw_expand)
        self.gate1 = SimpleGate()  # 输出通道减半
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

### NAFNet 的启示

NAFNet 论文最有意思的点不是它的具体设计，是它的**消融实验**：

- 把 SE 换成 SCA：效果不变
- 把 GELU 换成 SimpleGate：效果不变
- 移除所有 LayerNorm：效果略降但不大
- ……

结论：

> **低层视觉的关键是计算预算分配，不是花架构。**
>
> 同样的 FLOPs，简单网络和复杂网络效果差不了多少；
> 复杂网络的"复杂"主要带来训练不稳和部署困难。

这个结论对工程实践影响很大——**生产环境优先选简单 CNN**，除非有明确证据表明复杂网络带来质变。

## 6.8 上采样方法对比

低层视觉里上采样位置和方法是关键设计决策。

| 方法 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **bicubic + conv** | 输入先 bicubic 到 HR，再 CNN | 简单 | HR 空间计算贵 |
| **transpose conv** | 反卷积 | 学习上采样 | 棋盘伪影 |
| **nearest + conv** | 复制最近邻 + 卷积 | 无伪影 | 参数效率低 |
| **bilinear + conv** | 双线性 + 卷积 | 平滑 | 略糊 |
| **PixelShuffle** | sub-pixel conv | 参数效率最高 | 训练初期可能棋盘 |
| **PixelShuffle (ICNR init)** | 初始化做修正 | 修复棋盘问题 | 实现稍复杂 |

**ICNR 初始化**（Initialization for Convolutional NN with sub-pixel convolutions）：

```python
def icnr_init(tensor: torch.Tensor, scale: int = 2):
    """PixelShuffle 前的卷积权重初始化, 避免训练初期的棋盘伪影。
    本质: 让 r^2 个分组的初始权重相同。"""
    out_ch = tensor.shape[0]
    sub_ch = out_ch // (scale ** 2)
    sub_kernel = torch.zeros(sub_ch, *tensor.shape[1:])
    nn.init.kaiming_normal_(sub_kernel)
    sub_kernel = sub_kernel.repeat(scale ** 2, 1, 1, 1)
    tensor.copy_(sub_kernel)
    return tensor
```

工程经验：**默认 PixelShuffle + ICNR 初始化**，伪影问题基本消除。

## 6.9 归一化层选择

低层视觉里归一化层的选择和高层视觉很不同。

| 归一化 | 作用维度 | 在低层视觉中 | 推荐度 |
|-------|---------|-------------|-------|
| **BatchNorm** | (B, H, W) | **有害**（破坏尺度、test 不稳） | ❌ 不要用 |
| **GroupNorm** | (G, H, W) per channel group | 还行 | ⭐⭐ 可接受 |
| **InstanceNorm** | (H, W) per channel | 风格迁移用，去噪不合适 | ⭐ 谨慎 |
| **LayerNorm 2d** | (C) per pixel | 现代 Transformer 标准 | ⭐⭐⭐ 推荐 |
| **No Norm** | — | NAFNet/EDSR 等 | ⭐⭐⭐ 推荐 |

**为什么 LayerNorm 在 Transformer-based 低层视觉里 OK，而 BN 不行？**

- BN 的归一化跨 batch 维度——同一张图的某个像素在不同 batch 里行为不同
- LayerNorm 跨 channel 维度——只看当前像素自己的特征向量，每张图独立

LayerNorm 对单张图是确定性的，**没有 train-test mismatch**。

工程实践：

- 纯 CNN 网络：**No Norm**（EDSR/NAFNet 风格）
- Transformer-based：**LayerNorm**（SwinIR/Restormer 风格）
- 永远不要用 BN

## 6.10 激活函数选择

| 激活 | 形式 | 在低层视觉中 |
|------|------|-------------|
| **ReLU** | $\max(0, x)$ | EDSR 用，简单稳定 |
| **LeakyReLU** | $\max(0.01x, x)$ | ESRGAN 用，避免死神经元 |
| **GELU** | $x \cdot \Phi(x)$ | Transformer 标配 |
| **SiLU/Swish** | $x \cdot \sigma(x)$ | 现代默认 |
| **SimpleGate** | $x_1 \odot x_2$ | NAFNet，0 计算 |

工程经验：

- **保守选择**：LeakyReLU(0.2)。在所有 GAN 类训练中稳，避免梯度死亡
- **追求效率**：SimpleGate
- **追求效果**：SiLU (Swish)

## 6.11 注意力模块演进

CNN 里的注意力主要是 channel attention。演进路线：

| 模块 | 复杂度 | 来源 |
|------|-------|------|
| **SE** | $C^2/r$ 参数 | Squeeze-Excitation, 2018 |
| **CA (RCAN)** | 同 SE | 引入低层视觉 |
| **ECA** | $k$ 参数 (k=3 or 5) | Efficient CA, 2020 |
| **SCA** | $C^2$ 参数无 reduction | NAFNet, 2022 |
| **Sim-AM** | 0 参数 | 基于能量函数 |

**ECA (Efficient Channel Attention)** 用 1D 卷积替代 SE 的两层 FC：

```python
class ECA(nn.Module):
    """Efficient Channel Attention. 几乎没有参数。"""
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

### Spatial Attention 在低层视觉

CBAM 等模块加 spatial attention（让模型关注图像某些位置）。但**低层视觉里 spatial attention 收益有限**——任何位置的像素都重要，没有"该忽略"的位置。

实际工程里 channel attention 是主流，spatial attention 用得很少。第 7 章 Transformer 的 self-attention 才是真正的 spatial 信息建模。

## 6.12 一个完整的 EDSR-style 模型

把上面的概念拼成一个生产可用的 SR 网络。**这是工程基线**——你做新任务、不知道选什么时，先用这个。

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
    """生产可用的 SR baseline。
    深度 16 残差块 + 64 通道, 大约 1.5M 参数。
    实战表现: 在 RealESRGAN pipeline 数据上, 配 L1+VGG+RaGAN 损失, 
            达到接近 ESRGAN 的视觉质量, 训练快 2-3×。
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

        # ICNR 初始化 PixelShuffle 前的卷积
        # 注意 scale: 4× 总放大用两次 ×2, 所以这里 ICNR scale=2 与 UpsampleBlock 内部一致
        # 如果是 ×3 直接放大, 应传 scale=3
        per_step_scale = 2 if scale == 4 else scale
        for m in self.up.modules():
            if isinstance(m, nn.Conv2d):
                self._icnr_init(m.weight, scale=per_step_scale)

    @staticmethod
    def _icnr_init(weight, scale: int = 2):
        """ICNR: 初始化 PixelShuffle 前的卷积权重, 让 r^2 个分组初始权重相同。
        scale 必须等于 PixelShuffle 的上采样倍率。
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

## 6.13 设计一个 SR 网络的工程经验

如果要从零设计一个新任务的 CNN 增强网络，按这个顺序考虑：

### 第一步：定计算预算

- 端侧（手机 NPU）：< 0.5G FLOPs，参数 < 1M
- 桌面 GPU 实时：< 50G FLOPs
- 后端服务（A100）：可以 200G+ FLOPs
- 离线处理：不限

### 第二步：分配预算到深度 vs 宽度

经验：**先加深度再加宽度**。深度对感受野和表达能力贡献更大；宽度只是更多通道，边际效益快速递减。

典型配置：

- 1M 参数：32 通道 × 16 块
- 5M 参数：64 通道 × 16 块
- 16M 参数：64 通道 × 23 块（ESRGAN 配置）
- 50M+ 参数：考虑换 Transformer 架构

### 第三步：选上采样位置和方法

- 99% 用 PixelShuffle，放在网络末尾
- 输入 LR + 输出 HR 的整体残差（让网络只学高频补充）

### 第四步：选 attention

- 通用任务：加一个 SCA 或 ECA（极低开销，PSNR 涨 0.1-0.2 dB）
- 任务特别需要 attention（人脸、特定纹理）：考虑 CBAM 或 Transformer block

### 第五步：训练损失

- PSNR 导向：Charbonnier 单一损失
- 真实 SR：L1 + VGG + RaGAN（参考第 3 章）

## 6.14 小结

1. **CNN 在低层视觉的演进是渐进的**：SRCNN → VDSR (深+残差) → EDSR (无 BN) → RCAN (CA) → RRDB (dense) → NAFNet (简化)
2. **残差学习是低层视觉的事实标准**——所有现代网络都用
3. **Batch Normalization 在低层视觉有害**——破坏尺度，移除即可
4. **PixelShuffle + ICNR 初始化是上采样的事实标准**
5. **Channel Attention 是有效的小改进**（0.2-0.5 dB），SCA/ECA 是低开销选择
6. **NAFNet 的启示**：低层视觉关键是计算预算分配，不是花架构
7. **设计新网络的顺序**：定预算 → 深度 > 宽度 → PixelShuffle → 加 SCA → 选损失
8. **Real-ESRGAN 的成功证明**：用 2018 年的 RRDB + 2021 年的数据 pipeline，效果远超用 2022 年新架构 + 旧数据

CNN 的故事讲到这里。下一章 Transformer 进入低层视觉，会带来一个新维度的思考——**长距离依赖**。CNN 的感受野是局部的，Transformer 的 self-attention 让每个像素都能看到所有其他像素。这在去噪、去模糊、超分上各有不同的工程后果。

---

> 下一章 [Transformer 在低层视觉](07-transformer.md) → SwinIR、Restormer、HAT，注意力机制如何取代部分 CNN 的角色。
