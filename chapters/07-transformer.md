# 第 7 章 · Transformer 在低层视觉

> CNN 的感受野是局部的，它通过堆深度间接获得大感受野。
>
> Transformer 的 self-attention 是全局的，但代价是 $O(N^2)$ 计算。
>
> 低层视觉里 Transformer 怎么落地，是过去三年这个领域最有趣的工程故事之一。

## 7.0 阅读须知

本章是 Part II 的第二站，承接第 6 章的 CNN 时代，把自注意力机制（self-attention）引入低层视觉之后这四五年的设计演进梳理一遍。读完之后，你应当能够：

- 理解为什么长距离依赖对低层视觉确有实质帮助，而非单纯套用 Transformer 概念
- 看懂局部窗口注意力（W-MSA / SW-MSA）与转置通道注意力（MDTA）两条主流路线的工程取舍
- 在为新任务选择骨干网络时，能够在 CNN、SwinIR、Restormer、HAT 及混合架构之间做出有依据的判断
- 读懂 SwinIR 与 Restormer 的核心模块代码，明确每一行对应的算子作用

本章预设你已经掌握：

- 第 6 章 CNN 时代的演进脉络（尤其是残差块、PixelShuffle、通道注意力）
- 标准自注意力的形式 $\text{softmax}(QK^T / \sqrt{d}) V$ 及其关于序列长度的 $O(N^2)$ 复杂度
- LayerNorm 与 BatchNorm 的统计维度差异（第 6 章 §6.9 已展开）

不预设你读过 ViT、Swin、SwinIR、Restormer、HAT 的原始论文，也不预设你写过自定义注意力的底层算子实现。

本章首次出现或将反复出现的缩写如下，先集中列出一遍，后面用到具体术语时还会再展开一句话定义：

- **ViT**（Vision Transformer）：2020 年 Dosovitskiy et al. 提出，把图像切成 16×16 的 patch 当作 token 喂给标准 Transformer，是 Transformer 在视觉里的奠基工作
- **MSA**（Multi-head Self-Attention，多头自注意力）：标准 Transformer 的核心算子，多个 attention head 各看一段子空间
- **W-MSA**（Window-based Multi-head Self-Attention）：只在 $M \times M$ 的局部窗口内做 attention，把复杂度从 $O((HW)^2)$ 降到 $O(HW \cdot M^2)$
- **SW-MSA**（Shifted Window MSA）：W-MSA 的搭档，每隔一层把窗口偏移 $M/2$，让相邻层的窗口边界错开，使信息跨窗口流动
- **Swin Transformer**：2021 年 Liu et al. 提出，用 W-MSA / SW-MSA 交替的 hierarchical 视觉 Transformer
- **SwinIR**（Swin Transformer for Image Restoration）：2021 年 Liang et al. 把 Swin 移到低层视觉，做 SR/去噪/去模糊的统一架构
- **RSTB**（Residual Swin Transformer Block）：SwinIR 的中间层组件，多个 Swin Transformer Layer 套一层残差
- **STL**（Swin Transformer Layer）：RSTB 内部的基本单元，由一次 W-MSA 或 SW-MSA + 一次 MLP 组成
- **Restormer**（Restoration Transformer）：2022 年 Zamir et al. 提出，用通道维 attention（MDTA）+ 门控 FFN（GDFN）做去噪/去模糊/去雨的通用 SOTA
- **MDTA**（Multi-Dconv Head Transposed Attention）：Restormer 的核心算子，把 self-attention 从 token 维转到 channel 维，复杂度从 $O((HW)^2 d)$ 降到 $O(d^2 \cdot HW)$
- **GDFN**（Gated-Dconv Feed-Forward Network）：Restormer 的 FFN 模块，加入门控 + depthwise 卷积
- **HAT**（Hybrid Attention Transformer）：2023 年 Chen et al. 提出，把 W-MSA、Channel Attention、Overlapping Cross-Attention 三种 attention 组合
- **HAB**（Hybrid Attention Block）：HAT 的基础 block，W-MSA + CAB 并行
- **OCAB**（Overlapping Cross-Attention Block）：HAT 的扩展 block，在重叠窗口上做 cross-attention，弥补 SW-MSA 的间接性
- **CAB**（Channel Attention Block）：HAT 内部用的 channel attention 模块，结构上接近 RCAN 的 RCAB
- **LAM**（Local Attribution Map）：可视化模型实际利用哪些输入像素的工具，HAT 用它分析 SwinIR 的"容量没用满"问题
- **self-similarity**（自相似性）：自然图像里同一种纹理在多个位置出现的统计现象，是 attention 在低层视觉有用的核心理由
- **non-local means**：经典图像处理里利用自相似性做去噪的方法，attention 可看作它的可学习版本

## 7.1 为什么 Transformer 来到低层视觉

在第 6 章的演进里，CNN 的每一步改良几乎都在设法扩大感受野：加深层数、引入通道注意力、改用密集连接。但标准卷积的感受野受核尺寸与层数限制，理论感受野随深度线性增长，实际有效感受野（Effective Receptive Field）却远小于理论值。多项可视化研究表明，20 层 ResNet 的有效感受野半径只有理论值的约 1/3，且权重按高斯分布向中心快速衰减，越远的像素对中心位置的影响越微弱。

而在低层视觉里，有一条容易被忽视的统计事实：

> 远距离像素在图像复原任务中往往携带着直接可用的互补信息。

具体体现在四类场景：

- **去噪**：自然图像中常有大面积重复纹理（草地、砖墙、织物），远处受噪声污染较轻的相似块可以为当前位置提供干净先验，多处相似样本加权平均即可显著压低噪声方差
- **超分**：同一幅图中重复出现的几何结构（整排窗户、屋瓦、同一字体的相同笔画）可能处于不同的亚像素相位，互相借用即可拼出丢失的高频细节
- **去模糊**：当整张图受同一运动模糊核或散焦核退化时，单靠局部区域难以消除核估计的歧义，联合全图各处的强边缘才能收紧对模糊核的约束
- **去雾 / 去雨**：大气散射与雨纹具有全图尺度的统计一致性，远处天空区域的透射率与大气光估计能直接指导近处景物的色彩恢复

这正是经典图像处理中非局部均值（**non-local means**）的核心思想：自相似性（self-similarity）是自然图像的内禀统计性质，一个像素最有价值的参考未必来自空间紧邻位置，而可能来自图像另一端结构相似的区域。纯 CNN 要跨越几十上百像素建立这种联系必须堆叠很深的层数，传递效率低。

自注意力机制天然契合这一需求：任意两个空间位置都能在单层内直接计算相关性并聚合特征，跨距离的自相似匹配一步到位。

但直接套用全局自注意力的代价是 $O(H^2 W^2 \cdot C)$ 的算力与显存复杂度。在 $256 \times 256$ 分辨率的特征图上，单头注意力矩阵的大小就达到 $65536 \times 65536$，仅存储这一张 FP32 矩阵就需要约 17 GB 显存，根本无法直接训练。

因此，Transformer 在低层视觉里的全部架构演进，围绕的都是同一个核心问题：如何在保留长距离建模能力的同时，把二次方复杂度压回线性可控的范围。

为了直观对比"全局注意力、局部窗口注意力、转置通道注意力"三种主流路线的取舍，下图展示了在同一张 $H \times W$ 特征图上，单个查询像素在不同机制下的可见范围与计算代价：

```mermaid
graph TD
    subgraph Full["朴素 self-attention 全局"]
        F[查询像素 q] --- A1[整张图所有 HW 像素都参与]
        A1 --> Fcost[复杂度 O HW 平方 d<br/>256x256 显存爆]
    end

    subgraph Window["W-MSA 局部窗口 SwinIR"]
        W[查询像素 q] --- B1[只看自己所在的 8x8 窗口]
        B1 --> Wcost[复杂度 O HW M 平方 d<br/>SW-MSA 跨窗口靠 shift]
    end

    subgraph Channel["MDTA 通道维 Restormer"]
        C[查询像素 q] --- C1[空间上能看到所有 HW 像素<br/>但 attention 在通道间算]
        C1 --> Ccost[复杂度 O d 平方 HW<br/>对大图友好]
    end

    style Full fill:#ffebee
    style Window fill:#fff8e1
    style Channel fill:#e8f5e9
```

颜色分组反映"代价 vs 长距离能力"的权衡。红色 Full 是理想但跑不起来。黄色 Window 跑得起来，但每层只能看局部，长距离能力要靠多层累积。绿色 Channel 既能跑也能在每层"看到全图"，代价是 attention 不在像素间算而在通道间算，需要重新理解它在做什么，下面 §7.5 会详细展开。

## 7.2 朴素 ViT 的问题

直接把 Vision Transformer 用在低层视觉，几个直接的问题。

### 计算量爆炸

ViT 把图分成 $16 \times 16$ 的 patch。$256 \times 256$ 图分成 $16 \times 16 = 256$ 个 patch，attention 是 $256 \times 256$，可接受。$1024 \times 1024$ 图分成 $64 \times 64 = 4096$ 个 patch，attention 是 $4096 \times 4096$ 即 1600 万个元素，**显存爆**。低层视觉的输入分辨率往往比分类大一个数量级（分类是 224×224，去噪/超分常常是 1024×1024 或更大），ViT 的"分类专用预设"根本撑不住。

### Patch 粒度太粗

ViT 的 $16 \times 16$ patch 适合分类（语义级），不适合像素级任务。低层视觉需要**像素级细节**：SR 要恢复每个像素，去噪要保留每个像素的高频成分。把 16×16 像素打包成一个 token，等于在 attention 之前就把高频信息搅在一起，模型再怎么 attention 也找不回这些信息。

低层视觉的 Transformer 要么用更小的 patch（1×1 或 2×2，等价于不打包），要么用别的方式保持像素级表达（比如 SwinIR 的 1×1 patch + 窗口划分）。

### 缺少归纳偏置

CNN 的局部性（locality）和平移等变性（translation equivariance）是低层视觉的合理归纳偏置：相邻像素相关、同一种纹理出现在不同位置应该被同样处理。ViT 完全抛弃了这些，需要更多数据才能学到。

低层视觉数据集相对小（DF2K 几千张），不像分类有 ImageNet 一千万级。**ViT 在数据少的设置下劣势明显**。这也是 SwinIR 选择"窗口 + 相对位置编码"而不是纯 ViT 的原因：窗口划分把局部性偏置加回来，相对位置编码把平移等变性偏置加回来。

## 7.3 SwinIR（2021）：窗口注意力

Liang et al. 把 Swin Transformer 引入低层视觉，提出 SwinIR，一个覆盖 SR、去噪、去模糊三任务的通用架构。

### 核心思想：局部窗口 self-attention

不在全图上做 attention，而是把图分成 $M \times M$ 的窗口（典型 $M = 8$），**每个窗口内独立做 self-attention**。

- 窗口数量：$\frac{H}{M} \times \frac{W}{M}$
- 每个窗口的 attention：$M^2 \times M^2$
- 总复杂度：$\frac{H}{M} \times \frac{W}{M} \times M^4 \cdot C = HW \cdot M^2 \cdot C$

对比朴素 attention：$H^2 W^2 \cdot C$。当 $M = 8$ 时，复杂度从 $O((HW)^2)$ 降到 $O(HW \cdot 64)$，**从二次方降到线性**。$256 \times 256$ 输入下，朴素 attention 是 $256^4 = 4.3 \times 10^9$ 个 attention map 元素，W-MSA 是 $256^2 \times 64 = 4.2 \times 10^6$，降了三个数量级。

### Shifted Window 解决窗口隔离问题

W-MSA 的问题：窗口之间没有信息交换。第二层的 attention 仍然只能看到第一层的同一个窗口里的信息，长距离依赖在层数足够多之前完全建立不起来。

解决方案：**SW-MSA**（Shifted Window MSA）。每隔一层把整个窗口划分偏移 $M/2$，让相邻层的窗口边界错开。这样一个像素在第 1 层属于窗口 A，在第 2 层属于窗口 B，A 和 B 部分重叠，像素之间的信息通过这种"重叠传播"在多层后覆盖全图。

```
Layer 1 (W-MSA):           Layer 2 (SW-MSA):
+----+----+----+----+      +-+----+----+----+-+
|    |    |    |    |      | |    |    |    | |
+----+----+----+----+      +-+----+----+----+-+
|    |    |    |    |      | |    |    |    | |
+----+----+----+----+      +-+----+----+----+-+
|    |    |    |    |      | |    |    |    | |
+----+----+----+----+      +-+----+----+----+-+

窗口对齐                    窗口偏移 M/2
```

这样一次 W-MSA + 一次 SW-MSA 后，每个像素的有效感受野覆盖了 $2M \times 2M$ 区域。再过一对 W-MSA + SW-MSA 又翻倍，理论上 $\log(\max(H, W) / M)$ 对就能覆盖全图，SwinIR 的 6 个 RSTB × 6 个 STL = 36 层足以让任何位置的像素看到全图。

把"局部窗口 + shift"和"全局 attention"在同一张特征图上的可见范围画在一起：

```mermaid
graph LR
    subgraph FullView["全局 attention 视野"]
        F1[像素 q 在图中央] -.attention 到.-> F2[整张图所有像素]
        F2 --> Fbad[显存 O HW 平方]
    end

    subgraph L1View["SwinIR Layer 1 W-MSA 视野"]
        L1q[像素 q] -.attention 到.-> L1w[所在 8x8 窗口内 64 个像素]
    end

    subgraph L2View["SwinIR Layer 2 SW-MSA 视野"]
        L2q[同一像素 q 被划入新窗口] -.attention 到.-> L2w[新窗口里又包含来自旧窗口 A 和邻窗口 B 的像素]
        L2w --> L2info[信息通过窗口重叠跨界传播]
    end

    style FullView fill:#ffebee
    style L1View fill:#fff8e1
    style L2View fill:#e8f5e9
```

红色 Full 一次性看完所有像素但跑不起来；黄色 W-MSA 一层只看 64 个；绿色 SW-MSA 让下一层的窗口和上一层错开，跨界信息靠层数累积。SwinIR 选择"小窗口 + 浅层数"是显存可控的设计；HAT 的 OCAB 则进一步把窗口做大并允许重叠，下面 §7.6 会讲。

### Window Attention 的实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """把 (B, H, W, C) 切成 (B*num_windows, M, M, C)。"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int,
                   H: int, W: int) -> torch.Tensor:
    """切回 (B, H, W, C)。"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, H, W, -1)


class WindowAttention(nn.Module):
    """W-MSA: 窗口内 self-attention, 带 relative position bias。"""

    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        # Relative position bias: 让模型学窗口内不同相对位置的偏好
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords = coords.flatten(1)
        rel = coords[:, :, None] - coords[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        rel_index = rel.sum(-1)
        self.register_buffer("relative_position_index", rel_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*num_windows, M*M, C)
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # 各 (B_, head, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B_, head, N, N)

        # 加 relative position bias
        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)
```

**Relative position bias 在做什么。** 普通 ViT 用绝对位置编码：给每个 token 一个固定的位置向量。SwinIR 用相对位置 bias：给"窗口内两个 token 之间的相对偏移"一个可学的标量，直接加到 attention 分数上。这种 bias 满足"平移等变"，是低层视觉里很合理的归纳偏置：同样的纹理出现在窗口的左上还是右下，相对位置关系不变。

### SwinIR 的架构

SwinIR 在 LR 空间做特征提取，每个 stage 由几个 RSTB（Residual Swin Transformer Block）组成。RSTB 内部是 W-MSA / SW-MSA 交替。

```
LR Input
  ↓ Conv (shallow feature)
  ↓ RSTB × 6 (deep features)
  │   每个 RSTB:
  │     STL × 6 (Swin Transformer Layer)
  │       W-MSA → MLP
  │       SW-MSA → MLP
  ↓ Conv + global residual
  ↓ Upsample (PixelShuffle)
  ↓ Conv
HR Output
```

性能：在 Set5 4× 上约 32.9 dB，超过 RCAN（32.6 dB）和所有 CNN 模型。

## 7.4 SwinIR 的局限

W-MSA 虽然把注意力的计算复杂度压回了线性，但它的长距离交互能力依然受限：

- 单层感受野被限制在 $M = 8$ 的局部窗口内
- 跨窗口信息传递只能靠 SW-MSA 每隔一层移动半个窗口逐步扩散
- 真正跨越全图尺度的依赖仍然要堆叠几十层才能建立起来

更重要的是，SwinIR 的"全图可见性"是靠多层中转间接实现的，每次中转都会损耗细节信号。HAT 论文通过局部归因图（LAM）可视化发现，SwinIR 在重建某个局部区域时实际利用到的输入像素只占理论可见范围的约 30%，其余区域虽然在理论感受野内，梯度贡献却已衰减到几乎可以忽略。

下一节介绍的 Restormer 换了一个方向破局：把自注意力从空间维挪到通道维，直接绕开空间分辨率带来的平方爆炸。

## 7.5 Restormer（2022）：通道维度 Attention

Zamir et al. 提出的 Restormer 是去噪、去模糊、去雨等复原任务的代表性架构，核心创新是将自注意力矩阵的计算从空间维度转置到通道维度。

### MDTA：Multi-Dconv Head Transposed Attention

标准自注意力的计算式为：

$$
\text{attention}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d}) V
$$

其中 $Q, K, V \in \mathbb{R}^{N \times d}$，$N = HW$ 是展平后的空间像素数，$d$ 是单头通道维度。由于 $QK^T$ 的尺寸是 $N \times N$，时间与空间复杂度均为 $O(N^2 d)$。当输入是高分辨率图像时，$N$ 动辄数万到数十万，平方项直接成为瓶颈。

Restormer 的转置注意力（MDTA）将 $Q, K, V$ 视为 $\mathbb{R}^{d \times N}$，先沿空间维 $N$ 对 $Q, K$ 做 L2 归一化得到 $\hat{q}, \hat{k}$，再交换相乘顺序：

$$
\text{attention}(Q, K, V) = V \cdot \text{softmax}\!\left( \alpha \cdot \hat{k} \hat{q}^T \right)
$$

其中 $\alpha$ 是每个 head 独立学习的可训练温度标量。注意这里用 L2 归一化后的余弦相似度配合可学习温度 $\alpha$，替代了标准注意力里除以 $\sqrt{N}$ 的固定缩放。此时相关性矩阵 $\hat{k} \hat{q}^T$ 的尺寸变成了 $d \times d$，整体复杂度随之降为 $O(d^2 N)$，与空间像素数 $N = HW$ 呈严格的线性关系。

由于单头通道数 $d$ 通常很小（典型值 24–64），而空间像素数 $N = HW$ 在 $256 \times 256$ 下就已达 $65536$，因此 $d^2 \ll N^2$。这使 Restormer 能够在高分辨率大图上直接前向与训练，不会触发空间注意力矩阵的显存溢出。

### 这个 attention 在做什么

理解 MDTA 的关键是看清"谁与谁在计算内积"：

- **标准空间自注意力**：每个空间位置与其余所有空间位置计算通道向量的内积，得到 $N \times N$ 的空间权重图，再用它对 $V$ 做空间维加权求和，本质是"用全图相似位置的特征更新当前位置"
- **MDTA（转置通道注意力）**：每个通道与其余所有通道在全图空间维上计算内积（即两张特征图在全图 $HW$ 个像素上的空间互相关），得到 $d \times d$ 的通道协方差矩阵，再用它对 $V$ 做通道维线性组合，本质是"根据全图通道间的共现关系重整每个像素的通道表达"

从特征交互的角度看：

- 不同通道各自编码了不同属性（例如边缘方向、高频纹理、背景色温、噪声强度）
- 同一张图像内，这些特征通道之间存在全局统计耦合（例如某类边缘通道激活时，特定方向的纹理通道往往同步响应）
- MDTA 在全图范围内统计任意两个通道的共现强度，动态生成一张针对当前输入图像的 $d \times d$ 通道混合矩阵

因此，MDTA 可以看作 SE/RCAN 通道注意力的满秩推广：SE 只给每个通道乘一个标量权重（相当于对角矩阵），而 MDTA 学出的是完整的 $d \times d$ 通道混合矩阵，且矩阵元素由输入图像的全图空间统计实时决定。

那么空间信息是如何参与进来的？首先，在进入通道内积之前，$Q, K, V$ 先经过了一层 3×3 深度可分离卷积（depthwise conv）提取局部空域上下文；其次，通道相关性矩阵 $\hat{k} \hat{q}^T$ 的每一个元素都是两个通道在全图 $HW$ 个像素上的点积，天然汇聚了全图统计信息；最后，矩阵乘回 $V \in \mathbb{R}^{d \times HW}$ 时保留了完整的空间分辨率。

下图将标准空间自注意力与 MDTA 的数据流并列对比，维度转置前后的矩阵尺寸差异一目了然：

```mermaid
graph TD
    subgraph Vanilla["朴素 self-attention 沿 token 维"]
        V1[Q reshape C HW] --> V2[Q 转 HW C]
        V3[K reshape C HW] --> V4[K 转 HW C]
        V5[V reshape C HW] --> V6[V 转 HW C]
        V2 --> V7[Q K 转 -> HW HW 矩阵<br/>显存爆 当 HW 大]
        V4 --> V7
        V7 --> V8[softmax along HW]
        V8 --> V9[× V -> HW C]
        V6 --> V9
    end

    subgraph MDTA["MDTA 沿 channel 维"]
        M1[QKV 1x1 conv -> 3C HW] --> M2[depthwise 3x3 加局部信息]
        M2 --> M3[split Q K V 各 C HW]
        M3 --> M4[Q K 沿 HW 做 L2 归一化]
        M4 --> M5[Q K 转 -> C C 小矩阵<br/>显存小 与 HW 无关]
        M5 --> M6[softmax along C<br/>每个 head 一个温度]
        M6 --> M7[× V -> C HW]
        M7 --> M8[1x1 conv proj_out]
    end

    style Vanilla fill:#ffebee
    style MDTA fill:#e8f5e9
```

两条 pipeline 的差别集中在中间的 attention 矩阵尺寸：vanilla 是 $HW \times HW$，MDTA 是 $C \times C$。对 $256 \times 256 \times 96$ 的特征图，vanilla 要 $65536 \times 65536$（17 GB FP32），MDTA 只要 $96 \times 96$（37 KB）。这是 Restormer 在大图上可用的根本原因。

### Depthwise Convolution 加局部性

MDTA 把 attention 搬到通道维后，自然失去了"空间局部性"：通道间 attention 不关心像素位置，对 SR/去噪这种局部任务并不友好。Restormer 的补救是在 $Q, K, V$ 投影前先做 3×3 depthwise conv，给 attention 注入空间局部信息：

```python
class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention (Restormer)."""

    def __init__(self, dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 1x1 conv 生成 QKV
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        # depthwise conv 给 QKV 加空间局部信息
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, 3, padding=1,
                                    groups=dim * 3, bias=bias)
        self.proj = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # reshape: (B, C, H, W) -> (B, head, C/head, HW)
        q = q.view(B, self.num_heads, C // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)

        # 沿 token 维 (HW) L2 归一化, 让 K@Q^T 是 cosine 相似度
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 关键: 在 channel 维 softmax, 不是在 token 维
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # (B, head, c/h, c/h)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).view(B, C, H, W)
        return self.proj(out)
```

### Gated-Dconv Feed-Forward Network

Restormer 的 FFN 也做了修改，把普通 MLP 替换成 GDFN（Gated-Dconv Feed-Forward Network）：

```python
class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network."""

    def __init__(self, dim: int, ffn_expansion: float = 2.66, bias: bool = False):
        super().__init__()
        hidden = int(dim * ffn_expansion)
        self.proj_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1,
                                groups=hidden * 2, bias=bias)
        self.proj_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2          # gating
        return self.proj_out(x)
```

注意 `F.gelu(x1) * x2` 这个 gating，和 NAFNet 的 SimpleGate 是同样的思想，让模型自适应地决定每个通道在当前位置激活多少。这种"门控 FFN"在 Transformer 派和 CNN 派都已经成为现代默认，可以理解为对 vanilla FFN 的一次普适性改进。

### Restormer 整体

U-Net 形状的 encoder-decoder + skip connection + MDTA/GDFN block。每个 encoder/decoder level 由若干个 Transformer block 堆叠，level 之间通过 PixelShuffle/PixelUnshuffle 改变分辨率。在去噪、去模糊、去雨等任务上是 2022-2023 年的事实 SOTA，Real-World 数据集上比 SwinIR 提升约 0.3-0.5 dB。

它的工程优势在大图推理上特别明显：同样硬件下能跑的最大输入分辨率比 SwinIR 大一两倍，这对视频去噪和高分辨率照片修复是关键。

## 7.6 HAT（2023）：混合注意力

Chen et al. 在 SwinIR 基础上提出 HAT（Hybrid Attention Transformer），是 2022–2023 年这一代混合注意力架构的代表；2024 年之后，DRCT、ATD 等工作又在学术超分基准上进一步推高了指标。

### 关键观察

SwinIR 的窗口尺寸只有 $8 \times 8$。HAT 论文通过局部归因图（LAM，Local Attribution Map）可视化发现，SwinIR 在重建时实际激活利用的输入像素只占可见视野的约 30%。

这说明单靠小窗口自注意力并未能把模型的感受野潜力充分调动起来。HAT 的设计目标很直接：在局部窗口注意力的基础上引入通道全局统计与跨窗口直接交互，让模型激活更广范围内的输入像素。

### 两类 attention block 组合

HAT 的每个残差组（Residual Group）内部由两类模块串联配合：

1. **HAB**（Hybrid Attention Block）：在同一个模块内并行组合 W-MSA 与 CAB
   - **W-MSA**（窗口自注意力）：沿用 SwinIR 的局部窗口 + 移位窗口机制，但将窗口边长从 SwinIR 的 8 直接扩大到 16。扩大窗口尺寸本身就是 HAT 的核心改动之一，单层直接覆盖的空间面积随之扩大到原来的 4 倍
   - **CAB**（Channel Attention Block）：并联一条 RCAN 风格的通道注意力卷积支路，利用全局池化引入全图通道统计，同时补充局部卷积归纳偏置
2. **OCAB**（Overlapping Cross-Attention Block）：置于每个残差组末尾的跨窗口交互模块
   - 查询 $Q$ 仍按原窗口大小 $M=16$ 划分，而键 $K$ 和值 $V$ 则在向四周外扩的重叠窗口（overlap ratio 取 0.5 时有效窗口边长为 24）上提取，在两套窗口之间直接计算交叉注意力
   - 这样每个窗口都能在单层内直接读取相邻窗口边缘的特征，不必完全依赖 SW-MSA 的多层间接移位传递

### 性能

HAT-L（大参数版本）在 Set5 4× 上达到约 33.0–33.4 dB（视是否使用 ImageNet 预训练而定），相比 SwinIR 提升约 0.4–0.5 dB，在经典 bicubic 超分基准上属于相当可观的跨越。

代价则是参数量增至约 40M、推理显存与延迟显著上升。因此生产环境较少直接照搬 HAT-L，但它清晰验证了"扩大窗口 + 通道全局统计 + 跨窗口重叠交互"这套组合的有效性。

这里也补充一条并行的架构脉络：自注意力之外，2024 年前后兴起的状态空间模型（State Space Model）路线，以 Mamba 及其图像复原变体 MambaIR、MambaIRv2 为代表，用线性复杂度的选择性扫描替代自注意力，成为与窗口注意力和通道注意力直接竞争的另一分支。它在长序列与高分辨率大图上的复杂度优势明显，具体谱系留到第 18 章展开。

## 7.7 为什么 attention 对低层视觉有用

到这里可以归纳 attention 在低层视觉的具体贡献。

### 长距离自相似性

§7.1 提过：自然图像有 self-similarity，远处的相似 patch 可以帮助恢复。CNN 通过堆深度间接做，attention 直接做。

具体场景：

- **去噪**：远处干净的相似纹理 → 当前噪声位置的去噪先验
- **超分**：图中重复的边缘/角点 → 借用高频细节
- **去模糊**：同一物体的不同视角实例 → 联合约束去模糊结果

### 退化程度的不均匀性

真实图像里不同位置的退化程度不一样：

- 暗部噪声大、亮部噪声小
- 中心清晰、边缘模糊（镜头像差）
- 主体在焦内、背景在焦外
- 局部过曝、局部正常

CNN 的卷积是**位置无关**的：同一个 kernel 处理所有位置。attention 是**位置相关**的：可以让"轻退化区域"的信息流到"重退化区域"，自适应地分配修复力度。

### 通道维度的特征聚合

MDTA 揭示的：通道之间也有 dependency。channel attention（SE/CA）是简化版，MDTA 是完整版。这种通道间的信息交互在 CNN 里需要靠 1×1 卷积层间接做，效率低，且不能根据输入内容自适应调整通道间的混合权重。

## 7.8 计算效率分析

不同 attention 的计算量对比（输入 $256 \times 256$，$C = 96$）：

| 方法 | 复杂度 | $256 \times 256$ FLOPs | 显存（attention map） |
|------|-------|----------------------|---------------------|
| Vanilla ViT | $O((HW)^2 C)$ | ~16 TFLOPs | ~16 GB |
| SwinIR (M=8) | $O(HW \cdot M^2 C)$ | ~250 GFLOPs | ~256 MB |
| Restormer (MDTA) | $O(C^2 \cdot HW / \text{head})$ | ~500 GFLOPs | ~64 MB |
| HAT | SwinIR + CAB + OCAB | ~600 GFLOPs | ~512 MB |

工程意义：

- 朴素 ViT 完全不可用（除非小图）
- SwinIR/Restormer 都是"线性复杂度"（FLOPs 与 $HW$ 成线性）
- Restormer 的显存最低，对大图友好，这是它在视频去噪等大尺寸应用里的关键优势

## 7.9 移动端的 attention

理论上 attention 复杂度可控，但**端侧部署 attention 的瓶颈不在 FLOPs，在内存带宽和算子支持**。

### softmax 是瓶颈

NPU/移动 GPU 对 GEMM（General Matrix Multiply，矩阵乘）有硬件加速，对 softmax 没那么好。一个 attention 块的 softmax 可能比矩阵乘本身慢 2-3 倍，硬件流水线在 softmax 上利用率掉一半。

### Reshape 开销

attention 实现里大量 reshape/permute（把 $(B, C, H, W)$ 折成 $(B, \text{head}, C/\text{head}, HW)$ 之类）。在某些 NPU 上 reshape 是 memory-bound 操作，要把数据从一种内存布局搬到另一种，比浮点运算还慢。

### 实际工程

2026 年的端侧增强模型（手机 NPU）几乎全是 CNN，原因：

- 移动端 NPU 对 CNN 的优化最成熟（硬件 + 编译器 + 内存布局）
- attention 的 latency 不稳定（取决于输入大小），给推理引擎增加困难
- 端侧 OCR/AI 滤镜对延迟极敏感（< 30 ms），attention 模型的不稳定 latency 很难满足

何时端侧用 attention：

- **少量 attention block + 大量 CNN**（混合架构），让 attention 只出现在低分辨率瓶颈层
- **shape 固定的输入**（比如固定 $720 \times 1280$ 视频帧），编译器能做静态优化
- **用 linear attention 变体**（MobileViT 系列），避开 softmax

第 15 章会详细讲端侧部署。

## 7.10 Transformer 不擅长什么

Transformer 在低层视觉不是万能的，几个**不适合**的场景。

### 极端轻量级（< 1M 参数）

attention 本身有"基础开销"：QKV 投影、位置编码、相对位置 bias 等。这些固定开销在小模型里占比大。

实测：1M 参数预算下，纯 CNN 比 SwinIR 风格的 attention 模型在 PSNR 上高 0.2 dB。

### 极小输入

输入 $< 128 \times 128$ 时，window attention 的窗口都覆盖不了几个，效率不如直接全图 attention 或者直接 CNN。

### 对锐利度极敏感的任务

attention 有 averaging 的本质（softmax 加权和）。轻微的 attention 偏差会让边缘略糊。在文档增强、二值化等任务上，CNN（特别是有 gradient loss 的）效果更好。

### 训练数据少的场景

attention 缺少 CNN 的归纳偏置，需要更多数据才能学到一样的能力。如果你的训练集 < 10K 张图，CNN 通常比 Transformer 好，这是 §7.2 缺归纳偏置的延续。

## 7.11 选型决策表

| 场景 | 推荐 |
|------|------|
| 通用 SR / 去噪 / 去模糊（学术 benchmark） | SwinIR / Restormer / HAT |
| 真实退化 SR（生产） | RRDB（Real-ESRGAN）或 NAFNet |
| 端侧部署 | NAFNet / MobileNet 风格 CNN |
| 视频去噪 / 去模糊 | Restormer（U-Net 形状对长序列友好） |
| 极小输入（< 128 px） | CNN |
| 极端轻量（< 1M 参数） | CNN |
| 高质量人脸修复 | RestoreFormer / CodeFormer（混合架构） |
| 扩散派增强 | UNet 内部用 Transformer block（第 8 章） |

## 7.12 一个简化的 SwinIR-style 模块

把上面的概念拼成一个最小可读的 Transformer block：

```python
import torch.nn as nn


class SwinTransformerBlock(nn.Module):
    """简化的 Swin Transformer Block。"""

    def __init__(self, dim: int, num_heads: int, window_size: int = 8,
                 shift_size: int = 0, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # x: (B, H*W, C)
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # Cyclic shift (SW-MSA)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size),
                           dims=(1, 2))

        # 窗口分割 + attention + reverse
        windows = window_partition(x, self.window_size)             # (B*nW, M, M, C)
        windows = windows.view(-1, self.window_size ** 2, C)
        # 注意: 完整 SW-MSA 还需要传入 attention mask, 防止 cyclic shift 后的
        # "环绕" 像素跨真实图像边界相互 attend。这里省略 mask 让代码可读,
        # 实际 SwinIR / Swin Transformer 的 self.attn 接受一个 mask 参数。
        attn_windows = self.attn(windows)
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))
        x = x.view(B, L, C)

        # 残差 + MLP
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x
```

实际 SwinIR 的代码在这上面加 attention mask（处理 SW-MSA 的环绕问题）+ RSTB（多个 STL 包一层残差）+ patch merging（如果分多 stage）+ 上采样头。完整 SwinIR 大约 1000 行。

## 7.13 小结

1. **CNN 的局部性是低层视觉的合理偏置**，但限制了长距离依赖
2. **朴素 ViT 在低层视觉不可用**：计算量爆、粒度粗、缺归纳偏置
3. **SwinIR 用窗口注意力 + shifted window 解决计算量**，是 SR/去噪/去模糊的统一架构
4. **Restormer 用通道维度 attention（MDTA）** 把复杂度从 $O(N^2)$ 降到 $O(C^2)$，是去噪/去模糊 SOTA
5. **HAT 通过组合三种 attention 进一步提升**，但参数量大、推理慢
6. **attention 对低层视觉有用**：自相似性、退化不均匀、通道间依赖
7. **端侧仍然是 CNN 主场**：attention 在 NPU 上效率不好
8. **CNN 和 Transformer 不是替代关系**：现代增强网络（特别是扩散）通常是混合架构

到这里 Part II 的判别式架构（CNN + Transformer）讲完。下一章进入扩散模型，这是过去三年这个领域最大的范式转变，把"判别式恢复"扩展到"生成式恢复"。

---

> 下一章 [扩散模型基础](08-diffusion.md) → 从 DDPM 到 LDM，为什么扩散能"无中生有"，以及它在影像增强里的特殊角色。
