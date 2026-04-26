# 第 13 章 · 视频不只是图像加时间

> "把图像增强模型每帧跑一遍，就是视频增强了。"
>
> 这是一个**初学者最常犯**的错误。
>
> 视频增强有时序一致性这个独立问题——单独把每一帧做"好"，串起来可能完全没法看。

## 13.1 一个直观的失败案例

拿一段低质量视频（720P 手机录制），用 Real-ESRGAN 4× 每帧独立超分到 4K：

- **单帧看**：每一帧都更清晰，细节锐利
- **当作视频播放**：边缘**抖动**、纹理**闪烁**、细节**鬼影**

具体症状：

- 一个静止的物体，每一帧的细节略有不同
- 眼睛追踪一个移动物体时，物体表面在"沸腾"
- 平坦区域（天空、墙面）出现一闪一闪的纹理

为什么？

> Real-ESRGAN 是**生成式**模型——它在每张图上"生成"细节。
> 同一物体在两帧中略有差异（噪声、压缩），生成的细节就不同。
> 视觉上：**时序不一致** = 闪烁。

这个问题不是 Real-ESRGAN 的 bug，是**所有不考虑时序的模型在视频上都会出现的现象**。

## 13.2 时序一致性是独立问题

把视频增强目标重新定义：

- **空间质量**：每帧本身够清晰、细节够丰富（图像增强目标）
- **时序一致性**：相邻帧之间的变化只反映真实场景的变化，不引入"模型噪声"

这两个目标**部分冲突**：

- 让模型生成更多细节 → 同一物体在不同帧的细节可能不一致
- 让模型在时序上更稳定 → 容易输出"安全模糊"

视频增强的工程难点就是**同时优化这两个目标**。

## 13.3 时序一致性的物理基础：光流

什么叫"时序一致"？数学上：相邻两帧之间，对应像素的颜色应该（基本）不变。

"对应像素"由**光流**（optical flow）定义：物体在屏幕空间的运动矢量场。

$$
F_t \to t+1 (x, y) = (u, v)
$$

意思是：第 $t$ 帧 $(x, y)$ 位置的像素，在第 $t+1$ 帧移动到了 $(x + u, y + v)$。

**Brightness Constancy 假设**：

$$
I_{t+1}(x + u, y + v) \approx I_t(x, y)
$$

如果增强模型遵守这个假设，输出视频在时序上就一致；不遵守就闪烁。

### 光流的工程

主流光流估计方法：

- **传统**：Lucas-Kanade、Farneback、TV-L1（OpenCV 提供）
- **深度学习**：FlowNet → PWC-Net → **RAFT** （2020，目前最常用）→ GMA、SEA-RAFT

```python
# 用 RAFT 估计光流 (推荐用 torchvision 的预训练版本)
from torchvision.models.optical_flow import raft_large

model = raft_large(weights='C_T_SKHT_V2', progress=False).eval().cuda()

def estimate_flow(frame_a: torch.Tensor, frame_b: torch.Tensor) -> torch.Tensor:
    """
    frame_a, frame_b: (B, 3, H, W) in [-1, 1]
    返回: (B, 2, H, W) flow from a to b (u, v)
    """
    with torch.no_grad():
        flow_list = model(frame_a, frame_b)
    return flow_list[-1]   # 最终迭代结果
```

### Warping：用光流变换图像

给定光流 $F_{t \to t+1}$，可以把第 $t+1$ 帧 warp 到第 $t$ 帧的视角：

```python
import torch
import torch.nn.functional as F

def warp_with_flow(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    用光流把 image warp 到目标位置。
    image: (B, C, H, W)
    flow: (B, 2, H, W), 单位是像素位移
    返回: warped image
    """
    B, _, H, W = image.shape
    # 创建标准 grid
    yy, xx = torch.meshgrid(
        torch.arange(H, device=image.device),
        torch.arange(W, device=image.device),
        indexing='ij',
    )
    grid = torch.stack([xx, yy], dim=0).float()  # (2, H, W)

    # 加上 flow
    grid = grid.unsqueeze(0) + flow              # (B, 2, H, W)

    # 归一化到 [-1, 1] (grid_sample 的要求)
    grid_x = 2.0 * grid[:, 0] / (W - 1) - 1.0
    grid_y = 2.0 * grid[:, 1] / (H - 1) - 1.0
    grid_norm = torch.stack([grid_x, grid_y], dim=-1)  # (B, H, W, 2)

    return F.grid_sample(image, grid_norm, mode='bilinear',
                         padding_mode='border', align_corners=True)
```

Warping 是视频增强里反复使用的基础操作——用它来对齐相邻帧、做时序聚合、做时序一致性损失。

## 13.4 时序一致性损失

训练视频增强模型时，加一个时序一致性损失：

```python
def temporal_consistency_loss(
    model_outputs: list,    # [out_t, out_t+1, ...]  增强后的帧
    flows: list,            # [flow_t→t+1, ...]
) -> torch.Tensor:
    """
    用光流验证: 相邻帧的增强输出应该满足 brightness constancy。
    """
    loss = 0.0
    for t in range(len(model_outputs) - 1):
        out_t   = model_outputs[t]
        out_t_1 = model_outputs[t + 1]
        flow    = flows[t]

        # 把 out_{t+1} warp 回 t 时刻
        warped = warp_with_flow(out_t_1, flow)

        # 算遮挡 mask (光流不可靠的位置)
        occlusion_mask = compute_occlusion_mask(flow)

        # 在非遮挡位置算 L1
        loss = loss + (occlusion_mask * (out_t - warped).abs()).mean()

    return loss / max(1, len(model_outputs) - 1)
```

加到训练总损失里：

$$
\mathcal{L} = \mathcal{L}_{\text{spatial}} + \lambda \mathcal{L}_{\text{temporal}}
$$

$\lambda$ 经验值 0.1-0.5。

## 13.5 遮挡：光流不可靠的地方

Brightness constancy 假设在两种情况下失效：

- **遮挡（occlusion）**：物体被前景挡住，在 $t+1$ 帧消失（disocclusion 反过来）
- **运动模糊 + 大位移**：光流估计本身错

时序一致性损失必须**屏蔽这些位置**。常见的遮挡 mask 估计：

### 前后向一致性

如果前向光流 $F_{t \to t+1}$ 和后向光流 $F_{t+1 \to t}$ 经过 warping 后能"回到自己"，那这个像素是可信的：

```python
def forward_backward_check(flow_fwd: torch.Tensor,
                           flow_bwd: torch.Tensor,
                           threshold: float = 1.0) -> torch.Tensor:
    """
    返回 (B, 1, H, W) 的 mask, 1 表示像素可信。
    """
    # warp flow_bwd 到第 t 帧的视角
    warped_bwd = warp_with_flow(flow_bwd, flow_fwd)
    # 期望: flow_fwd + warped_bwd ≈ 0
    diff = (flow_fwd + warped_bwd).norm(dim=1, keepdim=True)
    return (diff < threshold).float()
```

## 13.6 视频退化的特殊性

视频不只是"多张图"，它的退化模型有视频专属内容：

### 视频压缩伪影

H.264/H.265/AV1 的压缩比 JPEG 复杂得多。除了块状伪影，还有：

- **运动估计误差导致的拖影**：编码器猜错运动方向，解码后物体边缘有"鬼影"
- **关键帧 (I 帧) 周期性质量震荡**：I 帧质量高，B/P 帧逐渐质量下降
- **GOP (Group of Pictures) 边界**：每 ~30 帧一个 GOP 边界，边界处压缩误差累积重置
- **Bitrate spikes**：高动态场景比特率突然下降，质量崩

### 运动模糊（运动相关）

相机抖动或物体快速移动导致的模糊。这种模糊**与运动方向相关**——同一帧不同位置可能有不同方向的模糊核。

视频去模糊的关键：**用相邻帧提供"清晰参考"**。如果第 $t$ 帧某个区域模糊，但第 $t+1$ 帧那个区域刚好清晰（运动停止瞬间），可以借用。

### 卷帘快门（Rolling Shutter）

CMOS 传感器逐行扫描，扫描时间内物体移动会产生几何变形：

- 静止物体保持原形
- 水平移动物体出现"歪斜"
- 旋转物体出现"螺旋扭曲"

这是手机视频特有的退化，老电影/胶片没有。专门的去卷帘算法存在但不在通用视频增强范围内。

### 视频专用合成 pipeline

第 5 章讲的图像退化 pipeline 在视频上**还要扩展**：

```python
class VideoDegradation:
    """视频退化合成 pipeline。"""

    def __init__(self, image_degrader_cls):
        # image_degrader_cls 应支持显式传入退化参数 (blur_kernel, noise_sigma, ...)
        # 这样我们可以一段视频共享一组参数, 而不是每帧重采样
        self.image_degrader_cls = image_degrader_cls

    def __call__(self, hr_video: torch.Tensor) -> torch.Tensor:
        """
        hr_video: (T, 3, H, W) high-quality video frames
        返回: lr_video (T, 3, H/4, W/4) degraded
        """
        T = hr_video.shape[0]

        # 1. 关键: 整段视频共享一组退化参数
        # 模糊核、噪声 sigma、JPEG quality 一段视频固定, 否则模型会学到
        # "每帧退化都不同" 的错误分布, 引入时序不一致
        clip_params = self._sample_clip_params()
        image_degrader = self.image_degrader_cls(**clip_params)

        lr_video = torch.stack([
            image_degrader(hr_video[t:t+1])
            for t in range(T)
        ], dim=0).squeeze(1)

        # 2. 视频专用退化 (整段一起处理)
        lr_video = self._add_motion_blur(lr_video)
        lr_video = self._video_compression(lr_video)
        lr_video = self._frame_drop_jitter(lr_video)  # 偶尔丢帧/重复帧

        return lr_video

    def _sample_clip_params(self):
        """采样一组退化参数, 整段视频共享。"""
        return {
            'blur_sigma': random.uniform(0.5, 3.0),
            'noise_sigma': random.uniform(0.005, 0.05),
            'jpeg_quality': random.randint(40, 95),
        }

    def _video_compression(self, video):
        """模拟 H.264/H.265 压缩。
        实际用 ffmpeg 编解码, 工程上通过 torchvision.io 或外部调用。
        """
        # 简化: 略
        return video

    def _add_motion_blur(self, video):
        """每帧加一个与运动方向相关的运动模糊。"""
        # 简化: 略
        return video

    def _frame_drop_jitter(self, video):
        """模拟丢帧/重复帧的 jitter。"""
        return video
```

视频退化合成的工程难点：

1. **大数据量**：一秒 30 帧、几分钟视频 = 几千张图
2. **时序一致**：同段视频的某些退化参数要固定，避免模型学到错误的退化分布
3. **可微 ffmpeg**：理想情况是可微 H.264 编码器，目前没有完美方案

## 13.7 几种主要的视频增强任务

| 任务 | 输入 | 输出 | 主要挑战 |
|------|------|------|---------|
| **Video SR (VSR)** | 低分辨率视频 | 高分辨率视频 | 时序一致 + 利用多帧信息 |
| **视频去噪** | 含噪视频 | 干净视频 | 利用相邻帧的"干净副本" |
| **视频去模糊** | 模糊视频 | 清晰视频 | 模糊核空间和时间变化 |
| **帧插值 (VFI)** | 低帧率视频 | 高帧率视频 | 中间帧不存在，必须生成 |
| **视频去抖** | 抖动视频 | 稳定视频 | 区分相机抖动和真实运动 |
| **视频上色** | 黑白视频 | 彩色视频 | 颜色时序一致 |
| **视频修复** | 有划痕/缺失 | 干净视频 | 利用前后帧补缺 |

## 13.8 视频增强的两种范式

按照"如何利用多帧信息"分两类：

### 滑动窗口（Sliding Window）

每次处理 $N$ 帧（典型 $N = 5$ 或 $7$），输出中间一帧的增强结果：

```
窗口 1: [F_1, F_2, F_3, F_4, F_5] → F_3'
窗口 2: [F_2, F_3, F_4, F_5, F_6] → F_4'
窗口 3: [F_3, F_4, F_5, F_6, F_7] → F_5'
...
```

代表：EDVR、ToFlow

优点：

- 训练简单（每个窗口独立）
- 推理可以并行（每个窗口独立）

缺点：

- 时序窗口有限（看不到 5 帧之外）
- 边界处理麻烦（视频开头结尾）

### 循环（Recurrent）

每一帧的处理利用前一帧的隐状态：

```
F_1 → F_1' + h_1
F_2, h_1 → F_2' + h_2
F_3, h_2 → F_3' + h_3
...
```

代表：BasicVSR、BasicVSR++、IconVSR

优点：

- 长时序信息可用（理论上无限）
- 推理时显存固定（只保留当前隐状态）

缺点：

- 训练复杂（BPTT）
- 推理必须串行（不能并行多帧）

### 混合：Bidirectional Recurrent

BasicVSR 引入了**双向**循环：

```
前向: F_1 → F_2 → F_3 → ...    产出 h^f_t
后向: ... → F_3 → F_2 → F_1    产出 h^b_t
合并: F_t' = G(F_t, h^f_t, h^b_t)
```

让每一帧能利用过去和未来的信息，是 VSR 的事实标准。第 14 章详谈。

## 13.9 时序对齐：让多帧信息真正可用

把多帧信息聚合时，必须先**对齐**——把其他帧 warp 到当前帧的视角。

### 显式对齐（用光流）

```python
def explicit_align(frames: torch.Tensor, ref_idx: int = None) -> torch.Tensor:
    """
    把所有帧 warp 到中心帧的视角。
    frames: (B, T, C, H, W)
    """
    if ref_idx is None:
        ref_idx = frames.shape[1] // 2

    ref = frames[:, ref_idx]
    aligned = []
    for t in range(frames.shape[1]):
        if t == ref_idx:
            aligned.append(ref)
            continue
        flow = estimate_flow(ref, frames[:, t])
        aligned.append(warp_with_flow(frames[:, t], flow))
    return torch.stack(aligned, dim=1)
```

优点：物理意义清晰、可解释。
缺点：依赖光流精度、光流出错就 cascade。

### 隐式对齐（可变形卷积）

EDVR 等模型用 **DCN (Deformable Convolution Network)** 学习对齐：

- 不显式估计光流
- 卷积核的采样位置是学习的
- 网络自动处理"看哪里"

优点：能处理光流难以估计的情况（半透明、复杂遮挡）。
缺点：CUDA 实现复杂、不易解释。

### 注意力对齐

更现代的：用 self-attention 跨帧聚合：

- 每个像素 attend 到相邻帧的所有像素
- 网络自动学到对应关系

代表：VRT、RVRT。计算量大但效果好。

## 13.10 视频增强的评估

视频评估在第 4 章和第 12 章的指标基础上加：

### 时序指标

- **tOF (temporal Optical Flow consistency)**：用光流验证相邻帧一致性
- **tLP (temporal LPIPS)**：相邻帧的 LPIPS 距离

```python
def temporal_lpips(frames: torch.Tensor, lpips_fn) -> float:
    """
    frames: (T, C, H, W)
    返回: 相邻帧之间的平均 LPIPS。越小越一致。
    """
    losses = []
    for t in range(frames.shape[0] - 1):
        losses.append(lpips_fn(frames[t:t+1], frames[t+1:t+2]).item())
    return sum(losses) / len(losses)
```

### 视频专用主观评测

不能只看单帧——必须看播放。流程：

1. 给评分人播放完整视频片段（5-10 秒）
2. 评 MOS 或 2AFC
3. 多次循环播放（让评分人能注意细节）

主观评测在视频里**比图像更重要**——闪烁是肉眼能看到但单帧 metric 看不到的问题。

### 评测时长

视频评测的样本量更大。原因：

- 闪烁/伪影偶发
- 不同场景表现不一
- 评分人需要多看几秒才能给评价

经验：视频主观评测每段 5-10 秒，每个模型至少 50-100 段视频。

## 13.11 视频增强 vs 视频生成的区别

为了避免混淆——**视频生成**（如 Sora、Kling、Veo）和**视频增强**是不同方向：

| 维度 | 视频增强 | 视频生成 |
|------|---------|---------|
| 输入 | 已有的低质量视频 | 文字 prompt（无视频输入）|
| 目标 | 提升质量、保留内容 | 创造一段新视频 |
| 时序约束 | 严格遵守原始时序 | 自由生成 |
| 评估 | 与 GT 比较 | 主观打分 / FVD |
| 当前 SOTA | BasicVSR++、Topaz Video AI | Sora、Kling、Veo3 |

这本书只覆盖前者。

## 13.12 视频增强的工程挑战

除了算法，工程上视频增强有几个独有难点：

### 数据存储和加载

- 视频比图像大几个数量级（每段 GB 级）
- 训练时随机采样片段
- 解码 H.264/H.265 是 CPU 瓶颈
- 推荐用 NVDEC 硬件解码

```python
# 用 torchvision.io 硬件解码 (PyTorch 2.0+)
import torchvision

reader = torchvision.io.VideoReader("video.mp4", "video", num_threads=4)
for frame in reader:
    pts = frame['pts']
    img = frame['data']
    # 处理...
```

### 推理时的内存管理

5 帧 4K 视频 = 5 × 4 × 3840 × 2160 × 4 bytes ≈ 400 MB（FP32）。模型激活几十 GB。**必须用 patch + tile + 流式处理**。

### 实时增强

视频会议/直播需要每帧 < 33ms（30 FPS）。这种场景：

- 不能用扩散模型
- 不能用厚 attention 网络
- 必须 CNN + 端侧加速
- 可能损失质量换速度

第 15 章会详谈端侧/实时部署。

## 13.13 小结

1. **视频增强不是图像增强 × N** —— 时序一致性是独立目标
2. **闪烁来自时序不一致** —— 单帧好不代表视频好
3. **光流是时序处理的基础** —— RAFT 是当前标准
4. **Warping + brightness constancy** 定义了"时序一致"
5. **遮挡 mask 必须**——光流不可靠的位置不能算损失
6. **视频退化更复杂**：H.264/H.265 压缩、运动模糊、卷帘快门
7. **滑动窗口 vs Recurrent**：两种范式，各有 trade-off
8. **双向 Recurrent (BasicVSR)** 是 VSR 的事实标准
9. **对齐方式**：显式光流、可变形卷积、跨帧 attention
10. **视频评估必须包含时序指标**（tOF/tLPIPS）+ **主观看播放**
11. **工程挑战**：数据存储、解码瓶颈、显存管理、实时性

下一章我们看具体的视频增强模型：BasicVSR++、RIFE 帧插值、视频修复——把这一章的概念落地到具体网络。

---

> 下一章 [VSR / 帧插值 / 视频修复](14-video-models.md) → BasicVSR++、RIFE、FILM 与视频去抖。
