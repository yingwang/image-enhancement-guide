# 第 14 章 · VSR / 帧插值 / 视频修复

> 第 13 章建立了视频增强的基础概念——时序一致、光流、对齐。
>
> 这一章看具体模型：BasicVSR++、RVRT、RIFE/FILM、视频去抖。
>
> 这些模型把第 13 章的概念落地，每个有不同的工程权衡。

## 14.1 这一章的结构

四个子任务：

```
14.2-14.6  视频超分 (VSR)：BasicVSR → BasicVSR++ → RVRT
14.7-14.8  帧插值：RIFE、FILM、AMT
14.9       视频去模糊
14.10      视频修复
14.11      视频去抖
```

每个任务给一个代表模型 + 几个工程要点。

## 14.2 视频超分（VSR）的演进

VSR 的演进路线和图像 SR 类似但晚两年：

```
2017  VESPCN     - 第一个端到端 VSR
2018  TDAN       - 隐式对齐（DCN）
2019  EDVR       - 滑动窗口 + DCN 对齐
2020  RBPN       - 循环 + 多帧补偿
2021  BasicVSR   - 双向循环 + 显式光流对齐
2022  BasicVSR++ - 二阶传播 + flow-guided DCN
2022  VRT        - 时序 Transformer
2023  RVRT       - 高效循环 Transformer
```

**BasicVSR++** 是 2022-2024 年的事实标准——简单、强、快。下面详谈。

## 14.3 BasicVSR++ 详解

Chan et al. 在 2022 年提出 BasicVSR++，是第 13 章 13.8 节讲的双向循环架构的代表。

### 整体结构

```
                    特征提取
LR frames ─────────────→ feature maps F_t
                              │
                              ↓
        ┌─────────────────────┴─────────────────────┐
        │                                            │
        ▼ Forward propagation                        ▼ Backward propagation
        h^f_1 ─→ h^f_2 ─→ ... ─→ h^f_T              h^b_T ─→ h^b_{T-1} ─→ ... ─→ h^b_1
        每一步用光流对齐前一步隐状态
        │                                            │
        └─────────────────────┬─────────────────────┘
                              ↓
                      Aggregation (concat + conv)
                              │
                              ↓
                      Upsample (PixelShuffle)
                              │
                              ↓
                      HR frames
```

### 关键创新 1：二阶传播

普通双向 RNN 每一步只用前一步：

$$
h_t = G(F_t, h_{t-1})
$$

BasicVSR++ 用**二阶**：

$$
h_t = G(F_t, h_{t-1}, h_{t-2})
$$

为什么有用？

- 一阶：$h_{t-1}$ 是从 $h_{t-2}$ warp 过来的，$h_{t-2}$ 又是从更早的 warp 过来的——光流误差一路累积
- 二阶：直接接触 $h_{t-2}$，绕过 $h_{t-1}$ 的累积误差
- **修正光流的局部错误**

### 关键创新 2：Flow-Guided Deformable Alignment

把光流和可变形卷积结合：

- 用光流给可变形卷积一个**初始的采样位置**
- 让 DCN 学习**对光流的修正**

```python
class FlowGuidedDCN(nn.Module):
    """Flow-guided deformable alignment (BasicVSR++)。"""

    def __init__(self, channels: int, num_groups: int = 8):
        super().__init__()
        # 预测 DCN 的 offset 修正量 (相对光流)
        self.offset_conv = nn.Conv2d(
            channels * 2 + 2,         # h_prev + features + flow
            num_groups * 2 * 9,        # 9 个采样点 × 2 维 × num_groups
            3, padding=1,
        )
        # 实际的 deformable conv (这里用 torchvision 的 DeformConv2d)
        from torchvision.ops import DeformConv2d
        self.dcn = DeformConv2d(channels, channels, 3, padding=1, groups=num_groups)

    def forward(self, h_prev, features_t, flow):
        """
        h_prev: 前一步隐状态 (B, C, H, W)
        features_t: 当前帧特征 (B, C, H, W)
        flow: 光流 (B, 2, H, W)
        """
        # 1. 用光流先把 h_prev warp 到当前视角
        warped_h = warp_with_flow(h_prev, flow)

        # 2. 网络预测 DCN offset (修正量)
        x = torch.cat([warped_h, features_t, flow], dim=1)
        offsets = self.offset_conv(x)
        # offsets shape: (B, num_groups * 2 * 9, H, W)

        # 3. 加上光流作为 offset 基准 (重要!)
        # 让 DCN 起始于光流的位置, 学的是修正量
        offsets = offsets + flow.repeat(1, offsets.shape[1] // 2, 1, 1)

        # 4. DCN 在修正后的位置采样
        return self.dcn(h_prev, offsets)
```

flow + DCN 组合的优势：光流提供物理意义，DCN 提供局部修正能力，**对快速移动 / 部分遮挡场景比纯光流稳**。

### BasicVSR++ 的训练

- **数据集**：REDS (240 个视频片段) + Vimeo-90K + 自合成的退化对
- **退化**：MM-CelebA 风格的视频退化 + REDS 标配的运动模糊和压缩
- **损失**：主要是 Charbonnier 重建损失（在每一输出帧上）—— 时序一致性主要靠**双向传播 + flow-guided alignment 的架构归纳偏置**自然涌现，而非显式时序损失项
- **训练时长**：1.6M 步在 8× A100，约 10 天

### 性能

在 REDS4 4× VSR 上 PSNR ~32.4 dB，明显高于 EDVR (31.1) 和 BasicVSR (31.4)。同时**保持实时性**——单帧 ~30ms 在 A100 上。

## 14.4 VSR 的训练数据

VSR 的数据要求比图像 SR 更高：

| 数据集 | 视频数 | 分辨率 | 用途 |
|-------|-------|-------|------|
| **REDS** | 270 (训) + 30 (测) | 720P | 通用 VSR 标准 |
| **Vimeo-90K** | 64,612 个 7-frame 片段 | 448×256 | 帧插值 + VSR |
| **Vid4** | 4 个片段 | 720P | 测试 |
| **UDM10** | 10 个片段 | 1080P | 测试 |
| **YouHQ40** | 40 个 4K 片段 | 4K | 真实高质量 |

### 视频退化合成

```python
def synthesize_video_pair(hr_video):
    """从 HR 视频合成 LR 训练对。"""
    
    # 1. 时序一致的退化 (同段视频用同样参数)
    blur_kernel = sample_blur_kernel()        # 一段视频固定
    noise_sigma = sample_noise_sigma()        # 一段视频固定
    
    lr_frames = []
    for hr_frame in hr_video:
        x = apply_blur(hr_frame, blur_kernel)
        x = downsample(x, scale=4)
        x = add_noise(x, noise_sigma)
        lr_frames.append(x)
    
    lr_video = torch.stack(lr_frames)
    
    # 2. 视频专用退化 (整段一起)
    lr_video = h264_compression(lr_video, bitrate=random.uniform(500, 5000))
    
    return lr_video
```

注意：**退化参数对一段视频固定**。这是和图像不同的地方——如果每帧用不同的退化参数，会引入"模型学到的不一致"。

## 14.5 VRT 与 RVRT：Video Restoration Transformer

Liang et al. 在 2022 年提出 VRT，2023 年改进为 RVRT。这条线把 Transformer 引入 VSR。

### 核心思想

不再用循环 RNN，而是**直接用 self-attention 跨帧聚合**：

```
F_1, F_2, F_3, F_4, F_5
   │   │   │   │   │
   └───┴───┼───┴───┘
           ▼
    Cross-frame attention
           │
           ▼
       聚合特征
```

### 计算量

天然的多帧 attention 复杂度爆炸——5 帧 $H \times W$ 的 attention 是 $(5HW)^2$。VRT 用窗口 + 沿时间轴的 attention 控制复杂度。

### VRT vs BasicVSR++

| 维度 | BasicVSR++ | VRT/RVRT |
|------|-----------|---------|
| 性能 | 强 | **更强**（PSNR +0.5 dB） |
| 速度 | 快 | 慢（2-3×） |
| 显存 | 中 | **大**（attention） |
| 实时性 | 可以 | 难 |
| 实现复杂度 | 简单 | 复杂 |

工程实践 2026 年：

- 离线高质量增强：RVRT 或 VRT
- 实时/接近实时：BasicVSR++
- 端侧：BasicVSR 或更轻量

## 14.6 帧插值（VFI）

帧插值是另一类视频任务——把低帧率视频提到高帧率（24 fps → 60 fps，60 fps → 240 fps）。

### 任务定义

给定相邻两帧 $F_t$ 和 $F_{t+1}$，生成中间帧 $F_{t+0.5}$。

注意 fini 的两个不同语境：

- **推理时**：用户给的视频里没有中间帧——这是任务难点
- **训练时**：标准做法是从高 FPS 视频（240 fps GoPro 等）取连续 3 帧，第一第三帧作为输入、第二帧作为真值监督——所以**训练时是有真值的**

### RIFE（2022）

Huang et al. 的 RIFE（Real-time Intermediate Flow Estimation）是当前帧插值的事实标准。核心创新：

- 不显式估计前向/后向光流，**直接估计中间帧到两端的光流**
- 一个 IFNet 同时输出 $F_{0.5 \to 0}$ 和 $F_{0.5 \to 1}$
- 用这两个光流分别 warp $F_0$ 和 $F_1$，融合得到 $F_{0.5}$

```python
class RIFEStub(nn.Module):
    """RIFE 简化骨架。"""

    def __init__(self):
        super().__init__()
        self.ifnet = IFNet()        # 输出中间帧到两端的光流
        self.fusion_net = FusionNet()  # 融合 warp 结果

    def forward(self, f0: torch.Tensor, f1: torch.Tensor) -> torch.Tensor:
        """
        f0, f1: 相邻两帧 (B, 3, H, W)
        返回: 中间帧 f_0.5
        """
        # 1. 估计中间帧到两端的光流
        flow_to_0, flow_to_1, mask = self.ifnet(f0, f1)

        # 2. 用光流 warp 两端帧
        warped_0 = warp_with_flow(f0, flow_to_0)
        warped_1 = warp_with_flow(f1, flow_to_1)

        # 3. mask 加权融合
        f_mid = mask * warped_0 + (1 - mask) * warped_1

        # 4. (可选) 用一个 refine 网络最后微调
        f_mid = self.fusion_net(f_mid, f0, f1)
        return f_mid
```

### RIFE 的优势

- **快**：原版 RIFE 在 1080P 上能跑 30 FPS
- **简洁**：单网络端到端
- **可扩展**：递归调用就能做 $4\times$, $8\times$ 帧率提升

### FILM（Google, 2022）

Reda et al. 的 FILM 用了不同思路——多尺度光流估计 + 渐进合成：

- 不依赖单步光流估计
- 在多个分辨率上递归细化
- **对大位移更鲁棒**（运动剧烈的场景）

实测：

- **慢速运动**（普通视频）：RIFE 和 FILM 接近
- **快速运动**（体育、舞蹈）：FILM 优于 RIFE

### AMT（2023）

更新的 SOTA：在 RIFE 基础上加 attention 模块，对**遮挡场景**特别强。

### 帧插值的失败模式

- **大位移**：物体跨度超过感受野，插出鬼影
- **新物体出现**（disocclusion）：相邻帧没有这个物体的信息，无法插出
- **半透明物体**：光流假设失效（玻璃、烟雾）
- **重复纹理**：光流容易匹配错位置（栅栏）

## 14.7 视频去模糊

视频去模糊和图像去模糊的不同：**相邻帧提供清晰参考**。

### 关键观察

视频里的模糊往往是**间歇性**的——某一帧模糊（运动瞬间），下一帧清晰（运动停止）。利用这个特性能大幅提升去模糊质量。

### EDVR

EDVR 不只是 VSR 的事实经典，也是视频去模糊的代表：

- 滑动窗口（5 或 7 帧）
- DCN 对齐
- 时空 attention 融合

### MIMO-UNet（多输入多输出）

不同分辨率上独立处理后融合——对不同尺度的模糊都能 cover。

### 数据：GoPro 数据集

视频去模糊的标准 benchmark：用高速相机拍摄（240 fps），把多个邻近帧平均得到"模糊帧"，原始帧作为真值。

## 14.8 视频修复（Inpainting / Restoration）

视频修复包含两类：

- **视频 inpainting**：补全被遮挡或被去除的区域
- **老电影修复**：去除划痕、闪烁、缺失帧

### Video Inpainting

给定一段视频和一个 mask（每帧标注要补全的区域），输出补全后的视频。

代表方法：**E2FGVI** (CVPR 2022)、**ProPainter** (ICCV 2023)

核心思路：

1. 用光流找到 mask 区域在其他帧的"对应像素"
2. 把这些信息聚合到当前帧
3. 用 transformer 在时空上融合

ProPainter 的关键改进：用一个 recurrent flow completion 模块，**先补全光流**（mask 区域光流也是缺的），再用补全的光流引导帧补全。

### 老电影修复

老电影的退化有几种特殊模式：

- **划痕**：随机位置的线条
- **闪烁**：整帧亮度/对比度抖动
- **缺失帧**：某些帧完全缺失
- **颜色衰减**：cyan/红色偏移

工程 pipeline：

```
Step 1: 划痕去除 (用相邻帧的对应位置)
Step 2: 闪烁稳定 (校正帧间亮度)
Step 3: 缺失帧补全 (用 RIFE 类插值)
Step 4: 颜色还原 (Lab 空间统计校正 + 学习的色彩还原)
Step 5: 增强 (VSR + 帧插值到 60 fps)
```

代表项目：

- **DeepRemaster**（Iizuka & Simo-Serra 2019）：老电影上色 + 修复
- **Bringing Old Films Back to Life**（Wan et al. 2022）：完整的电影修复 pipeline

## 14.9 视频去抖（Stabilization）

抖动来自相机移动（手持手机、动作相机）。**和真实运动区分**是难点。

### 经典方法

1. 用光流/特征点追踪估计相机的全局运动
2. 平滑这个全局运动轨迹
3. 用平滑后的轨迹反向 warp 每一帧

```
原始: 相机抖动 + 真实场景运动
     ↓
轨迹估计 + 平滑
     ↓
重 warp: 相机平滑 + 真实场景运动
```

### 现代方法

- **Stabnet**（学习的去抖）
- **Google 的 Pixel Camera 内置**（手机厂商各家方案）
- 部分手机用 IMU 数据辅助（陀螺仪比图像光流准）

### Crop 问题

去抖必然要**裁切边缘**——相机 warp 后画面边缘会留空白。一般的 trade-off：

- 强去抖：裁切多（视野变小）
- 弱去抖：裁切少（视野保留）

## 14.10 视频增强的工程组合

实际产品里通常不是单一模型，而是 pipeline：

### 例：手机视频后处理（vlog 增强）

```
原始 1080P 30fps 视频
  ↓ 去抖 (StabNet)
  ↓ 去噪 (BasicVSR 系)
  ↓ 帧插值 (RIFE) → 60 fps
  ↓ 超分 (BasicVSR++) → 4K
  ↓ 颜色校正 (LUT-based)
4K 60 fps 增强视频
```

每个步骤可能用不同模型，整体在 GPU 上跑约 5-10× 实时（5-10 秒 = 1 秒视频）。

### 例：直播视频增强

实时性要求严格（< 33ms/帧）：

```
原始 720P 30fps 直播流
  ↓ 轻量去噪 (NAFNet 小版本, < 5ms)
  ↓ 端侧超分 (ESRGAN-Lite 蒸馏, < 20ms) → 1080P
  ↓ 颜色 LUT (< 1ms)
1080P 30fps 增强流
```

不能用扩散、不能用厚 transformer、不能用滑动窗口（太慢）——只能用极轻量 CNN。

### 例：老电影修复（离线）

不要求实时，质量优先：

```
原始 480P 24fps 黑白老电影
  ↓ 划痕去除 (DeepRemaster)
  ↓ 闪烁稳定
  ↓ 上色 (DeepRemaster + 手工调整)
  ↓ 修复 (Bringing Old Films Back to Life)
  ↓ 帧插值 (RIFE) → 48 fps
  ↓ 超分 (Real-ESRGAN 调用每帧 + 时序一致性后处理)
2K 48fps 彩色修复版
```

## 14.11 评估视频增强模型

复习第 13 章 13.10 节，加几个具体指标：

| 指标 | 类型 | 工具 |
|------|------|------|
| PSNR / SSIM | 单帧 | 标准 |
| LPIPS | 单帧感知 | 标准 |
| **tOF** | 时序光流一致 | 自实现 |
| **tLPIPS** | 时序感知一致 | 自实现 |
| **VMAF** | 视频质量评估 | Netflix 开源 |
| **FVD** (Fréchet Video Distance) | 分布距离 | 生成式视频用 |

### VMAF

Netflix 在 2016 年推出的视频质量指标，融合多个子指标（VIF、ADM、运动评分），训练数据是真实人评。**生产环境视频质量的标准**。

```python
# 调用 VMAF (用 ffmpeg)
import subprocess

def compute_vmaf(reference_video, distorted_video):
    """用 ffmpeg + libvmaf 算 VMAF score。"""
    cmd = [
        'ffmpeg', '-i', distorted_video, '-i', reference_video,
        '-lavfi', 'libvmaf=log_path=vmaf.json:log_fmt=json',
        '-f', 'null', '-'
    ]
    subprocess.run(cmd)
    # 解析 vmaf.json
    import json
    with open('vmaf.json') as f:
        result = json.load(f)
    return result['pooled_metrics']['vmaf']['mean']
```

## 14.12 小结

1. **VSR 演进**：滑动窗口（EDVR）→ 双向 Recurrent（BasicVSR++）→ Transformer（RVRT）
2. **BasicVSR++ 是 2024 年事实标准**：双向循环 + 二阶传播 + flow-guided DCN
3. **VSR 数据要求高**：合成时退化参数对一段视频固定，避免引入不一致
4. **帧插值（VFI）三巨头**：RIFE（快）、FILM（大位移强）、AMT（遮挡强）
5. **视频去模糊** 利用相邻帧的清晰副本——这是图像去模糊没有的优势
6. **视频修复** 老电影是经典场景，工程是多步 pipeline 而不是单一模型
7. **视频去抖** 关键是区分相机抖动和真实运动
8. **生产 pipeline 是组合**：不是单模型，是去抖+去噪+插值+SR 的链
9. **实时增强严格受限**：< 33ms/帧只能用轻量 CNN
10. **VMAF 是生产视频质量评估的事实标准**

到这里 Part IV 视频两章完成。Part V 进入工程部署——前面讲了模型本身，这部分讲怎么把模型推到生产环境（量化、TensorRT、CoreML、移动端、tile）。

---

> 下一章 [推理优化](15-inference.md) → 量化、TensorRT、CoreML、torch.compile、动态分辨率、tile 推理。
