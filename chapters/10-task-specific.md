# 第 10 章 · 任务特化模型

> 通用增强模型（Real-ESRGAN、SUPIR）能处理大多数场景。但有几类任务**通用模型做不好**，必须用任务特化模型。
>
> 这一章讲：人脸、文档、医疗影像各自需要哪些归纳偏置，对应模型怎么设计。

## 10.1 为什么需要任务特化

通用 SR 在自然图像上表现优秀，但放到下面这些场景会翻车：

- **人脸**：通用模型修复出的脸常常**变了样**——眼睛大小不对、鼻形改了、肤色漂移
- **文字/文档**：通用模型把"日"修复成"目"——结构错了一笔
- **医疗影像**：通用模型在 X 光/MRI 上加了不存在的"病灶"——可能是事故级错误
- **卫星遥感**：通用模型把多光谱通道当 RGB 处理，光谱信息全错
- **科学显微镜**：通用模型不理解物理成像过程，重建出非物理结构

通用模型失败的共同原因：**它学到的先验是"自然图像的一般分布"，不是"这一类图像的特殊分布"**。

任务特化的核心 = 给模型注入这个特殊分布的知识：

- 人脸：身份不变（identity preservation）+ 五官几何约束
- 文档：字符级正确性 + 直线/曲线结构
- 医疗：物理成像模型 + 不允许"创造"
- 遥感：多通道光谱物理 + 大尺度地物结构
- 显微：成像理论（PSF、衍射）+ 物理重建

这一章按任务展开，重点是**人脸**（最成熟、工程实践最丰富），其他类型概述。

## 10.2 人脸增强：先验最强、模型最多的子领域

人脸增强是影像增强里**研究最深入**的子领域，原因有三：

1. **数据丰富**：FFHQ（70K）、CelebA-HQ（30K）、VFHQ 都是高质量大规模数据集
2. **应用价值高**：老照片修复、远程会议增强、视频通话美颜，市场巨大
3. **失败成本低（且高）**：低成本：娱乐应用错一点没关系；高成本：身份不能变（错一点就是别人了）

人脸增强的归纳偏置主要有三类：

### 偏置 1：人脸有强先验（StyleGAN 学到的）

StyleGAN（2019）和 StyleGAN2/3 已经把"人脸的分布"学得非常好。任何高质量人脸都可以**反演**（GAN inversion）成 StyleGAN 潜空间的一个 W 向量。

关键洞察：

> 人脸潜空间是低维的（StyleGAN W+ 维度与生成分辨率相关：1024px FFHQ 的 StyleGAN2 是 18×512 = 9216 维；512px 是 14×512）。
> 一张高质量人脸 = 这个低维空间里的一个点。
>
> 增强 = 从 LR 推出最合理的 W 向量，再用 StyleGAN 解码出 HR。

这是 **PULSE / GFPGAN / GPEN** 等模型的核心思路。

### 偏置 2：身份必须保留

人脸识别模型（ArcFace、FaceNet）已经学到了"什么决定一张脸的身份"。增强模型必须**让 ArcFace 看到的 embedding 不变**。

这是第 3 章 3.8 节讲过的身份保留损失：

```python
class IdentityLoss(nn.Module):
    """ArcFace 身份保留损失。"""
    def __init__(self, arcface_path: str):
        super().__init__()
        self.arcface = load_arcface(arcface_path).eval()
        for p in self.arcface.parameters():
            p.requires_grad_(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # 输入是对齐过的人脸, 范围 [-1, 1], 通常 112x112
        emb_pred   = self.arcface(pred)
        emb_target = self.arcface(target)
        return 1.0 - F.cosine_similarity(emb_pred, emb_target).mean()
```

### 偏置 3：对齐很重要

人脸有标准的几何结构——双眼水平、鼻子在中间、嘴巴下面。**对齐过的人脸**（face alignment）能让模型用更简单的网络达到同样效果，因为模型不需要学"五官位置可能在哪"。

工程实践：人脸增强模型几乎都先用 face detector + landmark detector 把人脸对齐到固定 crop（通常 512×512），增强完再贴回原图。

## 10.3 GFPGAN：StyleGAN 先验 + 通用 backbone

Wang et al. 在 2021 年的 GFPGAN（Generative Facial Prior GAN）是人脸增强的经典代表。

### 核心架构

```
LR Face (degraded)
  ↓ Encoder (类似 ESRGAN backbone)
  ↓ 提取多尺度特征 F_1, F_2, ..., F_n
  ↓
  ↓ 用某种方式调制 StyleGAN2 generator
  ↓
StyleGAN2 Generator (frozen, FFHQ pretrained)
  ↓
HR Face
```

关键点：

- **StyleGAN2 generator 冻结不训**——已经学到了人脸分布
- **训练的是 encoder + 一些调制层**——把 LR 信息映射到 StyleGAN 的潜空间和中间特征

### CS-SFT（Channel-Split Spatial Feature Transform）

GFPGAN 在 StyleGAN generator 的每一层注入 LR 特征，但不是直接 concat，是用 SFT（spatial feature transform）：

$$
F' = F \odot (1 + \gamma) + \beta
$$

其中 $\gamma, \beta$ 是从 LR 特征 $F_{\text{LR}}$ 预测的空间特征。这种调制让 LR 信息能精细地影响每个空间位置，但又不破坏 StyleGAN 的整体生成能力。

CS-SFT 的"CS"（Channel-Split）：把 StyleGAN 的特征在通道维分成两半，一半用 SFT 调制（接受 LR 信息），一半保留原 StyleGAN 输出（保留生成先验）。这样平衡 fidelity 和 quality。

### GFPGAN 训练损失

经典组合：

$$
\mathcal{L} = \lambda_1 \mathcal{L}_1 + \lambda_2 \mathcal{L}_{\text{percep}} + \lambda_3 \mathcal{L}_{\text{adv}} + \lambda_4 \mathcal{L}_{\text{id}} + \lambda_5 \mathcal{L}_{\text{component}}
$$

其中 $\mathcal{L}_{\text{component}}$ 是"五官局部 GAN 损失"——专门给眼睛、鼻子、嘴单独训判别器，强制每个局部都真实。

### GFPGAN 性能与局限

效果：在严重退化的老照片上效果惊艳，远超通用 SR。

局限：

- **依赖人脸对齐**：不对齐就不能用
- **依赖 FFHQ 分布**：对 FFHQ 不常见的脸（比如老人、小孩、特殊种族）效果下降
- **只能处理人脸**：必须配合通用 SR 处理背景

工程组合：**通用 SR（Real-ESRGAN）背景 + GFPGAN 人脸**。这是 2022-2024 年大多数老照片修复工具的标准 pipeline。

## 10.4 CodeFormer：codebook 离散化的优势

Zhou et al. 在 2022 年的 CodeFormer 用了不同思路——不依赖 StyleGAN，改用 **VQ-VAE 学到的离散 codebook**。

### 核心思想

把人脸的"局部特征"离散化为 codebook 里的若干 code（比如 1024 个 code），高质量人脸 = 这些 code 的某种组合。

流程：

```
LR Face
  ↓ Encoder
  ↓ Transformer (预测 code 序列)
  ↓ 从 codebook 查表得到对应特征
  ↓ Decoder
HR Face
```

为什么离散化有用？

- **离散化 = 强先验**：模型只能生成 codebook 里"见过"的特征，不会编造无意义的局部
- **Transformer 自然适合预测离散序列**：和 LLM next-token prediction 是同一范式
- **可调"控制强度"**：CodeFormer 提供 $w \in [0, 1]$ 让用户在"严格遵守 LR" vs "充分利用先验"之间调节

### CodeFormer 的 fidelity 调节

CodeFormer 的一个工程亮点是 `fidelity_weight` 参数。推理时：

```python
def codeformer_inference(model, lr_face, fidelity_weight=0.5):
    """
    fidelity_weight ∈ [0, 1]:
      0 -> 完全用 codebook 先验 (高质量但可能变样)
      1 -> 严格用 LR 信息 (低质量但保真)
    """
    return model(lr_face, w=fidelity_weight)
```

实际工程里常见的选择：

- 老照片修复（看起来好就行）：$w = 0.3$
- 视频通话美颜（不能变样）：$w = 0.7$
- 法律证据（绝对保真）：不用 CodeFormer，用判别式

### CodeFormer vs GFPGAN

| 维度 | GFPGAN | CodeFormer |
|------|--------|-----------|
| 先验来源 | StyleGAN2（连续） | VQ codebook（离散） |
| 推理速度 | 快 | 慢（多步 transformer） |
| Fidelity 可调 | 不可（固定） | 可调 $w$ |
| 对极端退化 | 容易"变脸" | 离散化让变化受限 |
| 工程友好度 | 中 | **高** |

2024 年起 CodeFormer 是人脸修复的更主流选择，工程灵活性是关键原因。

## 10.5 RestoreFormer / RestoreFormer++

Wang et al. 的 RestoreFormer（2022）用 **cross-attention** 直接连接 LR 特征和 codebook，去掉了 CodeFormer 的多步预测，速度快很多。

RestoreFormer++ 进一步优化，是 2023 年人脸修复速度/质量平衡最好的模型。

代码骨架：

```python
class RestoreFormerBlock(nn.Module):
    """简化的 RestoreFormer block。"""

    def __init__(self, dim: int, codebook_size: int, num_heads: int):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, lr_features: torch.Tensor) -> torch.Tensor:
        """
        lr_features: (B, N, D) - LR 编码出的特征序列
        cross-attn 到整个 codebook
        """
        codes = self.codebook.weight.unsqueeze(0).expand(
            lr_features.shape[0], -1, -1
        )  # (B, K, D)

        out, _ = self.attn(query=lr_features, key=codes, value=codes)
        return self.norm(out + lr_features)
```

## 10.6 老照片修复完整 pipeline

把上面的人脸技术组合起来，构成一个生产级 pipeline：

```python
def restore_old_photo(image_path: str, fidelity: float = 0.5):
    """老照片修复完整流程。"""

    img = load_image(image_path)

    # 1. 通用增强 (background)
    bg_enhanced = real_esrgan.enhance(img, scale=4)

    # 2. 检测 + 对齐人脸
    faces, landmarks = face_detector.detect(img)

    enhanced_faces = []
    for face_crop in extract_aligned_faces(img, landmarks):
        # 3. 人脸专用修复
        enhanced = codeformer.enhance(face_crop, fidelity_weight=fidelity)
        enhanced_faces.append(enhanced)

    # 4. 把增强人脸贴回背景
    final = paste_faces_back(bg_enhanced, enhanced_faces, landmarks)

    return final
```

这是 Topaz Photo AI、Tencent ARC、Adobe 等商业产品的简化版逻辑。

## 10.7 文档与文字增强

文档增强（OCR 之前的预处理）有完全不同的归纳偏置。

### 文字的特殊性

- **结构离散**：每个字符都是有限集合（汉字 ~5000 + 西文字符）
- **形状是 bilevel 的**：黑色笔画 + 白色背景
- **拓扑保持是核心**：把"日"修复成"目"是结构错（多了一笔），不是细节错

### 通用 SR 的失败

通用 SR 学到的是"自然图像的统计先验"——平滑、纹理、自然色彩。这些在文字上完全错误：

- 通用 SR 把笔画**模糊化** → 字看不清
- 通用 SR 试图加"自然纹理" → 字边缘出现雾状
- 通用 SR 不知道笔画结构 → 多/少一笔的错误

### 解决方案 1：DocSR / Text-SR

专门用文字数据训的 SR 模型。关键：

- **训练数据**：合成文字图（已知字符）+ 真实扫描文档对
- **损失加权**：在文字区域加大 L1 权重，强调像素级正确
- **失败模式针对性训练**：在合成数据里故意加"会让字错的"退化（强 JPEG、低分辨率）

### 解决方案 2：OCR 引导

把 OCR 模型作为额外的损失：

```python
class OCRGuidedLoss(nn.Module):
    """OCR 引导损失。
    输出图过 OCR, 字符识别错的位置加大像素损失权重。
    """
    def __init__(self, ocr_model):
        super().__init__()
        self.ocr = ocr_model.eval()

    def forward(self, pred, target):
        # 标准像素损失
        pixel_loss = F.l1_loss(pred, target, reduction='none')

        # OCR 在 pred 上的结果
        with torch.no_grad():
            chars_pred   = self.ocr(pred)
            chars_target = self.ocr(target)

        # 字符不一致的位置加 5× 权重
        char_mask = (chars_pred != chars_target).float()
        char_mask = expand_to_pixel_mask(char_mask)  # (B, 1, H, W)
        weighted = pixel_loss * (1 + 5 * char_mask)

        return weighted.mean()
```

### 解决方案 3：Bilevel 增强

文档大多数是 bilevel 的（黑字白底）。可以学一个**先二值化**再 SR 的两步流程：

```
LR 文档
  ↓ Binarization (Otsu / U-Net)
  ↓ Bilevel SR (专门训的 SR)
  ↓ 后处理 (反走样)
HR 文档
```

工程实践：文档增强商业产品（ABBYY、Adobe Scan）用上面这套。开源世界的 DocSR 系列、PaddleOCR 的 doc enhance 都是类似思路。

## 10.8 医疗影像增强

医疗影像（X 光、CT、MRI、超声）有最严格的约束。

### 核心约束：**不允许编造**

医疗诊断里"加一个不存在的病灶"是事故。所以：

- **生成式模型（扩散、GAN）原则上不能用**——它们会编造
- **判别式模型也要小心**——L1 训出来的也可能"加平滑"掩盖病灶
- **必须有物理约束**：成像物理（CT 的 Radon 变换、MRI 的 k-space 采样）必须建模

### 物理约束的形式

医疗增强通常是 inverse problem 的精确建模：

$$
y = A x + n
$$

其中 $A$ 是**已知的物理算子**（不是合成的退化）：

- CT：Radon 变换（投影到 sinogram 域）
- MRI：FFT + 欠采样掩码
- 超声：散射 + 衰减模型

可以用**展开网络**（unrolled networks）—— 把传统迭代算法（如 ADMM、共轭梯度）展开成神经网络，每一步既有数据保真项（强制 $\hat{x}$ 与 $y$ 一致），又有先验项（学习的）。

### 一个简化的展开网络结构

```python
class UnrolledMRIRecon(nn.Module):
    """简化的 MRI 重建展开网络。
    每步: 数据保真投影 + 学习的去噪。
    """

    def __init__(self, num_iter: int = 5):
        super().__init__()
        self.num_iter = num_iter
        self.denoisers = nn.ModuleList([UNet(in_ch=2, out_ch=2) for _ in range(num_iter)])

    def forward(self, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        y: (B, 2, H, W) under-sampled k-space (real, imag)
        mask: (B, 1, H, W) sampling mask (1=sampled)
        """
        # 初始化: zero-filled IFFT
        x = ifft2c(y * mask)

        for denoiser in self.denoisers:
            # 1. Denoise step (学习的)
            x = denoiser(x)

            # 2. Data consistency step (硬约束)
            x_kspace = fft2c(x)
            x_kspace = mask * y + (1 - mask) * x_kspace
            x = ifft2c(x_kspace)

        return x
```

数据保真步（`mask * y + (1 - mask) * x_kspace`）是关键——模型只能在**未采样**的 k-space 位置自由发挥，已采样的位置必须严格用真实测量值。这从工程上保证了"不编造已观测信号"。

### 工程实践

- 不要直接搬通用 SR 模型到医疗
- 有物理模型时优先用展开网络
- 必须有放射科医生参与评估（不能只看 PSNR/SSIM）
- 失败模式分析比平均指标更重要

医疗 AI 是一个独立的大领域，这本书只点到为止。深入要看专业的医疗影像处理教材。

## 10.9 卫星遥感与多光谱

卫星图有几个特殊点：

- **多光谱**：除了 RGB 还有 NIR、SWIR、热红外等多个通道（4-13 通道是常见的）
- **大尺度**：单张图可能 10000×10000 像素以上
- **物理意义**：每个像素值有具体物理含义（反射率、温度），不能随意改
- **应用是分析（不是观感）**：植被指数、水体提取、土地分类，这些任务对像素值精确度要求高

通用 SR 在这些场景的失败：

- 把多光谱当 RGB 处理 → 光谱信息全错
- "美化"输出 → 物理意义改变
- 不能处理大尺度图 → 必须 tile 但通用模型 tile 会有边界问题

任务特化方向：

- **HSI-SR**（高光谱图像超分）专门模型
- **物理约束损失**：保证某些波段比值（植被指数 NDVI）不变
- **tile + overlap blending**（第 9 章 9.9 节）

## 10.10 显微镜超分

显微镜成像有完整的物理理论：

- **PSF (Point Spread Function)** 决定了分辨率上限
- **衍射极限** 是物理约束
- **荧光显微**有独特的噪声模型（光子噪声主导）

主流方法：

- **物理建模 + 学习先验**：知道 PSF，用学习的去噪做后处理
- **STORM/PALM 类算法 + 神经网络加速**：超分辨显微镜
- **Cycle GAN 风格**：从一类显微图迁移到另一类（不依赖配对数据）

通用 SR 模型完全不适用——它们没有衍射极限的概念。

## 10.11 视频增强里的任务特化（预热）

视频增强也有任务特化的需求，第 13-14 章会展开：

- **视频会议人脸增强**：低带宽下保证说话人脸清晰
- **直播视频增强**：实时（< 30ms/帧）+ 保持品牌色调
- **监控视频增强**：身份保留 + 不允许编造（与法医证据类似）
- **老电影修复**：胶片划痕去除 + 帧率提升（用 RIFE）+ 颜色还原

这些任务都有自己的归纳偏置和约束。

## 10.12 任务特化模型的设计方法论

如果你要为一个新任务设计特化模型，按这个流程：

### 第一步：识别任务的归纳偏置

问几个问题：

- 这一类图像有什么**结构性约束**？（人脸：五官位置；文字：笔画离散；医疗：物理成像）
- 哪些信息**绝对不能改**？（人脸：身份；文字：字符；医疗：病灶）
- 哪些信息可以"创造"？（人脸：高频细节如皱纹纹理；文字：不能创造）

### 第二步：设计先验注入

把识别出的偏置编码进模型：

- 离散结构（文字、CodeFormer）→ codebook
- 强语义先验（人脸）→ pretrained generator (StyleGAN)
- 物理过程（医疗）→ unrolled network
- 多光谱物理 → 多通道架构 + 光谱损失

### 第三步：设计任务专用损失

通用增强损失（L1 + perceptual + adv）不够，要加：

- **身份保留**（人脸）
- **OCR 一致**（文字）
- **数据保真**（医疗）
- **光谱一致**（遥感）

### 第四步：设计任务专用评估

通用指标（PSNR/LPIPS）可能不反映任务质量：

- 人脸：身份相似度 + 主观评测
- 文字：OCR 准确率
- 医疗：放射科医生盲评 + 病灶检出率
- 遥感：下游任务准确率（分类、分割）

### 第五步：失败模式针对性测试

通用增强失败案例（第 17 章详谈）+ 任务特定失败：

- 人脸：极端角度、遮挡、墨镜、口罩
- 文字：手写、印章、低对比度
- 医疗：罕见病灶、伪影、运动模糊

## 10.13 小结

1. **通用模型在某些领域必然失败**——人脸、文字、医疗、遥感、显微，各有不同的归纳偏置
2. **人脸增强**是最成熟的子领域：StyleGAN 先验 + 身份保留 + 对齐
3. **GFPGAN** 用 StyleGAN2 generator + CS-SFT 调制，是 2021 经典
4. **CodeFormer** 用 VQ codebook + Transformer，工程灵活，可调 fidelity
5. **文档增强** 需要 OCR 引导损失 + 字符级正确性
6. **医疗增强** 必须用物理约束 + 展开网络，**不能用生成模型**
7. **遥感** 需要多光谱物理 + 大尺度 tile
8. **显微镜** 必须建模 PSF + 衍射极限
9. **设计方法论**：识别偏置 → 注入先验 → 专用损失 → 专用评估 → 失败测试
10. **任务特化是边际收益最大的增强方向**——通用模型只能做到 80 分，特化模型在专门场景可以做到 99 分

到这里 Part II 五章全部完成。Part III 进入训练和评估的工程细节——前面章节讲了"用什么"，这两章讲"怎么训得稳、怎么评得准"。

---

> 下一章 [训练稳定性](11-training.md) → GAN 崩、扩散调度、混合损失权重的工程经验。
