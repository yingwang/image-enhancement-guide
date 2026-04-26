# 第 17 章 · 失败案例集

> 平均指标好的模型在生产里依然会翻车。
>
> 这一章把过去几年影像增强里**最常见的失败模式**收集起来——每一种都给具体场景、原因、应对。
>
> 这是这本书里最"实战"的一章，也是最值得反复读的一章。

## 17.1 为什么需要这一章

第 12 章 12.10 节讲过"失败案例集"作为评估方法。这一章是它的内容版——**把领域里反复出现的失败模式系统化**。

经验法则：

> 一个工业级增强模型的成熟度，不看它在 Set5 上 PSNR 多高，看它的失败案例集多大。
>
> SOTA 论文优化的是平均指标。生产环境优化的是**最差的那 5%**。

下面 10 类失败模式，每一类配场景、原因、应对。

## 17.2 失败模式 1：扩散模型编造内容

### 场景

老照片修复时，扩散模型把"模糊的小婴儿脸"修成了"清晰但不像本人的脸"——爷爷拿着照片说"这不是我儿子"。

### 原因

扩散模型在严重退化的人脸上做的是**生成**，不是恢复：

- LR 信息不够，模型从训练分布里采样一个"合理人脸"
- 这个采样出来的脸**视觉上真实**，但**不是原始那个脸**
- 训练数据里 FFHQ 主要是欧美人脸，其他人种的"合理脸"分布偏

### 应对

1. **fidelity 参数给用户**：CodeFormer 的 `w` 参数允许调节
2. **检测严重退化区域，禁用生成式模型**：LR 太差就只用 bicubic 上采样
3. **后处理身份验证**：用 ArcFace 比对原图和增强图，相似度过低警告用户
4. **明确产品定位**：标明"AI 增强可能改变细节"，让用户有心理预期

```python
def safe_face_enhance(lr_face, model, identity_threshold=0.4):
    enhanced = model(lr_face)
    
    # 用 ArcFace 比对
    sim = arcface_similarity(lr_face, enhanced)
    
    if sim < identity_threshold:
        # 警告用户 (或回退到保守方法)
        return {
            'output': bicubic_upscale(lr_face, 4),
            'warning': '原图细节不足，AI 修复结果与原始可能差异较大',
        }
    return {'output': enhanced, 'warning': None}
```

## 17.3 失败模式 2：放大对抗噪声 / 伪影

### 场景

用户上传一张已经被 PS 软件 oversharpen 过的图，再用 Real-ESRGAN 处理——伪影被放大成"糖纸"纹理。

### 原因

模型学到的是"低质量到高质量"的映射。"低质量"的训练数据没有 oversharpen 过的图，所以模型把锐化伪影**当成需要恢复的细节**——锐化它。

更广义地：训练数据没 cover 的退化分布上，模型行为不可预测。

### 应对

1. **退化检测前置**：用一个分类器预测输入图的"退化类型"
2. **多分支模型**：不同退化类型用不同的增强模型
3. **训练数据补全**：加入更多"奇怪退化"（包括 oversharpen、过度饱和、过度去噪）

```python
def adaptive_enhance(image, classifier, models):
    """根据检测到的退化类型选择模型。"""
    degradation_type = classifier(image)
    
    if degradation_type == 'normal_lr':
        return models['real_esrgan'](image)
    elif degradation_type == 'oversharpened':
        return models['mild_smooth_then_sr'](image)
    elif degradation_type == 'oversaturated':
        return models['color_normalize_then_sr'](image)
    else:
        return models['safe_baseline'](image)
```

## 17.4 失败模式 3：人脸身份漂移

### 场景

视频会议增强模型，每一帧增强后看起来"美化"了，但同事发现**用户看起来像变了个人**。

### 原因

第 10.3 节讲过身份保留损失，但训练时这个损失权重不够：

- 像素损失主导 → 模型学到"标准脸"的统计
- 模型推理时把每张脸都"标准化"——独特特征（鼻型、嘴角等）被磨平

### 应对

1. **身份保留损失权重提高**：从 0.1 提到 0.5
2. **训练数据多样性**：FFHQ 之外加 IMDB-Face、Asian Face 等
3. **推理时的身份引导**：每次推理给一个"参考脸 embedding"作为额外输入

```python
class IdentityGuidedEnhancer(nn.Module):
    """每次增强用用户的参考人脸作为引导。"""

    def __init__(self, base_model, arcface):
        super().__init__()
        self.base_model = base_model
        self.arcface = arcface.eval()

    def forward(self, lr_face, reference_face):
        with torch.no_grad():
            ref_embedding = self.arcface(reference_face)
        # 把 embedding 作为条件注入 base_model
        return self.base_model(lr_face, condition=ref_embedding)
```

## 17.5 失败模式 4：视频闪烁

### 场景

用 Real-ESRGAN 逐帧处理视频，输出视频中的纹理、平坦区域、人脸细节都在"沸腾"——肉眼难受。

### 原因

第 13 章 13.1 节详谈过：单帧模型不考虑时序一致。同样的纹理在两个相邻帧中略有不同的 LR 输入 → 模型生成略有不同的细节 → 闪烁。

### 应对

1. **使用时序模型**（BasicVSR++ / VRT）替代单帧模型
2. **后处理：时序滤波**

```python
def temporal_filter_post(frames: list, alpha: float = 0.7) -> list:
    """对增强后的视频做指数移动平均, 减少闪烁。
    代价: 损失部分细节, 略带"运动模糊"感。
    """
    smoothed = [frames[0]]
    for t in range(1, len(frames)):
        # 用光流先对齐前一帧
        flow = estimate_flow(frames[t], smoothed[-1])
        warped_prev = warp_with_flow(smoothed[-1], flow)
        # 加权平均
        s = alpha * frames[t] + (1 - alpha) * warped_prev
        smoothed.append(s)
    return smoothed
```

3. **训练数据：视频对 + 时序一致性损失**——根本解决方案

## 17.6 失败模式 5：量化崩溃

### 场景

模型在 PyTorch FP32 推理 PSNR 33 dB，转 INT8 部署到端侧后 PSNR 跌到 26 dB——视觉上明显糖纸。

### 原因

低层视觉对量化敏感（第 15.2.5 节）：

- **激活分布异常**：某些层激活值范围广（max 远大于 mean），INT8 量化损失大
- **首层 / 末层敏感**：图像 → 特征 / 特征 → 图像的转换特别精细
- **PixelShuffle 后的卷积**：量化后产生明显棋盘伪影

### 应对

1. **混合精度量化**：第一层、最后一层、归一化层保留 FP16/FP32

```python
quant_config = {
    'first_conv':       'fp16',     # 输入 conv 不量化
    'pixel_shuffle_conv': 'fp16',   # 上采样前不量化
    'last_conv':        'fp16',     # 输出 conv 不量化
    'others':           'int8',
}
```

2. **QAT (Quantization-Aware Training)**：训练时加入量化扰动
3. **per-channel 量化**：不要 per-tensor，per-channel 精度高

4. **直接放弃 INT8**：低层视觉很多场景 FP16 已经够，没必要冒精度风险

## 17.7 失败模式 6：边界 / Tile 接缝

### 场景

4K 图分成 1024 tile 处理，输出图在 tile 边界有明显**接缝**——像方形拼图。

### 原因

不同 tile 独立推理：

- 边界附近的像素**只看到 tile 内的上下文**
- 邻 tile 的边界**看到不同的上下文**
- 输出在边界处不连续

### 应对

1. **Overlap + blend**（第 15.8 节）：必须做，不是可选
2. **更大 overlap**：256 像素 overlap 比 64 像素效果好（但慢）
3. **Mirror padding 边缘**：图像边缘 tile 用反射 padding 而不是常数 padding
4. **Shared noise（扩散）**：所有 tile 用同一个 noise seed，结构连续性

## 17.8 失败模式 7：极端输入崩溃

### 场景

用户上传一张**纯黑** / **纯白** / **纯随机噪声**图。增强模型输出的是各种艺术抽象。

### 原因

模型训练时没见过极端 OOD 输入：

- 纯黑：所有激活接近 0，归一化层 div by 0
- 纯白：饱和，激活异常
- 噪声：高频分量主导，模型当作"细节"放大

### 应对

1. **输入校验**：极端输入直接 bypass 模型

```python
def safe_inference(model, image):
    # 输入特征检查
    mean = image.mean()
    std = image.std()
    
    if std < 1e-3:                  # 几乎平坦
        return image                # 直接返回, 不增强
    
    if mean < 0.02 or mean > 0.98:  # 极端亮暗
        return image                # 跳过增强
    
    # 检查噪声占比
    high_freq_ratio = compute_high_freq_ratio(image)
    if high_freq_ratio > 0.7:        # 噪声主导
        # 走"先去噪再增强"分支
        return enhance_after_denoise(image)
    
    # 正常路径
    return model(image)
```

2. **训练数据补全**：合成时加入极端样本（pure black、pure white、Gaussian noise）

## 17.9 失败模式 8：色调漂移 / 偏色

### 场景

用户拍的暖色调（夕阳）照片，经过增强后**色调变冷**——失去夕阳氛围。

### 原因

模型训练数据偏向"标准白平衡"图：

- 训练 HR 经过"美化白平衡"
- 模型把所有输入往这个方向拉
- 暖色调被当作"色温偏差"修正掉

### 应对

1. **颜色一致性损失**（第 3 章 3.8 节）：训练时用
2. **后处理：颜色匹配**

```python
def color_match(enhanced: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    """把 enhanced 的色调匹配到 original。
    保留 enhanced 的细节, 借用 original 的颜色统计。
    """
    # 转 Lab 颜色空间
    enhanced_lab = rgb_to_lab(enhanced)
    original_lab = rgb_to_lab(original)
    
    # L 通道用 enhanced (细节)
    l = enhanced_lab[:, 0:1]
    # a, b 通道做大幅模糊后用 original (颜色)
    enhanced_ab_blur = F.avg_pool2d(enhanced_lab[:, 1:], 21, stride=1, padding=10)
    original_ab_blur = F.avg_pool2d(original_lab[:, 1:], 21, stride=1, padding=10)
    
    # 颜色"shift"
    color_diff = original_ab_blur - enhanced_ab_blur
    matched_ab = enhanced_lab[:, 1:] + color_diff
    
    matched_lab = torch.cat([l, matched_ab], dim=1)
    return lab_to_rgb(matched_lab)
```

3. **训练数据多样性**：覆盖各种白平衡

## 17.10 失败模式 9：文字 / OCR 损坏

### 场景

文档图增强后，原本能识别的字变成无法识别——比如"日"修复成"目"，或者笔画被磨平。

### 原因

第 10.7 节讲过：通用 SR 学到的是"自然图像先验"，与文字结构相反。

### 应对

1. **文字检测前置**：检测到文字区域用专用模型
2. **OCR 引导损失训练专用模型**
3. **保守策略**：文字区域只做最低限度增强（去模糊，不 SR）

```python
def hybrid_enhance(image, text_detector, sr_model, doc_sr_model):
    text_boxes = text_detector(image)
    
    if len(text_boxes) == 0:
        # 纯图像 -> 通用 SR
        return sr_model(image)
    
    # 有文字 -> 分区域处理
    background = sr_model(image)
    
    for box in text_boxes:
        text_crop = crop(image, box)
        enhanced_text = doc_sr_model(text_crop)
        background = paste(background, enhanced_text, box)
    
    return background
```

## 17.11 失败模式 10：训练数据偏差

### 场景

模型在白人/年轻人脸上效果好，在亚洲老年人/儿童脸上效果差——增强后人脸不像本人。

### 原因

FFHQ 数据集：

- 70K 张人脸，主要来源 Flickr
- 年龄分布偏年轻（20-40）
- 种族分布偏白人
- 性别分布相对均衡

模型学到的是这个分布的"平均脸"。在分布外的人脸（老人、儿童、特定族裔）上表现差。

### 应对

1. **数据集多样性**：补充 IMDB-Face、Asian Face、African Face 等
2. **公平性测试**：不同人群的指标分别报
3. **失败案例集**：明确标注偏差场景，定期评估

### 这不是"算法问题"，是**数据问题 + 工程纪律问题**

很多 AI 公平性问题的根源都在数据。**评估时按人群分组**，是工程上的最低纪律。

## 17.12 失败模式 11：视频场景切换

### 场景

视频在两个场景之间切换（剪辑），第二个场景的第一帧增强后**严重崩坏**——比之后的稳定状态差很多。

### 原因

循环模型（BasicVSR++）依赖隐状态：

- 隐状态从前一帧传过来
- 场景切换后，前一帧的隐状态对应**完全不同的视觉**
- 第一帧的增强用了"错的"上下文

### 应对

1. **场景切换检测 + 隐状态重置**（第 16.5 节末尾的代码）
2. **训练数据加场景切换**：合成时引入随机切换，让模型学会处理

```python
class SceneAwareVideoModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.scene_threshold = 0.3

    def forward(self, frame, hidden_state, prev_frame=None):
        if prev_frame is not None:
            diff = (frame - prev_frame).abs().mean()
            if diff > self.scene_threshold:
                hidden_state = self.base_model.init_hidden(frame.shape)
        out, new_hidden = self.base_model(frame, hidden_state)
        return out, new_hidden
```

## 17.13 失败模式 12：长序列累积误差

### 场景

视频处理几分钟后，增强质量**逐渐下降**——开头很好，结尾糊。

### 原因

循环模型的隐状态不断累积：

- 每一帧的微小误差被传到下一帧
- 长时间后误差累积到显著程度
- 甚至发散

### 应对

1. **定期重置隐状态**：每 N 帧（比如 60 帧）重置一次
2. **双向 RNN（BasicVSR++）+ 端到端训练长序列**
3. **检测异常并 fallback**：监控输出统计，异常时切换到单帧模型

## 17.14 失败模式 13：放大 watermark / logo

### 场景

用户上传带水印的图，增强后水印**变得更清晰、更显眼**。

### 原因

模型把水印当作"图像内容"——平等对待，平等增强。

### 应对

1. **水印检测前置**：先识别水印位置
2. **水印区域特殊处理**：可以选择忽略 / 增强 / 删除（但删除有版权风险）
3. **训练数据**：训练时加 watermark 增广，让模型学到"watermark 就保持 watermark 不要锐化"

## 17.15 失败模式 14：人物姿势改变

### 场景

人物动作图（健身、舞蹈），扩散增强后**手的位置变了**、**手指多了一根**。

### 原因

扩散模型在结构理解上的著名弱点：

- 训练数据里手的多样性远小于其他物体
- 手的"细节"对模型来说不是"恢复"，是"生成"
- 生成时容易违反结构（多/少手指、错位）

### 应对

1. **避免用扩散做严重退化的手部增强**
2. **如果必须用扩散，加 ControlNet 注入 pose**
3. **后处理：手部检测 + 用判别式模型替换**

## 17.16 失败模式 15：批次尺寸推理差异

### 场景

模型在 `batch_size=1` 推理时正常，在 `batch_size=8` 推理时**输出有微小差异**——长视频累积后产生明显闪烁。

### 原因

CUDA 核 / cuDNN 的非确定性：

- 不同 batch size 走不同的 kernel 路径
- 某些操作（reduction）的顺序与 batch 大小相关
- 浮点运算非结合性导致结果略不同

### 应对

1. **推理时固定 batch size**：生产推理 batch size 选定后不要随业务负载抖动；与训练 batch size 是否相同**不是关键**——关键是在生产里**保持稳定**
2. **设置 cuDNN deterministic**：

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

代价：速度略降，但结果可复现。

3. **在生产 pipeline 末端做后处理时序滤波**——掩盖微小不一致

## 17.17 一个统一的应对哲学

15 类失败模式归纳出一些通用原则：

### 原则 1：边界处理 > 平均优化

模型在 95% 场景上的指标涨 0.1 dB 不重要。在 5% 场景上从崩坏到能用更重要。

### 原则 2：检测 + 路由

不要试图让一个模型处理所有输入。前置一个分类器/检测器，把不同输入路由到不同处理路径。

### 原则 3：失败时优雅退化

模型崩坏时不能输出垃圾，要 fallback 到保守方法（bicubic、原图）。**永远有"保底"路径**。

### 原则 4：训练数据是失败的根源

绝大多数失败模式追溯回去都是**训练数据没 cover 那个分布**。补数据 > 调模型。

### 原则 5：用户 UI 设计帮助管理失败

- 给用户调节参数（fidelity）
- 失败时给警告而不是默默输出
- 设计 retry / undo 机制

## 17.18 失败案例集的工程化

把这一章的内容工程化为可执行的测试集：

```python
class FailureCaseSuite:
    """失败案例 regression 测试。"""

    def __init__(self):
        self.cases = [
            # (name, input_loader, expected_property)
            ('extreme_lr_face',  load_extreme_lr_face,  self.identity_preserved),
            ('oversharpened',    load_oversharp_image,  self.no_amplified_noise),
            ('pure_black',       load_pure_black,        self.no_artifacts),
            ('text_document',    load_text_doc,          self.ocr_consistent),
            ('non_white_face',   load_non_white_face,    self.identity_preserved),
            # ... 几十个 case
        ]

    def identity_preserved(self, input_face, output_face):
        sim = arcface_similarity(input_face, output_face)
        return sim > 0.4

    def no_amplified_noise(self, input_img, output_img):
        return high_freq_energy(output_img) < 1.5 * high_freq_energy(input_img)

    def ocr_consistent(self, input_doc, output_doc):
        return ocr(input_doc) == ocr(output_doc)

    def no_artifacts(self, input_img, output_img):
        return output_img.std() < 0.05  # 纯黑输入应该输出几乎纯黑

    def run(self, model):
        results = {}
        for name, loader, check in self.cases:
            input_img = loader()
            output = model(input_img)
            results[name] = {
                'pass': check(input_img, output),
                'output': output,
            }
        return results
```

每个新模型版本必须跑这套测试。**单纯 PSNR 涨了不算 ship-ready**——所有失败案例都通过才行。

## 17.19 小结

15 类常见失败模式：

1. **扩散编造**：fidelity 调节 + 身份验证
2. **放大对抗噪声**：检测分类 + 分支处理
3. **身份漂移**：身份损失加权 + 引导参考
4. **视频闪烁**：时序模型 + 时序后滤波
5. **量化崩溃**：混合精度 + QAT
6. **Tile 接缝**：overlap + blend + shared noise
7. **极端输入崩溃**：输入校验 + bypass
8. **色调漂移**：颜色一致性损失 + 颜色匹配
9. **文字损坏**：检测路由 + 专用模型
10. **训练数据偏差**：多样性 + 公平性测试
11. **场景切换**：切换检测 + 隐状态重置
12. **长序列累积误差**：定期重置 + 双向 RNN
13. **放大水印**：检测 + 特殊处理
14. **姿势/手部错乱**：避免扩散用于此 / ControlNet pose
15. **batch size 不一致**：固定 batch + cuDNN 确定性

通用应对哲学：

- 边界处理 > 平均优化
- 检测 + 路由分流
- 失败时优雅退化
- 训练数据是根源
- UI 帮助管理失败

这一章值得反复回看——你做的每个增强模型都会在这些失败模式里至少踩中一半。**提前知道、提前应对、提前测试**比上线后救火便宜得多。

最后一章，我们看 2026 年最值得关注的几个 SOTA 模型，作为这本书的"快速参考"。

---

> 下一章 [SOTA 模型](18-sota.md) → 2026 年值得用的 7 个学术 SOTA + 选型决策树。
