# 第 15 章 · 推理优化

> 训练完一个好模型只是开始。
>
> 把它推到生产环境——服务器、桌面 GPU、移动端、嵌入式——是另一个完整的工程领域。
>
> 这一章讲量化、TensorRT、CoreML、torch.compile、tile 推理、流式处理。

## 15.1 推理优化的部署目标

不同平台对模型的要求差异巨大：

| 平台 | 延迟要求 | 显存预算 | 模型大小预算 | 优先级 |
|------|---------|---------|-----------|-------|
| **A100 服务器** | < 1 秒/张 | 80 GB | 几 GB | 吞吐量 |
| **消费 GPU**（4090） | < 100 ms/张 | 24 GB | 几 GB | 用户体验 |
| **桌面 CPU** | < 5 秒/张 | 16 GB | < 500 MB | 兼容性 |
| **手机 NPU** | < 30 ms/帧 | < 1 GB | < 50 MB | 实时性 + 功耗 |
| **嵌入式（IoT）** | < 100 ms/张 | < 100 MB | < 10 MB | 严格预算 |

每个平台有自己的优化栈。这一章按平台分别讲。

## 15.2 通用优化（所有平台都要）

### 15.2.1 模型导出格式

PyTorch 训练模型要导出到推理格式：

```
PyTorch model (.pt)
  ├── ONNX (.onnx)              ← 跨平台中间格式
  │   ├── TensorRT (.plan)      ← NVIDIA GPU
  │   ├── OpenVINO              ← Intel CPU/GPU
  │   ├── DirectML              ← Windows 通用
  │   └── ONNX Runtime          ← 通用 CPU/GPU
  ├── TorchScript (.pt)         ← PyTorch 原生
  ├── CoreML (.mlpackage)       ← Apple 设备
  ├── TFLite (.tflite)          ← Android / 嵌入式
  └── 自定义 (Custom)            ← 厂商 SDK
```

**ONNX 是事实标准的中间格式**——大多数推理引擎都接受 ONNX。

```python
# 导出 PyTorch 模型到 ONNX
import torch

model.eval()
dummy_input = torch.randn(1, 3, 256, 256)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input':  {0: 'batch', 2: 'height', 3: 'width'},  # 动态尺寸
        'output': {0: 'batch', 2: 'height', 3: 'width'},
    },
    opset_version=17,
)
```

### 15.2.2 算子融合

把多个连续算子合并成一个，减少 kernel 启动开销和内存访问：

- `Conv + BatchNorm` → `Conv'`（合并 BN 到 conv 权重）
- `Conv + ReLU` → `Conv-ReLU` fused op
- `LayerNorm + Linear` → fused op

PyTorch 2.0+ 的 `torch.compile` 自动做这些。

### 15.2.3 torch.compile

PyTorch 2.0 引入的 JIT 编译，能把模型自动优化：

```python
import torch

model = build_model().eval().cuda()
model = torch.compile(model, mode="reduce-overhead")  # 或 "max-autotune"

# 第一次推理会慢 (编译)
with torch.no_grad():
    _ = model(warmup_input)  # 触发编译

# 之后快
output = model(real_input)
```

`mode` 选择：

- `"default"`：稳妥，提速 1.5-2×
- `"reduce-overhead"`：减少 Python 开销，适合小 batch
- `"max-autotune"`：最大优化，编译慢但运行最快

实测影响：

- Real-ESRGAN 在 4090 上原生 PyTorch 跑 200ms/256×256，`torch.compile` 后 90ms

工程注意：

- 输入形状变化时会**重新编译**，造成第一次运行卡顿
- 用 `dynamic=True` 让编译器准备好动态形状
- 兼容性问题：某些自定义算子不支持

### 15.2.4 FP16 / BF16 推理

训练用 FP32 / BF16，推理可以用 FP16 / BF16 减一半显存 + 加速 1.5-2×：

```python
model = build_model().eval().cuda().half()  # FP16
output = model(input.half())
```

注意事项：

- VAE encoder/decoder 在 FP16 下可能溢出，**保留 FP32**
- 某些算子（softmax、归一化）在 FP16 下精度损失大，自动 cast 到 FP32
- BF16 数值范围大但精度低，**A100+ 推荐 BF16**

### 15.2.5 量化（INT8 / INT4）

把权重和激活从 FP16 降到 INT8 甚至 INT4：

- **PTQ（Post-Training Quantization）**：训练后量化，简单但精度损失
- **QAT（Quantization-Aware Training）**：训练时考虑量化，精度损失小但训练复杂

INT8 量化的潜在收益：

- 模型大小：4×（FP32→INT8）或 2×（FP16→INT8）
- 速度：在支持 INT8 的硬件上 2-4×
- 显存：相应减少

低层视觉的特殊困难：**输出像素精度对量化误差敏感**。INT8 量化可能让 PSNR 跌 0.5-2 dB，视觉上能看出"格点状"伪影。

实践经验：

- **大模型**（Restormer/HAT/扩散 UNet）：INT8 后效果还行
- **小模型**（NAFNet 小版本、ESRGAN-Lite）：INT8 后明显下降，建议 FP16 即可
- **VAE**：永远不要 INT8，会崩

```python
# 简化的 PyTorch INT8 PTQ
import torch.ao.quantization as quant

model = build_model().eval()

# 1. 准备校准数据
calibration_data = [load_calibration_batch(i) for i in range(100)]

# 2. 配置量化方案
qconfig = quant.get_default_qconfig('fbgemm')  # CPU
# 或 qconfig = quant.get_default_qconfig('x86') for newer x86

model.qconfig = qconfig
model_prepared = quant.prepare(model)

# 3. 跑校准 (用代表性数据)
with torch.no_grad():
    for batch in calibration_data:
        model_prepared(batch)

# 4. 转换成 INT8 模型
model_int8 = quant.convert(model_prepared)
```

GPU 上 INT8 量化更复杂，通常通过 TensorRT 或 ONNX Runtime 做。

## 15.3 NVIDIA GPU 部署：TensorRT

### TensorRT 是 NVIDIA 的推理引擎

它做的事：

- 接受 ONNX 或 TF/PyTorch 模型
- 算子融合 + 内核选择 + 量化
- 输出**针对特定 GPU 优化**的二进制（`.plan` 文件）
- 提供 C++/Python API 推理

### 工作流

```python
import tensorrt as trt

# 1. 从 ONNX 构建 TensorRT 引擎
def build_engine(onnx_path: str, engine_path: str,
                 fp16: bool = True, int8: bool = False):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GB

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = MyCalibrator(...)  # 需要校准数据

    # 动态形状支持
    profile = builder.create_optimization_profile()
    profile.set_shape("input",
                     min=(1, 3, 64, 64),
                     opt=(1, 3, 256, 256),
                     max=(1, 3, 1024, 1024))
    config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    with open(engine_path, 'wb') as f:
        f.write(engine)
```

### TensorRT 推理

```python
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np

class TRTInferencer:
    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def infer(self, input_array: np.ndarray) -> np.ndarray:
        # 1. 设置动态输入 shape, 然后查询输出实际 shape
        # 注意: SR 模型的输出 shape != 输入 shape (放大了 scale 倍)
        self.context.set_input_shape("input", input_array.shape)
        out_shape = tuple(self.context.get_tensor_shape("output"))

        # 2. 按真实 shape 分配显存
        in_size  = int(np.prod(input_array.shape) * np.dtype(np.float32).itemsize)
        out_size = int(np.prod(out_shape) * np.dtype(np.float32).itemsize)
        d_input  = cuda.mem_alloc(in_size)
        d_output = cuda.mem_alloc(out_size)
        cuda.memcpy_htod(d_input, input_array.astype(np.float32))

        # 3. TensorRT 10+ 推荐用 name-based API
        self.context.set_tensor_address("input",  int(d_input))
        self.context.set_tensor_address("output", int(d_output))
        stream = cuda.Stream()
        self.context.execute_async_v3(stream.handle)
        stream.synchronize()

        # 4. 拷贝回 CPU
        output = np.empty(out_shape, dtype=np.float32)
        cuda.memcpy_dtoh(output, d_output)
        return output
```

### TensorRT 加速效果

| 模型 | 原 PyTorch | TensorRT FP16 | TensorRT INT8 |
|------|-----------|--------------|--------------|
| Real-ESRGAN (RRDB) | 100% | 35% | 18% |
| SwinIR | 100% | 40% | — |
| BasicVSR++ | 100% | 50% | — |
| SDXL UNet | 100% | 45% | — |

**TensorRT FP16 通常能加速 2-3×**，INT8 再加速 2× 但精度风险大。

## 15.4 Apple 设备部署：CoreML

### CoreML 是什么

Apple 的推理框架，能跑在：

- **CPU**（兼容所有设备）
- **GPU**（M 系列芯片、iPhone GPU）
- **Apple Neural Engine (ANE)**（M 系列 Mac + A 系列 iPhone/iPad，超低功耗）

### 工作流

```python
import coremltools as ct
import torch

# 1. 把 PyTorch 模型 trace 出来
model.eval()
dummy_input = torch.randn(1, 3, 256, 256)
traced = torch.jit.trace(model, dummy_input)

# 2. 转 CoreML
mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(name="input",
                         shape=(1, 3, 256, 256),
                         scale=1/255.0,
                         color_layout=ct.colorlayout.RGB)],
    outputs=[ct.TensorType(name="output")],
    compute_precision=ct.precision.FLOAT16,    # iPhone 默认 FP16
    compute_units=ct.ComputeUnit.ALL,          # CPU + GPU + ANE
    minimum_deployment_target=ct.target.iOS17,
)
mlmodel.save("model.mlpackage")
```

### Apple Neural Engine 的特殊性

- 极低功耗（10× CPU）
- 但**只支持有限算子**（Conv、ReLU、PixelShuffle 等基础算子）
- 自定义层、不规则形状会 fallback 到 CPU/GPU

工程后果：**为 ANE 设计模型必须用受限算子集**：

- ✓ Conv2d (3×3, 5×5)
- ✓ ReLU / LeakyReLU / GELU
- ✓ BatchNorm
- ✓ PixelShuffle
- ✗ Custom CUDA kernels
- ✗ Dynamic shapes (受限)
- ✗ 复杂 attention（部分支持）

实测：

- Real-ESRGAN 在 iPhone 14 ANE 上 720P 输入 ~80ms
- CoreFormer 同设备 ~150ms

### CoreML 的 4-bit 量化

iOS 17+ 支持 4-bit 权重量化，模型大小再减一半：

```python
import coremltools.optimize.coreml as cto

config = cto.OptimizationConfig(
    global_config=cto.OpPalettizerConfig(
        nbits=4,
        granularity="per_grouped_channel",
        group_size=16,
    ),
)
mlmodel_quantized = cto.palettize_weights(mlmodel, config)
```

实测：模型大小 50MB → 12MB，质量损失 <0.3 dB。

## 15.5 Android / 嵌入式：TFLite + NNAPI

Android 设备的事实标准：

```python
# 通过 ONNX → TFLite 转换 (用 onnx-tf)
import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("model.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("model.pb")

# 然后用 tf converter 转 TFLite
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("model.pb")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]   # 或 INT8
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)
```

NNAPI（Android 8+）能把 TFLite 模型路由到设备 NPU，但兼容性差异大（不同芯片厂商支持的算子集不同）。

实践：**Android 端常用厂商专用 SDK**——高通的 SNPE、联发科的 NeuroPilot、华为的 HiAI。

## 15.6 模型蒸馏：极致轻量

预训练模型太大？训一个**学生模型**（小模型）模仿教师模型（大模型）的输出。

### 蒸馏的训练目标

```python
def distillation_loss(student_out, teacher_out, hr_target):
    """蒸馏损失 = 模仿教师 + 还学真值。"""
    # 模仿教师输出
    distill = F.l1_loss(student_out, teacher_out.detach())
    # 也看真值, 防止学生只学教师的错误
    gt = F.l1_loss(student_out, hr_target)
    return 0.7 * distill + 0.3 * gt
```

### 适合低层视觉的蒸馏

- **特征蒸馏**：让学生中间层特征模仿教师中间层
- **关系蒸馏**：让学生输出的内部关系（GRAM 矩阵等）匹配教师

### 蒸馏的工程效果

- 学生模型 1/4 大小，效果接近教师 95%
- 推理速度 3-5×
- 是端侧部署的关键技术

代表项目：

- **Real-ESRGAN-Mini**：1.5M 参数蒸馏版，性能 ESRGAN 70%
- **SwinIR-Lite**：用 Swin Transformer 蒸馏到纯 CNN

## 15.7 LCM / Turbo 蒸馏（扩散专用）

扩散模型 50 步推理太慢。**Latent Consistency Models (LCM)** 把它蒸馏到 4-8 步：

### 核心思想

训练学生模型（CM）让它从任意 $t$ 一步预测 $x_0$。这样推理可以**一步直接采样**。

### LCM-LoRA

更轻量：训一个 LoRA 而不是完整模型，可以加到任何 SD 模型上：

```python
# 加载 LCM-LoRA 到 SD pipeline
from diffusers import LCMScheduler, AutoPipelineForImage2Image

pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")

# 推理 4 步
output = pipe(prompt, image=lr_image, num_inference_steps=4, guidance_scale=1.5)
```

实测：

- 原 SDXL 50 步 25 秒
- LCM-LoRA 4 步 2 秒
- 质量损失：FID 略升、LPIPS 略升，肉眼基本无区别（4× 增强场景）

## 15.8 Tile 推理：处理大图

第 9 章 9.9 节讲过扩散的 tile，这里扩展到所有模型。

### 何时需要 tile

- 输入分辨率 > 训练分辨率（特别是 4K/8K）
- 显存不够（24GB 卡跑 1080P 4× SR 也可能 OOM）

### 工程实现

```python
import torch
import torch.nn.functional as F


def tile_inference(model, image, tile_size=512, overlap=64, scale=4):
    """
    通用的 tile 推理。
    image: (1, 3, H, W) 输入
    返回: (1, 3, H*scale, W*scale) 输出
    """
    _, _, H, W = image.shape
    out_H, out_W = H * scale, W * scale

    stride = tile_size - overlap
    # 输出累加器
    output = torch.zeros((1, 3, out_H, out_W), device=image.device)
    weight = torch.zeros((1, 1, out_H, out_W), device=image.device)

    # 渐变 mask
    mask = torch.ones((1, 1, tile_size * scale, tile_size * scale),
                      device=image.device)
    edge = overlap * scale
    for i in range(edge):
        v = (i + 1) / (edge + 1)
        mask[:, :, i, :]    *= v
        mask[:, :, -i-1, :] *= v
        mask[:, :, :, i]    *= v
        mask[:, :, :, -i-1] *= v

    for top in range(0, max(1, H - tile_size + 1), stride):
        for left in range(0, max(1, W - tile_size + 1), stride):
            # 取 tile
            tile_h = min(tile_size, H - top)
            tile_w = min(tile_size, W - left)
            tile = image[:, :, top:top + tile_h, left:left + tile_w]

            # pad 到 tile_size (如果是边角)
            pad_h = tile_size - tile_h
            pad_w = tile_size - tile_w
            if pad_h > 0 or pad_w > 0:
                tile = F.pad(tile, (0, pad_w, 0, pad_h), mode='reflect')

            # 推理
            with torch.no_grad():
                out_tile = model(tile)

            # 裁回原 tile 大小 × scale
            out_tile = out_tile[:, :, :tile_h * scale, :tile_w * scale]
            local_mask = mask[:, :, :tile_h * scale, :tile_w * scale]

            # 累加
            out_top  = top * scale
            out_left = left * scale
            output[:, :, out_top:out_top + tile_h * scale,
                         out_left:out_left + tile_w * scale] += out_tile * local_mask
            weight[:, :, out_top:out_top + tile_h * scale,
                         out_left:out_left + tile_w * scale] += local_mask

    return output / (weight + 1e-8)
```

### Tile 的 trade-off

| 维度 | tile 大 | tile 小 |
|------|--------|--------|
| 显存 | 高 | 低 |
| 速度 | 快（fewer tiles） | 慢 |
| 边界伪影 | 少 | 多 |
| Overlap 比例 | 低足够 | 必须高 |

工程经验：

- 显存够用：tile_size = 1024，overlap = 128
- 显存吃紧：tile_size = 512，overlap = 64
- 极端紧张：tile_size = 256，overlap = 32

## 15.9 流式视频处理

视频处理时不能等整段视频加载完，要**流式**——逐帧处理逐帧输出。

```python
class StreamingVideoEnhancer:
    """流式视频增强 (适合实时直播 / 长视频)。"""

    def __init__(self, model, num_history: int = 5):
        self.model = model.eval()
        self.history = []                 # 保留最近 N 帧
        self.num_history = num_history

    def process_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """处理一帧, 利用历史帧。"""
        # 加入历史
        self.history.append(frame)
        if len(self.history) > self.num_history:
            self.history.pop(0)

        # 用历史 + 当前帧推理 (假设 model 接受序列)
        if len(self.history) >= 2:
            stacked = torch.stack(self.history, dim=1)  # (B, T, C, H, W)
            with torch.no_grad():
                out = self.model(stacked)
            return out[:, -1]   # 只输出最新一帧
        else:
            return frame  # 首帧直接返回
```

### 流式处理的难点

- **延迟 vs 一致性**：滑动窗口需要"未来帧"，但实时场景拿不到
- **隐状态管理**：循环模型的隐状态在长视频里要重置
- **边界处理**：场景切换时隐状态要清空

工程实践：实时视频增强的 SOTA 大多用因果（causal）模型——只看历史不看未来。

## 15.10 多模型 pipeline 优化

实际产品常常是多个模型组合（去噪 → SR → 上色 → 帧插值）。优化点：

### 流水线（Pipeline）

```
Stage 1 (去噪)  ┐
Stage 2 (SR)    ├── 三 GPU 并行
Stage 3 (颜色)  ┘
```

如果三个模型在不同 GPU：

- GPU 1 处理 frame N 的去噪
- GPU 2 处理 frame N-1 的 SR
- GPU 3 处理 frame N-2 的颜色

吞吐量 3×。

### 中间表示传递

不要每阶段都 decode 到 RGB → encode 到 latent。如果两个模型都用同一 VAE，**保留 latent 跨模型传递**：

```python
# Bad: 每阶段都过 VAE
img1 = vae.decode(latent_denoise)
latent_sr = vae.encode(img1)
img2 = vae.decode(model_sr(latent_sr))

# Good: latent 直接传
img2 = vae.decode(model_sr(latent_denoise))
```

## 15.11 推理监控

生产环境需要监控：

- **延迟分布**（p50/p90/p99）
- **吞吐量**（QPS）
- **GPU 利用率**
- **OOM 发生率**
- **失败率**（NaN 输出、超时）
- **质量指标**（在线 NIQE 等无参考指标）

```python
import time
from collections import deque

class InferenceMonitor:
    def __init__(self, window=1000):
        self.latencies = deque(maxlen=window)
        self.failures = 0
        self.total = 0

    def record(self, fn):
        def wrapped(*args, **kwargs):
            self.total += 1
            t0 = time.time()
            try:
                result = fn(*args, **kwargs)
                self.latencies.append(time.time() - t0)
                return result
            except Exception as e:
                self.failures += 1
                raise
        return wrapped

    def stats(self):
        if not self.latencies:
            return {}
        sorted_lats = sorted(self.latencies)
        n = len(sorted_lats)
        return {
            'p50':  sorted_lats[n // 2],
            'p90':  sorted_lats[int(n * 0.9)],
            'p99':  sorted_lats[int(n * 0.99)],
            'failure_rate': self.failures / max(1, self.total),
        }
```

## 15.12 Cost 估算

工程决策需要 cost 数据。一些参考值（2026 年 4 月）：

| 平台 | 单位成本 |
|------|---------|
| AWS A100 (8×) | ~$32/小时 |
| RunPod A100 | ~$1.5/小时 |
| GCP TPU v5p | ~$5/小时 |
| Apple Neural Engine | 设备本身一次性 |
| 手机 NPU | 设备本身一次性 |

按任务计算的 cost：

```
输入: 1080P 视频 → 4K 增强
模型: BasicVSR++ TensorRT FP16
设备: A100
延迟: 80ms/帧 (4K 输出)
30 fps 视频: 30 × 80ms = 2.4 秒/秒视频
吞吐量: 0.42× 实时

成本: A100 ($1.5/h) × (1/0.42) = $3.6/小时视频
```

## 15.13 部署清单

把模型推到生产前的 checklist：

- [ ] 导出 ONNX，验证算子兼容性
- [ ] 构建 TensorRT/CoreML/TFLite 引擎
- [ ] 跑 benchmark（延迟、吞吐量、显存）
- [ ] 跑端到端质量测试（PSNR/LPIPS 对比 PyTorch baseline）
- [ ] 极端输入测试（小图、大图、纯白、纯黑、噪声）
- [ ] 长跑测试（连续 10000 张图，看是否有内存泄漏）
- [ ] 多线程并发测试
- [ ] 设备温度/功耗测试（移动端）
- [ ] 不同输入分辨率的兼容性
- [ ] OOM 处理（自动降级到 tile）
- [ ] 监控埋点
- [ ] 回滚方案

## 15.14 小结

1. **不同平台不同优化栈**：A100 用 TensorRT，Apple 用 CoreML，Android 用 TFLite + 厂商 SDK
2. **ONNX 是事实标准中间格式**——大多数推理引擎都接受
3. **torch.compile 是 PyTorch 2.0 起的免费午餐**——加几行代码加速 1.5-2×
4. **FP16/BF16 推理**几乎无损，应当默认开启
5. **INT8 量化**对低层视觉精度敏感，谨慎使用
6. **TensorRT FP16 + INT8 加速 4-6×**，NVIDIA 部署的事实标准
7. **CoreML 上 ANE 极省功耗**但算子受限
8. **蒸馏（包括 LCM/Turbo）是端侧部署的关键技术**
9. **Tile 推理处理大图**：tile_size + overlap + blend mask
10. **流式视频处理**用因果模型，避免延迟
11. **生产监控**：延迟分布、失败率、质量指标

下一章看真实场景案例——把前面所有章节的内容应用到具体的产品场景。

---

> 下一章 [真实场景案例](16-cases.md) → 老照片修复、暗光增强、UGC、4K 直播、ISP 端侧增强各自的 pipeline。
