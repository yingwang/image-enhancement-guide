# 第 11 章 · 训练稳定性

> 前 10 章讲了"用什么模型、什么损失、什么数据"。
>
> 这一章讲"把它们组装起来训练，怎么训得稳"。
>
> 增强模型的训练比分类/LLM 更脆弱——多损失冲突、GAN 动态、扩散调度，每一个都能让你训三天结果发现是个崩的模型。

## 11.1 为什么训练稳定性是大问题

LLM 训练的损失函数是单一 cross-entropy，训练动力学相对简单。增强模型完全不同：

- **多损失混合**：L1 + VGG + GAN + 任务特化，权重失衡就崩
- **GAN 训练**：D 和 G 的动态平衡，一方过强就崩
- **扩散训练**：时间步采样、loss weighting、EMA 缺一不可
- **数据 pipeline 复杂**：退化合成在 GPU 上做，bug 容易隐藏

这一章把工程上的踩坑归纳成可执行的 checklist。

## 11.2 训练 anatomy：基础组件

一个增强模型的训练循环包含：

```python
# 抽象骨架
optimizer = build_optimizer(model)
scheduler = build_scheduler(optimizer)
scaler = torch.cuda.amp.GradScaler()                # AMP
ema = EMA(model, decay=0.999)                       # 指数移动平均

for step, batch in enumerate(train_loader):
    # 1. 预热 (warmup)
    if step < warmup_steps:
        adjust_lr_for_warmup(optimizer, step, warmup_steps)

    # 2. 前向 (with AMP)
    with torch.cuda.amp.autocast():
        loss = compute_loss(model, batch)

    # 3. 反传
    scaler.scale(loss).backward()

    # 4. 梯度裁剪
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 5. 优化器步进
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    # 6. 学习率调度
    scheduler.step()

    # 7. EMA 更新
    ema.update(model)

    # 8. 监控
    if step % log_interval == 0:
        log_metrics(...)
    if step % eval_interval == 0:
        evaluate(...)
    if step % save_interval == 0:
        save_checkpoint(...)
```

下面逐项展开。

## 11.3 优化器选择

| 优化器 | 何时用 |
|-------|------|
| **Adam** | 经典选择，稳 |
| **AdamW** | 加了 decoupled weight decay，**默认推荐** |
| **Lion** | 2023 新优化器，省显存（只需要 momentum），效果接近 AdamW |
| **Adafactor** | 大模型节省显存（不存二阶矩） |
| **SGD + momentum** | 不推荐用于增强（收敛慢，超参敏感） |

**起点配置**（增强任务通用）：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-4,                      # SR/去噪经验值
    betas=(0.9, 0.99),            # 0.99 而不是 0.999, 更稳
    weight_decay=0.01,
    eps=1e-8,
)
```

学习率经验：

- **CNN（EDSR/NAFNet）**：$2 \times 10^{-4}$
- **Transformer（SwinIR/Restormer）**：$2 \times 10^{-4}$，warmup 必须
- **GAN finetune**：$10^{-4}$ 给 G，$10^{-4} \times 4 = 4 \times 10^{-4}$ 给 D（TTUR）
- **扩散从头训**：$10^{-4}$
- **扩散 finetune**：$10^{-5}$ 到 $5 \times 10^{-6}$
- **ControlNet 训练**：$10^{-5}$

## 11.4 学习率调度

### Linear Warmup

训练前期 LR 从 0 线性涨到目标值。**必须**在 Transformer 和扩散训练里用——否则前几步可能直接发散。

```python
def linear_warmup(step: int, warmup_steps: int, target_lr: float) -> float:
    if step >= warmup_steps:
        return target_lr
    return target_lr * (step + 1) / warmup_steps
```

warmup_steps 经验值：

- 总训练步数的 **1-5%**
- 大模型（扩散）：5000-10000 步
- 小模型（CNN SR）：500-1000 步

### Cosine Decay

warmup 后用 cosine 把 LR 降到最终值（通常 LR 的 1-10%）：

$$
\eta_t = \eta_{\min} + \frac{1}{2} (\eta_{\max} - \eta_{\min}) \left( 1 + \cos\left( \frac{t}{T} \pi \right) \right)
$$

```python
import torch.optim.lr_scheduler as sched

# Warmup + cosine
scheduler = sched.SequentialLR(
    optimizer,
    schedulers=[
        sched.LinearLR(optimizer, start_factor=0.001, end_factor=1.0,
                       total_iters=warmup_steps),
        sched.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps,
                                eta_min=target_lr * 0.01),
    ],
    milestones=[warmup_steps],
)
```

### Multi-step decay（经典 SR 用）

ESRGAN/Real-ESRGAN 用阶梯式衰减：

```python
scheduler = sched.MultiStepLR(
    optimizer,
    milestones=[200_000, 400_000, 600_000, 800_000],   # 训 1M 步
    gamma=0.5,                                          # 每次 LR×0.5
)
```

### Cosine Restart

训练后期周期性"重启"LR 到高值，跳出局部最优：

```python
scheduler = sched.CosineAnnealingWarmRestarts(
    optimizer, T_0=100_000, T_mult=1, eta_min=1e-6
)
```

工程经验：训练扩散模型尤其推荐 cosine restart，能在长训练中持续涨点。

## 11.5 Batch Size 与 Patch Size

低层视觉训练的特殊性：**几乎不用整张图训**，用 patch。

### Patch-based 训练的标准

- 从 HR 随机裁剪 patch（典型 $256 \times 256$ 或 $128 \times 128$）
- 经过退化合成得到对应 LR patch
- 训练 batch 是 patch batch

为什么不直接训整张图：

- 内存：整张 $2048 \times 2048$ 一个 batch 都装不下
- 数据增强：裁剪本身是隐式的数据增强
- 训练效率：相同 GPU 时间，patch 训练能见到更多多样性

### Patch size 选择

| 任务 | 推荐 HR patch size | 原因 |
|------|-----------------|------|
| 4× SR | 256 | LR=64, 算力够 |
| 8× SR | 384 | LR=48, 需要更多空间上下文 |
| 去噪 | 128-192 | 局部纹理够用 |
| 去模糊 | 256-384 | 模糊核大需要大 patch |
| 扩散 | 512 | 与预训练 SD 一致 |

### Effective batch size

```
effective_batch_size = num_patches_per_image × image_batch × gradient_accumulation × num_gpus
```

常见配置：

- 单卡 A100，CNN SR：8 张图 × 1 patch × 1 = 8
- 4× A100，扩散：16 张图 × 1 patch × 2 GA × 4 GPU = 128

### 大 batch 的 LR 缩放

经验法则：batch size × 2，LR × √2。但低层视觉对 LR 敏感，建议**实测**而不是盲目缩放。

## 11.6 Gradient Clipping

防止梯度爆炸的标配。

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,    # 增强任务经验值
)
```

`max_norm` 经验值：

- CNN：1.0 - 5.0
- Transformer：1.0
- GAN：0.5（更激进）
- 扩散：1.0

## 11.7 EMA：扩散和高质量 GAN 必备

EMA（Exponential Moving Average）维护一份模型权重的指数移动平均：

$$
\theta_{\text{EMA}}^{(t)} = \beta \cdot \theta_{\text{EMA}}^{(t-1)} + (1-\beta) \cdot \theta^{(t)}
$$

```python
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.clone().detach() for n, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def apply_to(self, model):
        for n, p in model.named_parameters():
            p.data.copy_(self.shadow[n])

    def state_dict(self):
        return {'decay': self.decay, 'shadow': self.shadow}

    def load_state_dict(self, state):
        self.decay = state['decay']
        self.shadow = state['shadow']
```

EMA decay 经验值：

- CNN SR：0.999（不是必须，但 PSNR 涨 0.05-0.1 dB）
- GAN：0.999（推理用 EMA 输出更稳）
- 扩散：**0.9999** 或 **0.99995**（必须，论文标准）

EMA 在扩散训练里尤其重要——纯训练权重的 FID 通常比 EMA 权重的 FID 差几个点。

## 11.8 Mixed Precision (AMP)

用 FP16/BF16 计算，FP32 累积。能节省显存 + 加速 ~1.5-2×。

```python
scaler = torch.cuda.amp.GradScaler()

for batch in loader:
    optimizer.zero_grad()

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):  # 或 torch.float16
        loss = compute_loss(model, batch)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

### FP16 vs BF16

| 维度 | FP16 | BF16 |
|------|------|------|
| 数值范围 | 小（容易溢出） | 大（接近 FP32） |
| 精度 | 高 | 低 |
| 硬件 | V100, A100, H100 | A100, H100 |
| 推荐 | 老硬件 | **A100 及以后默认** |

低层视觉里 BF16 几乎总是更好的选择——数值稳定性强，不需要 GradScaler。

### 哪些层不能用低精度

- **VAE encoder/decoder**：扩散里 VAE 通常用 FP32 或 FP16，不要用 BF16（数值精度不够）
- **softmax**：在 Transformer 里 softmax 用 FP32 更稳，diffusers/transformers 自动处理
- **Loss 计算**：可以 FP32

## 11.9 GAN 训练崩塌：诊断与处理

GAN 训练在低层视觉里是最容易崩的部分。常见症状：

### 症状 1：模式崩溃

输出图全部相似（不论输入什么 LR）。诊断：

- 检查 G 的输出多样性（用一组固定 LR 看输出）
- 看 D loss：如果 D loss 极低（< 0.01），D 太强，G 没法学

### 症状 2：D 过强

```
D_loss → 0
G_loss 不下降 / 抖动
```

处理：

- 降低 D 学习率（保持 G 不变）
- 加 spectral normalization 到 D
- 加 R1 regularization
- D 训练频率降低（每 2 步训一次 D）

### 症状 3：D 过弱

```
D_loss → 1 / 不收敛
G_loss 也不收敛
```

处理：

- 提高 D 学习率
- 加大 D 容量（更深/更宽）
- 减少 G 容量

### 症状 4：训练发散

```
loss 突然变 NaN
```

处理：

- 检查梯度 norm 是否爆炸（log 出来）
- 加 gradient clipping
- 降低 LR
- 检查数据是否有 inf/NaN

## 11.10 GAN 稳定化技巧

### Spectral Normalization

控制 D 的 Lipschitz 常数，让 D 不会任意陡峭。

```python
import torch.nn.utils.spectral_norm as spectral_norm

class StableDiscriminator(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.layers = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, 64, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            # ...
            spectral_norm(nn.Conv2d(512, 1, 4, stride=1, padding=1)),
        )
```

工程经验：**几乎所有现代 GAN 都用 SpectralNorm**，加上去就稳很多。

### R1 Regularization

对真实样本上 D 的梯度做惩罚：

$$
\mathcal{L}_{R_1} = \frac{\gamma}{2} \mathbb{E}_{x \sim p_{\text{data}}} \left[ ||\nabla_x D(x)||^2 \right]
$$

```python
def r1_penalty(d_real: torch.Tensor, real_imgs: torch.Tensor) -> torch.Tensor:
    grad = torch.autograd.grad(
        outputs=d_real.sum(),
        inputs=real_imgs,
        create_graph=True,
        only_inputs=True,
    )[0]
    return grad.flatten(1).pow(2).sum(1).mean() * 0.5
```

加到 D loss 上：

```python
d_loss = d_loss_main + 10.0 * r1_penalty(d_real, real_imgs)
```

R1 让 D 在真实数据附近不要太陡，减少 D 过强问题。

### TTUR

D 用比 G 更大的学习率（典型 4×）：

```python
opt_g = AdamW(g.parameters(), lr=1e-4)
opt_d = AdamW(d.parameters(), lr=4e-4)   # 4× lr
```

## 11.11 两阶段训练（强烈推荐）

低层视觉的 GAN 训练经验：**永远先单独训 G 到收敛，再加 GAN 损失**。

### 阶段 1：Pretrain G

```python
# 只用像素损失 + 感知损失, 不用 GAN
loss = l1_loss + 1.0 * vgg_loss
# 训到 PSNR 收敛
```

通常需要 200K-500K 步。

### 阶段 2：GAN finetune

```python
# 加上 GAN 损失, 权重小 (0.005 - 0.1)
g_loss = l1_loss + 1.0 * vgg_loss + 0.05 * adv_loss
# 学习率降到 pretrain 的 1/2
```

通常 100K-200K 步。

### 为什么要两阶段

直接联合训练（all-in-one）的问题：

- GAN loss 早期梯度大、噪声大，会破坏像素一致性
- 模型还没学到基本恢复能力，GAN 强行推它"生成细节"会输出鬼东西
- 损失加权不容易调到合适

ESRGAN、Real-ESRGAN、BSRGAN 等都用两阶段策略。**这是事实标准**，不要尝试创新省一阶段。

## 11.12 扩散训练的特殊性

### 时间步采样

均匀采样（uniform）是 baseline，但有些 $t$ 区间贡献更大：

```python
def importance_sampling_t(B, num_steps=1000):
    """非均匀时间步采样, 偏向中间区域。"""
    # 经验: t ∈ [200, 800] 对最终质量贡献最大
    weights = torch.ones(num_steps)
    weights[:200]  *= 0.5
    weights[800:]  *= 0.5
    weights = weights / weights.sum()
    return torch.multinomial(weights, B, replacement=True)
```

### Min-SNR 加权（第 3 章 3.6 节复习）

```python
def min_snr_weight(t, alphas_cumprod, gamma=5.0):
    """Min-SNR 加权, SDXL 标配。"""
    snr = alphas_cumprod[t] / (1 - alphas_cumprod[t])
    return torch.minimum(snr, torch.full_like(snr, gamma)) / snr
```

加到 loss 上：

```python
loss = (min_snr_weight(t, alphas_cumprod).view(-1, 1, 1, 1) *
        (pred_noise - true_noise) ** 2).mean()
```

### v-prediction vs eps-prediction

第 3 章 3.6 节讲过。工程实践：

- **新训练**：v-prediction 更稳定
- **基于 SD 1.5 finetune**：保留原 eps-prediction
- **基于 SD 2.x / SDXL refiner**：v-prediction

### 大模型多卡（FSDP）

SDXL UNet 2.6B 参数，单卡装不下。用 FSDP（Fully Sharded Data Parallel）：

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

model = FSDP(
    unet,
    auto_wrap_policy=partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={UNetBlock},
    ),
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
)
```

## 11.13 混合损失权重的工程经验

### 起点配方（不要从零调）

参考第 3 章 3.10 节的决策表，**直接抄成熟模型的权重**：

```python
# Real-ESRGAN 风格 SR
loss = l1 + 1.0 * vgg + 0.1 * adv

# Restormer 风格去噪
loss = charb + 0.05 * fft + 0.001 * tv

# SUPIR 风格扩散
loss = simple_loss + 0.1 * latent_lpips + 0.01 * clip
```

### 启发式调参方法

如果初始权重效果差，按这个顺序调：

1. **先看主损失（像素或扩散 simple loss）是否正常下降**
2. 如果不正常，主损失权重提到 1，把所有辅助损失权重砍到 1/10
3. 主损失收敛后，**逐个**加大辅助损失权重，看效果
4. 永远不要同时改两个权重

### GradNorm（自适应权重）

更高级的方法：用 GradNorm 自动平衡多任务梯度：

```python
def gradnorm_step(losses: dict, weights: dict, alpha=0.12):
    """简化的 GradNorm: 让每个任务的 normalized gradient 接近平均值。"""
    grads = {}
    for name, loss in losses.items():
        grads[name] = torch.autograd.grad(
            loss * weights[name], 
            shared_params(),                # 共享层的参数
            retain_graph=True,
        )[0].norm()

    avg_grad = sum(grads.values()) / len(grads)

    # 学习率慢慢调整 weights, 让各 grad 接近 avg
    for name in weights:
        ratio = (grads[name] / avg_grad) ** alpha
        weights[name] = weights[name] / ratio  # 梯度大的减小权重
```

工程实践：GradNorm 在某些场景有用（多个差异大的辅助损失），但**不是默认选择**。先用经验权重，效果不够好再尝试。

## 11.14 训练监控：该看什么

每个 step 记录：

```python
# 每 50 步记录
log_dict = {
    # 损失项
    'loss/total': loss.item(),
    'loss/l1':    l1_loss.item(),
    'loss/vgg':   vgg_loss.item(),
    'loss/adv':   adv_loss.item(),

    # 梯度健康
    'grad/total_norm': grad_norm.item(),

    # 学习率
    'lr/g': optimizer_g.param_groups[0]['lr'],
    'lr/d': optimizer_d.param_groups[0]['lr'],
}
```

每 N 个 epoch 在 validation 集上算指标 + 视觉对比：

```python
# Validation
metrics = run_validation(model_ema, val_loader)
log_dict.update({
    'val/psnr':  metrics['psnr'],
    'val/lpips': metrics['lpips'],
    'val/dists': metrics['dists'],
})

# 视觉对比 (固定一组 LR, 看模型变化)
sample_outputs = generate_samples(model_ema, fixed_lr_batch)
log_image_grid(sample_outputs, step=step)
```

### 必须看的几条曲线

1. **总损失下降趋势**：稳定下降是健康
2. **各损失项相对量级**：一个项压倒其他项是不健康
3. **PSNR 在 val 上**：是否还在涨
4. **LPIPS 在 val 上**：感知质量
5. **GAN 时**：D 和 G loss 应该 oscillate 而不是单调
6. **梯度 norm**：是否爆炸（突然 spike）

### 视觉对比比指标更重要

有些问题只有视觉看才发现：

- 颜色漂移（指标都正常）
- 局部伪影（平均指标正常）
- 细节失真（PSNR 涨 LPIPS 反而升）

**每 N step 把固定一组 LR 跑一遍，把对比图存到 wandb/tensorboard**，比读数字更直观。

## 11.15 常见崩溃场景与诊断

| 症状 | 可能原因 | 检查 |
|------|---------|------|
| Loss = NaN | 梯度爆炸 / FP16 溢出 | 检查 grad_norm，换 BF16 |
| Loss 不下降 | LR 太大/太小、数据 bug | 先 overfit 一个 batch 看能不能收敛 |
| Val 不下降但 train 下降 | 过拟合 | 加 dropout / 减小模型 / 加数据 |
| Val 不下降也不上升 | 数据分布太窄 | 增加 degradation 多样性 |
| 输出全黑/白 | 数据归一化错（[-1,1] vs [0,1]）| 检查 dataloader 输出 |
| 某个 epoch 后突然崩 | LR schedule 出错 | 看 lr 曲线是否合理 |
| GAN D loss = 0 | D 过强 | 加 SpectralNorm + R1 |
| GAN 输出鬼东西 | G 没 pretrain | 先 pretrain G |
| 训练 1 epoch 极慢 | 数据加载瓶颈 | 增 num_workers，用 LMDB |

## 11.16 Checkpoint 与恢复

### 保存什么

```python
def save_checkpoint(path, step, model, optimizer, scheduler, ema, scaler):
    torch.save({
        'step': step,
        'model':     model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'ema':       ema.state_dict(),
        'scaler':    scaler.state_dict(),
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, ema, scaler):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    ema.load_state_dict(ckpt['ema'])
    scaler.load_state_dict(ckpt['scaler'])
    return ckpt['step']
```

### 频率

- **小模型**：每 5K-10K 步
- **大模型（扩散）**：每 1K-2K 步
- **保留几个**：最近 3 个 + 最佳 PSNR/LPIPS 一个

### 自动恢复

训练脚本启动时检测最新 checkpoint 自动 resume：

```python
def auto_resume(ckpt_dir, model, optimizer, scheduler, ema, scaler):
    ckpts = sorted(glob(f'{ckpt_dir}/step_*.pt'))
    if ckpts:
        latest = ckpts[-1]
        print(f'Resuming from {latest}')
        return load_checkpoint(latest, model, optimizer, scheduler, ema, scaler)
    return 0
```

这是长训练（几天甚至几周）的必备——总会遇到机器重启、CUDA OOM、网络断、电源故障。

## 11.17 工程经验：从一个能跑通的基线开始

### 不要从零搭新模型直接训

经验顺序：

1. **找一个开源模型 + 公开权重**，确认能跑、能在你机器上推理
2. **用一个小数据集（100 张图）pretrain**，确认训练循环正常
3. **overfit 一个 batch**：把训练数据缩到 8 张，跑 1000 步，看能不能 PSNR > 50（基本完美 fit）
4. 不能 overfit → 训练循环或模型有 bug
5. 能 overfit → 切到完整数据集

### 一次只改一处

科学方法：

- **基线明确**：知道 baseline 的指标和视觉效果
- **改一个变量**（一个损失权重、一个学习率、一个数据 augmentation）
- **跑足够长时间**确认效果
- **记录**：每个实验单独 wandb run，commit ID 关联

不这么做的代价：你改了 5 个东西，不知道哪个有效、哪个有害。

### 先小后大

- 先在 $128 \times 128$ patch 上证明思路
- 再上 $256 \times 256$
- 最后才上完整训练

## 11.18 小结

1. **优化器**：AdamW + cosine schedule + warmup（必须）
2. **EMA**：扩散和高质量 GAN 必须，decay 0.9999+
3. **AMP**：BF16 优于 FP16（A100+）
4. **GAN 训练靠 SpectralNorm + R1 + TTUR**，且**必须两阶段**（pretrain G → GAN finetune）
5. **扩散训练靠 Min-SNR 加权 + 时间步重要性采样 + EMA**
6. **混合损失从经验权重起步**，逐个调，不要同时改多个
7. **训练监控**：损失曲线 + 验证指标 + 视觉对比，三者缺一不可
8. **Checkpoint 频繁保存 + 自动恢复**，长训练的生存技能
9. **从基线开始** + **overfit 验证** + **一次改一处**，是工程经验最浓缩的三条

到这里 Part III 的训练章节完成。下一章讲评估方法论——客观指标的局限我们已经在第 4 章讲过，第 12 章重点是**怎么做主观评测**和**怎么在生产环境做 A/B 测试**。

---

> 下一章 [评估方法论](12-evaluation.md) → MOS、2AFC、统计显著性，影像增强里"做实验"这件事的方法论。
