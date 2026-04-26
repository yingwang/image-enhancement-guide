# Chapter 11 · Training Stability

> The first 10 chapters covered "which model, which loss, which data."
>
> This chapter is about "putting them together to train, and how to train stably."
>
> Training enhancement models is more fragile than training classifiers / LLMs—multi-loss conflicts, GAN dynamics, diffusion schedules, any one of them can leave you with three days of training and a collapsed model.

## 11.1 Why training stability is a big deal

LLM training has a single cross-entropy loss; the training dynamics are relatively simple. Enhancement models are entirely different:

- **Multi-loss mixtures**: L1 + VGG + GAN + task-specific—imbalanced weights and it collapses
- **GAN training**: the dynamic balance between D and G; if either side becomes too strong, it collapses
- **Diffusion training**: timestep sampling, loss weighting, EMA—you cannot skip any of them
- **Complex data pipeline**: degradation synthesis runs on the GPU, where bugs hide easily

This chapter consolidates the engineering pitfalls into an actionable checklist.

## 11.2 Training anatomy: basic components

The training loop of an enhancement model contains:

```python
# Abstract skeleton
optimizer = build_optimizer(model)
scheduler = build_scheduler(optimizer)
scaler = torch.cuda.amp.GradScaler()                # AMP
ema = EMA(model, decay=0.999)                       # exponential moving average

for step, batch in enumerate(train_loader):
    # 1. Warmup
    if step < warmup_steps:
        adjust_lr_for_warmup(optimizer, step, warmup_steps)

    # 2. Forward (with AMP)
    with torch.cuda.amp.autocast():
        loss = compute_loss(model, batch)

    # 3. Backward
    scaler.scale(loss).backward()

    # 4. Gradient clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 5. Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    # 6. Learning rate schedule
    scheduler.step()

    # 7. EMA update
    ema.update(model)

    # 8. Monitoring
    if step % log_interval == 0:
        log_metrics(...)
    if step % eval_interval == 0:
        evaluate(...)
    if step % save_interval == 0:
        save_checkpoint(...)
```

Each item is unpacked below.

## 11.3 Optimizer choice

| Optimizer | When to use |
|-------|------|
| **Adam** | Classic choice, stable |
| **AdamW** | Adds decoupled weight decay, **default recommendation** |
| **Lion** | New optimizer from 2023, saves memory (only needs momentum), close to AdamW |
| **Adafactor** | Saves memory on large models (no second moment) |
| **SGD + momentum** | Not recommended for enhancement (slow convergence, sensitive hyperparameters) |

**Starting configuration** (general for enhancement tasks):

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-4,                      # SR/denoising empirical value
    betas=(0.9, 0.99),            # 0.99 instead of 0.999, more stable
    weight_decay=0.01,
    eps=1e-8,
)
```

Learning rate empirics:

- **CNN (EDSR/NAFNet)**: $2 \times 10^{-4}$
- **Transformer (SwinIR/Restormer)**: $2 \times 10^{-4}$, warmup mandatory
- **GAN finetune**: $10^{-4}$ for G, $10^{-4} \times 4 = 4 \times 10^{-4}$ for D (TTUR)
- **Diffusion from scratch**: $10^{-4}$
- **Diffusion finetune**: $10^{-5}$ to $5 \times 10^{-6}$
- **ControlNet training**: $10^{-5}$

## 11.4 Learning rate schedule

### Linear Warmup

In the early phase, LR ramps linearly from 0 to the target value. **Mandatory** for Transformer and diffusion training—without it the first few steps may diverge directly.

```python
def linear_warmup(step: int, warmup_steps: int, target_lr: float) -> float:
    if step >= warmup_steps:
        return target_lr
    return target_lr * (step + 1) / warmup_steps
```

Empirical values for warmup_steps:

- **1-5%** of total training steps
- Large models (diffusion): 5000-10000 steps
- Small models (CNN SR): 500-1000 steps

### Cosine Decay

After warmup, use cosine to bring LR down to the final value (typically 1-10% of the LR):

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

### Multi-step decay (used by classic SR)

ESRGAN/Real-ESRGAN use stepwise decay:

```python
scheduler = sched.MultiStepLR(
    optimizer,
    milestones=[200_000, 400_000, 600_000, 800_000],   # train for 1M steps
    gamma=0.5,                                          # LR×0.5 each time
)
```

### Cosine Restart

In the late phase, periodically "restart" LR to a high value to escape local optima:

```python
scheduler = sched.CosineAnnealingWarmRestarts(
    optimizer, T_0=100_000, T_mult=1, eta_min=1e-6
)
```

Engineering experience: cosine restart is especially recommended for diffusion training—it keeps gaining points over long training runs.

## 11.5 Batch Size and Patch Size

A peculiarity of low-level vision training: **almost never train on whole images**, use patches.

### Standards for patch-based training

- Randomly crop a patch from HR (typically $256 \times 256$ or $128 \times 128$)
- Apply degradation synthesis to obtain the corresponding LR patch
- The training batch is a batch of patches

Why not train directly on whole images:

- Memory: a single batch of $2048 \times 2048$ won't fit
- Data augmentation: cropping itself is implicit data augmentation
- Training efficiency: for the same GPU time, patch training sees more diversity

### Patch size choice

| Task | Recommended HR patch size | Reason |
|------|-----------------|------|
| 4× SR | 256 | LR=64, enough compute |
| 8× SR | 384 | LR=48, more spatial context needed |
| Denoising | 128-192 | Local texture is enough |
| Deblurring | 256-384 | Large blur kernels need large patches |
| Diffusion | 512 | Matches pretrained SD |

### Effective batch size

```
effective_batch_size = num_patches_per_image × image_batch × gradient_accumulation × num_gpus
```

Common configurations:

- Single A100, CNN SR: 8 images × 1 patch × 1 = 8
- 4× A100, diffusion: 16 images × 1 patch × 2 GA × 4 GPU = 128

### LR scaling for large batches

Rule of thumb: batch size × 2, LR × √2. But low-level vision is sensitive to LR, so it is recommended to **measure** rather than blindly scale.

## 11.6 Gradient Clipping

The standard equipment against gradient explosions.

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0,    # empirical value for enhancement
)
```

Empirical values for `max_norm`:

- CNN: 1.0 - 5.0
- Transformer: 1.0
- GAN: 0.5 (more aggressive)
- Diffusion: 1.0

## 11.7 EMA: required for diffusion and high-quality GAN

EMA (Exponential Moving Average) maintains an exponential moving average of the model weights:

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

Empirical values for EMA decay:

- CNN SR: 0.999 (not mandatory, but PSNR gains 0.05-0.1 dB)
- GAN: 0.999 (inference output is steadier with EMA)
- Diffusion: **0.9999** or **0.99995** (mandatory, paper standard)

EMA is especially important in diffusion training—the FID of pure training weights is usually a few points worse than that of EMA weights.

## 11.8 Mixed Precision (AMP)

Compute in FP16/BF16, accumulate in FP32. Saves memory + ~1.5-2× speedup.

```python
scaler = torch.cuda.amp.GradScaler()

for batch in loader:
    optimizer.zero_grad()

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):  # or torch.float16
        loss = compute_loss(model, batch)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

### FP16 vs BF16

| Dimension | FP16 | BF16 |
|------|------|------|
| Numerical range | Small (overflows easily) | Large (close to FP32) |
| Precision | High | Low |
| Hardware | V100, A100, H100 | A100, H100 |
| Recommendation | Old hardware | **Default for A100 and later** |

In low-level vision BF16 is almost always the better choice—numerically stable, and no GradScaler needed.

### Which layers cannot use low precision

- **VAE encoder/decoder**: in diffusion, VAE is usually FP32 or FP16, not BF16 (insufficient numerical precision)
- **softmax**: in Transformers, softmax is more stable in FP32; diffusers/transformers handle this automatically
- **Loss computation**: can be FP32

## 11.9 GAN training collapse: diagnosis and treatment

GAN training is the most collapse-prone part of low-level vision. Common symptoms:

### Symptom 1: mode collapse

All output images are similar (regardless of the input LR). Diagnosis:

- Check the diversity of G's output (use a fixed set of LR and look at outputs)
- Look at D loss: if D loss is extremely low (< 0.01), D is too strong and G can't learn

### Symptom 2: D too strong

```
D_loss → 0
G_loss not decreasing / oscillating
```

Treatment:

- Lower D's learning rate (keep G unchanged)
- Add spectral normalization to D
- Add R1 regularization
- Reduce D training frequency (train D every 2 steps)

### Symptom 3: D too weak

```
D_loss → 1 / not converging
G_loss also not converging
```

Treatment:

- Raise D's learning rate
- Increase D capacity (deeper/wider)
- Reduce G capacity

### Symptom 4: training divergence

```
loss suddenly becomes NaN
```

Treatment:

- Check whether gradient norm is exploding (log it out)
- Add gradient clipping
- Lower LR
- Check whether data has inf/NaN

## 11.10 GAN stabilization techniques

### Spectral Normalization

Controls D's Lipschitz constant so that D cannot become arbitrarily steep.

```python
from torch.nn.utils import spectral_norm

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

Engineering experience: **almost all modern GANs use SpectralNorm**, and adding it makes things much more stable.

### R1 Regularization

Penalize D's gradient on real samples:

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

Add it to D loss:

```python
d_loss = d_loss_main + 10.0 * r1_penalty(d_real, real_imgs)
```

R1 keeps D from being too steep around the real data, mitigating the D-too-strong problem.

### TTUR

D uses a larger learning rate than G (typically 4×):

```python
opt_g = AdamW(g.parameters(), lr=1e-4)
opt_d = AdamW(d.parameters(), lr=4e-4)   # 4× lr
```

## 11.11 Two-stage training (strongly recommended)

Experience with low-level vision GAN training: **always train G alone to convergence first, then add the GAN loss**.

### Stage 1: Pretrain G

```python
# Only pixel loss + perceptual loss, no GAN
loss = l1_loss + 1.0 * vgg_loss
# Train until PSNR converges
```

Typically requires 200K-500K steps.

### Stage 2: GAN finetune

```python
# Add GAN loss with a small weight (0.005 - 0.1)
g_loss = l1_loss + 1.0 * vgg_loss + 0.05 * adv_loss
# Lower learning rate to 1/2 of pretrain
```

Typically 100K-200K steps.

### Why two stages

Issues with direct joint training (all-in-one):

- The GAN loss has large, noisy gradients early on, which destroy pixel consistency
- The model has not yet learned basic restoration ability; forcing it to "generate details" with GAN produces garbage
- Loss weighting is hard to tune properly

ESRGAN, Real-ESRGAN, BSRGAN, etc. all use a two-stage strategy. **This is the de facto standard**, do not try to innovate by skipping a stage.

## 11.12 Specifics of diffusion training

### Timestep sampling

Uniform sampling is the baseline, but some $t$ ranges contribute more:

```python
def importance_sampling_t(B, num_steps=1000):
    """Non-uniform timestep sampling, biased toward the middle region."""
    # Empirical: t ∈ [200, 800] contributes the most to final quality
    weights = torch.ones(num_steps)
    weights[:200]  *= 0.5
    weights[800:]  *= 0.5
    weights = weights / weights.sum()
    return torch.multinomial(weights, B, replacement=True)
```

### Min-SNR weighting (review of Chapter 3, Section 3.6)

```python
def min_snr_weight(t, alphas_cumprod, gamma=5.0):
    """Min-SNR weighting, standard in SDXL."""
    snr = alphas_cumprod[t] / (1 - alphas_cumprod[t])
    return torch.minimum(snr, torch.full_like(snr, gamma)) / snr
```

Add it to loss:

```python
loss = (min_snr_weight(t, alphas_cumprod).view(-1, 1, 1, 1) *
        (pred_noise - true_noise) ** 2).mean()
```

### v-prediction vs eps-prediction

Discussed in Chapter 3, Section 3.6. Engineering practice:

- **New training**: v-prediction is more stable
- **Finetuning from SD 1.5**: keep the original eps-prediction
- **Based on SD 2.x / SDXL refiner**: v-prediction

### Multi-GPU large models (FSDP)

The SDXL UNet has 2.6B parameters and won't fit on a single card. Use FSDP (Fully Sharded Data Parallel):

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

## 11.13 Engineering experience for mixed loss weights

### Starting recipe (don't tune from zero)

Refer to the decision table in Chapter 3, Section 3.10, and **directly copy the weights of a mature model**:

```python
# Real-ESRGAN-style SR
loss = l1 + 1.0 * vgg + 0.1 * adv

# Restormer-style denoising
loss = charb + 0.05 * fft + 0.001 * tv

# SUPIR-style diffusion
loss = simple_loss + 0.1 * latent_lpips + 0.01 * clip
```

### Heuristic tuning method

If the initial weights perform poorly, tune in this order:

1. **First check whether the main loss (pixel or diffusion simple loss) is descending normally**
2. If not, raise the main loss weight to 1 and cut all auxiliary loss weights to 1/10
3. Once the main loss converges, **incrementally** raise the auxiliary loss weights and observe
4. Never change two weights at the same time

### GradNorm (adaptive weights)

A more advanced method: use GradNorm to automatically balance multi-task gradients:

```python
def gradnorm_step(losses: dict, weights: dict, alpha=0.12):
    """Simplified GradNorm: bring each task's normalized gradient close to the average."""
    grads = {}
    for name, loss in losses.items():
        grads[name] = torch.autograd.grad(
            loss * weights[name], 
            shared_params(),                # parameters of the shared layers
            retain_graph=True,
        )[0].norm()

    avg_grad = sum(grads.values()) / len(grads)

    # Slowly adjust weights so each grad is close to avg
    for name in weights:
        ratio = (grads[name] / avg_grad) ** alpha
        weights[name] = weights[name] / ratio  # tasks with large grads get smaller weights
```

Engineering practice: GradNorm is useful in some scenarios (multiple very different auxiliary losses), but **it is not the default choice**. Use empirical weights first, and try GradNorm if results are not good enough.

## 11.14 Training monitoring: what to watch

Log every step:

```python
# Log every 50 steps
log_dict = {
    # Loss terms
    'loss/total': loss.item(),
    'loss/l1':    l1_loss.item(),
    'loss/vgg':   vgg_loss.item(),
    'loss/adv':   adv_loss.item(),

    # Gradient health
    'grad/total_norm': grad_norm.item(),

    # Learning rate
    'lr/g': optimizer_g.param_groups[0]['lr'],
    'lr/d': optimizer_d.param_groups[0]['lr'],
}
```

Every N epochs, compute metrics and visual comparisons on the validation set:

```python
# Validation
metrics = run_validation(model_ema, val_loader)
log_dict.update({
    'val/psnr':  metrics['psnr'],
    'val/lpips': metrics['lpips'],
    'val/dists': metrics['dists'],
})

# Visual comparison (fixed set of LR, observe model evolution)
sample_outputs = generate_samples(model_ema, fixed_lr_batch)
log_image_grid(sample_outputs, step=step)
```

### Curves you must watch

1. **Total loss descent trend**: steady descent is healthy
2. **Relative magnitudes of loss terms**: one term dominating the others is unhealthy
3. **PSNR on val**: is it still rising
4. **LPIPS on val**: perceptual quality
5. **In GAN training**: D and G loss should oscillate rather than be monotonic
6. **Gradient norm**: is it exploding (sudden spike)

### Visual comparison matters more than metrics

Some problems are only revealed by looking:

- Color drift (metrics all normal)
- Local artifacts (average metrics normal)
- Detail distortion (PSNR rising while LPIPS instead increases)

**Every N steps, run a fixed set of LR, save the comparison images to wandb/tensorboard**—it's far more intuitive than reading numbers.

## 11.15 Common collapse scenarios and diagnosis

| Symptom | Possible cause | Check |
|------|---------|------|
| Loss = NaN | Gradient explosion / FP16 overflow | Check grad_norm, switch to BF16 |
| Loss not decreasing | LR too large/small, data bug | First overfit a single batch and see if it converges |
| Val not decreasing while train decreases | Overfitting | Add dropout / shrink model / add data |
| Val neither decreasing nor increasing | Data distribution too narrow | Increase degradation diversity |
| Output all black/white | Wrong data normalization ([-1,1] vs [0,1]) | Check dataloader output |
| Sudden collapse after some epoch | LR schedule misconfigured | Check whether the lr curve is reasonable |
| GAN D loss = 0 | D too strong | Add SpectralNorm + R1 |
| GAN outputs garbage | G not pretrained | Pretrain G first |
| 1 epoch extremely slow | Data loading bottleneck | Increase num_workers, use LMDB |

## 11.16 Checkpoint and recovery

### What to save

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

### Frequency

- **Small models**: every 5K-10K steps
- **Large models (diffusion)**: every 1K-2K steps
- **Keep a few**: latest 3 + one with best PSNR/LPIPS

### Auto resume

When the training script starts, detect the most recent checkpoint and auto-resume:

```python
def auto_resume(ckpt_dir, model, optimizer, scheduler, ema, scaler):
    ckpts = sorted(glob(f'{ckpt_dir}/step_*.pt'))
    if ckpts:
        latest = ckpts[-1]
        print(f'Resuming from {latest}')
        return load_checkpoint(latest, model, optimizer, scheduler, ema, scaler)
    return 0
```

This is essential for long training runs (days or weeks)—you will always run into machine restarts, CUDA OOM, network interruption, power failure.

## 11.17 Engineering experience: start from a baseline that runs

### Don't train a brand-new model from scratch

Empirical order:

1. **Find an open-source model + public weights**, confirm it runs and you can do inference on your machine
2. **Pretrain on a small dataset (100 images)**, confirm the training loop is fine
3. **Overfit a single batch**: shrink training data to 8 images, run 1000 steps, see if you can hit PSNR > 50 (basically perfect fit)
4. Cannot overfit → there's a bug in the training loop or model
5. Can overfit → switch to the full dataset

### Change one thing at a time

Scientific method:

- **Have a clear baseline**: know the metrics and visual results of the baseline
- **Change a single variable** (one loss weight, one learning rate, one data augmentation)
- **Run long enough** to confirm the effect
- **Record**: each experiment as its own wandb run, linked to the commit ID

The cost of not doing this: you change 5 things and don't know which helped, which hurt.

### Start small, then go big

- First prove the idea on $128 \times 128$ patches
- Then move up to $256 \times 256$
- Only then run full training

## 11.18 Summary

1. **Optimizer**: AdamW + cosine schedule + warmup (mandatory)
2. **EMA**: mandatory for diffusion and high-quality GAN, decay 0.9999+
3. **AMP**: BF16 is better than FP16 (A100+)
4. **GAN training relies on SpectralNorm + R1 + TTUR**, and **must be two-stage** (pretrain G → GAN finetune)
5. **Diffusion training relies on Min-SNR weighting + timestep importance sampling + EMA**
6. **For mixed losses, start from empirical weights**, tune them one at a time, never change multiple at once
7. **Training monitoring**: loss curves + validation metrics + visual comparison, all three are necessary
8. **Save checkpoints frequently + auto-resume**, the survival skill of long training runs
9. **Start from a baseline** + **overfit verification** + **change one thing at a time**, the three most distilled engineering lessons

That completes the training chapter of Part III. The next chapter discusses evaluation methodology—we already covered the limitations of objective metrics in Chapter 4; Chapter 12 focuses on **how to do subjective evaluation** and **how to do A/B testing in production**.

---

> Next chapter [Evaluation Methodology](12-evaluation.md) → MOS, 2AFC, statistical significance: the methodology of "running experiments" in image enhancement.
