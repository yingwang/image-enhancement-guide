# Image Enhancement: From Principles to Engineering

> Starting from the degradation model y = D(x) + n, build first-principles understanding of image enhancement and master the full engineering practice from CNN/Transformer to diffusion models.

**Languages**: [中文](../README.md) | [English](README.md)
**Read online**: [yingwang.github.io/image-enhancement-guide](https://yingwang.github.io/image-enhancement-guide/)

For engineers comfortable with PyTorch and with prior LLM or image-classification experience, but new to low-level vision systematically. This book does not teach you how to run a particular training script — it gives you the framework to look at a degraded image and immediately know "this kind of degradation needs that kind of prior, that kind of loss, that kind of architecture; if I have to deploy on-device I'll cut here."

## The Logic of This Book

```mermaid
flowchart LR
    subgraph P1["<b>Part I: Foundations</b>"]
        direction TB
        C1["1. Degradation Model"]
        C2["2. Pixel/Feature/Latent"]
        C3["3. Loss Landscape"]
        C4["4. Pitfalls of Metrics"]
        C5["5. Data & Degradation"]
        C1 --> C2 --> C3 --> C4 --> C5
    end

    subgraph P2["<b>Part II: Architectures</b>"]
        direction TB
        C6["6. CNN Era"]
        C7["7. Transformer"]
        C8["8. Diffusion Basics"]
        C9["9. Diffusion Control"]
        C10["10. Task-Specific Models"]
        C6 --> C7 --> C8 --> C9 --> C10
    end

    subgraph P3["<b>Part III: Training & Eval</b>"]
        direction TB
        C11["11. Training Stability"]
        C12["12. Evaluation Methodology"]
        C11 --> C12
    end

    subgraph P4["<b>Part IV: Video</b>"]
        direction TB
        C13["13. Video ≠ Image × N"]
        C14["14. VSR / VFI / Restoration"]
        C13 --> C14
    end

    subgraph P5["<b>Part V: Engineering</b>"]
        direction TB
        C15["15. Inference Optimization"]
        C16["16. Real-World Cases"]
        C17["17. Failure Modes"]
        C15 --> C16 --> C17
    end

    subgraph P6["<b>Part VI: Reference</b>"]
        direction TB
        C18["18. SOTA Models"]
    end

    P1 --> P2 --> P3 --> P4
    P3 --> P5
    P5 --> P6

    classDef chapter fill:#ffffff,stroke:#555,color:#222
    class C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18 chapter

    style P1 fill:#e8eaf6,stroke:#3949ab,color:#1a237e
    style P2 fill:#fff3e0,stroke:#e65100,color:#bf360c
    style P3 fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    style P4 fill:#fce4ec,stroke:#c62828,color:#880e4f
    style P5 fill:#e0f2f1,stroke:#00695c,color:#004d40
    style P6 fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c
```

**Core logic**: first understand "how images degrade" (degradation model) → then "how we measure improvement" (loss & metrics) → then architectures, training tricks, video-specific issues, and engineering deployment.

Most existing material on image enhancement is either papers (one specific method) or open-source READMEs (how to run them). This book is what was missing in the middle: **the engineering philosophy of the field**, made explicit.

## In Scope vs Out of Scope

**In scope**: super-resolution (SR), denoising, deblurring, JPEG/compression-artifact removal, demoiré, low-light enhancement, HDR, derain/dehaze, face restoration, old-photo restoration, colorization, video super-resolution, frame interpolation, video stabilization, ISP enhancement basics.

**Out of scope**:

- Pure generation (text-to-image / text-to-video, e.g. SD/Sora/Veo)
- Image editing (clothing/face swap, style transfer)
- 3D / NeRF / Gaussian Splatting
- Video understanding (action recognition, video QA, Video-LLM)
- Image compression and codecs (H.264/AV1, neural compression)
- Medical reconstruction, satellite remote sensing (principle mentioned, not developed)
- Hardware ISP (only touched briefly in the low-light chapter)

The dividing line: **if there is a notion of an "original signal" we are estimating, it is enhancement; if it's "from nothing", it is generation**.

## Table of Contents

### Part I · Foundations

| # | Chapter | Core question |
|---|---------|---------------|
| 1 | [Degradation Model & Inverse Problems](chapters/01-degradation-model.md) | Enhancement is an ill-posed inverse problem — where do priors come from? |
| 2 | [Pixel, Feature, Latent](chapters/02-representation.md) | Why do all modern methods work in latent space? |
| 3 | [Loss Landscape](chapters/03-losses.md) | What is each of L1/perceptual/adversarial/diffusion losses optimizing? |
| 4 | [Pitfalls of Metrics](chapters/04-metrics.md) | What does each of PSNR/SSIM/LPIPS/FID hide from you? |
| 5 | [Data & Degradation Synthesis](chapters/05-degradation-pipeline.md) | Real-ESRGAN's true contribution is the data pipeline |

### Part II · Architectures

| # | Chapter | Core question |
|---|---------|---------------|
| 6 | [The CNN Era](chapters/06-cnn.md) | The evolution from SRCNN to NAFNet |
| 7 | [Transformer in Low-Level Vision](chapters/07-transformer.md) | Why is attention useful for restoration? |
| 8 | [Diffusion Basics](chapters/08-diffusion.md) | From DDPM to LDM — why can diffusion "create from nothing"? |
| 9 | [Conditioning Diffusion](chapters/09-control.md) | ControlNet / IP-Adapter / Tile / multi-condition control |
| 10 | [Task-Specific Models](chapters/10-task-specific.md) | Inductive biases of face / document / medical |

### Part III · Training & Evaluation

| # | Chapter | Core question |
|---|---------|---------------|
| 11 | [Training Stability](chapters/11-training.md) | GAN collapse, diffusion scheduling, multi-loss balancing |
| 12 | [Evaluation Methodology](chapters/12-evaluation.md) | Limits of objective metrics + how to design subjective evals |

### Part IV · Video

| # | Chapter | Core question |
|---|---------|---------------|
| 13 | [Video Is Not Image × N](chapters/13-video-basics.md) | Temporal consistency is an independent problem |
| 14 | [VSR / VFI / Restoration](chapters/14-video-models.md) | BasicVSR++, RIFE, video stabilization |

### Part V · Engineering & Deployment

| # | Chapter | Core question |
|---|---------|---------------|
| 15 | [Inference Optimization](chapters/15-inference.md) | Quantization, TensorRT, CoreML, mobile |
| 16 | [Real-World Cases](chapters/16-cases.md) | Old-photo restoration, low-light, UGC, 4K live, ISP |
| 17 | [Failure Modes](chapters/17-failures.md) | The typical ways enhancement models fail in production |

### Part VI · Reference

| # | Chapter | Core question |
|---|---------|---------------|
| 18 | [SOTA Models](chapters/18-sota.md) | 7 academic SOTA models worth knowing in 2026 |

## Relationship to other books

| | [LLM Training Guide](https://github.com/yingwang/llm-tutorial) | [Thinking in LLM](https://github.com/yingwang/thinking-in-llm) | This book |
|---|---|---|---|
| **Domain** | LLM training | LLM applications | Low-level vision |
| **Form** | Engineering guide + code | First principles | Engineering guide + snippets |
| **Audience** | Training engineers | LLM application engineers | Image/CV engineers |
| **Prerequisites** | ML basics | Programming basics | PyTorch + a bit of ML |

## How to Read

- **Front to back**: Part I → II → III → IV → V → VI. Chapter 1 is the foundation, read it regardless.
- **In a hurry**: Ch 1 → Ch 5 → Ch 8 → Ch 18 (central equation + data synthesis + diffusion basics + current SOTA).
- **Already familiar with the area**: jump to Part V cases, look back at earlier chapters when something is unclear.
- **Building a product**: read Ch 16-17 first, then Ch 9 and Ch 15.

## Author

Ying Wang

## License

This repository is **dual-licensed**:

- **Prose and diagrams** (Mermaid, explanatory text, chapter content): [CC BY-NC-SA 4.0](../LICENSE) — Attribution · NonCommercial · ShareAlike
- **Code snippets** (Python examples within chapters): [MIT](../LICENSE-CODE) — free to use, including commercial, with copyright notice retained

---

> Last updated: 2026-04-27
