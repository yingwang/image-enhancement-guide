# Chapter 12 · Evaluation Methodology

> Chapter 4 discussed **the metrics themselves**—where PSNR/LPIPS/FID each deceive you.
>
> This chapter discusses **how to run an evaluation**—how to design a subjective evaluation, how to compute statistical significance, how to do A/B testing in production.
>
> In image enhancement, "running an experiment" itself has a methodology, and most papers do not get this step right.

## 12.1 The three layers of evaluation

Any serious enhancement evaluation should cover three layers:

```
Layer 1: Automatic metrics
  PSNR / SSIM / LPIPS / DISTS / FID / NIQE
  Pros: fast, reproducible, cheap
  Cons: limited correlation with human eye (covered in Chapter 4)

Layer 2: Subjective human evaluation
  MOS / 2AFC / preference test
  Pros: reflects real visual experience
  Cons: slow, expensive, noisy

Layer 3: Real-world downstream tasks
  A/B test / user behavior / downstream recognition accuracy
  Pros: directly reflects value
  Cons: requires deployment, limited data
```

**The three layers must validate each other**: an automatic-metric improvement that does not show up in subjective evaluation or downstream tasks is suspect.

This chapter focuses on Layer 2 and Layer 3.

## 12.2 MOS: single-image scoring

MOS (Mean Opinion Score) is the most classic form of subjective evaluation.

### Procedure

1. Show the rater an image (the enhancement model's output)
2. The rater scores on a 5-level Likert scale:

```
1 - Bad (severe artifacts/blur)
2 - Poor (clear problems)
3 - Fair (usable but not good)
4 - Good (high quality)
5 - Excellent (flawless)
```

3. Multiple raters score the same image independently; take the average

### Pros

- **Simple**: raters need no professional training
- **Scalable**: scoring an image takes seconds, can cover many samples
- **Task-agnostic**: can compare models across tasks

### Cons

- **Inconsistent scales**: rater A's 4 ≠ rater B's 4 (A is strict, B is lenient)
- **No reference comparison**: looking at an isolated image, the rater may not know "is this a 4 or a 5"
- **High data noise**: standard deviation is often ±0.5 points

### Tricks for raising MOS SNR

1. **At least 5-10 raters per image**, take the average to reduce per-rater bias
2. **Calibration samples**: place a few images of known quality up front (HR ground truth gets 5, severe blur gets 1) so the rater calibrates their own scale
3. **Z-score normalization**: subtract each rater's own mean and divide by their own std before merging

```python
import numpy as np

def normalize_mos_scores(scores: dict) -> dict:
    """
    scores: {rater_id: {image_id: score}}
    returns: {image_id: normalized average}
    """
    # Z-score each rater's scores
    normalized = {}
    for rater, ratings in scores.items():
        s = np.array(list(ratings.values()))
        mean, std = s.mean(), s.std() + 1e-8
        normalized[rater] = {img: (sc - mean) / std for img, sc in ratings.items()}

    # Average across raters
    image_scores = {}
    for rater_dict in normalized.values():
        for img, sc in rater_dict.items():
            image_scores.setdefault(img, []).append(sc)
    return {img: np.mean(scs) for img, scs in image_scores.items()}
```

### MOS is recommended for

- Overall quality evaluation
- Quality distribution across multiple samples of a single model
- Cross-task comparison (**not recommended**—different tasks have different baselines)

## 12.3 2AFC: forced choice

2AFC (Two-Alternative Forced Choice) makes raters **directly compare** two candidates.

### Procedure

```
Show the reference image (HR ground truth)
Show candidate A (output of model 1)
Show candidate B (output of model 2)
Rater chooses: A is closer / B is closer / cannot tell
```

### Pros

- **Forced comparison**: humans are far more sensitive to "A vs B" than to "absolute scores"
- **Less data noise**: no need to calibrate scales
- **Simple statistics**: directly compute preference rate (A's win rate)

### Cons

- **Only pairwise**: 5 models means $\binom{5}{2} = 10$ pairs, each pair with multiple images
- **No absolute score**: you know A is better than B but not by how much
- **Order bias**: raters may prefer the option on the left (must randomize)

### Practical recommendation: **more reliable than MOS**

The vast majority of serious paper/product comparisons use 2AFC rather than MOS—unless you are only evaluating the overall quality of one single model.

```python
def compute_preference_rate(votes: list) -> dict:
    """
    votes: list of (model_a, model_b, winner)
    returns: {(a, b): a_win_rate}
    """
    from collections import defaultdict
    counts = defaultdict(lambda: {'a_wins': 0, 'b_wins': 0, 'tie': 0, 'total': 0})

    for a, b, winner in votes:
        key = tuple(sorted([a, b]))   # normalize direction
        counts[key]['total'] += 1
        if winner == 'tie':
            counts[key]['tie'] += 1
        elif winner == key[0]:
            counts[key]['a_wins'] += 1
        else:
            counts[key]['b_wins'] += 1

    return {k: v['a_wins'] / v['total'] for k, v in counts.items()}
```

## 12.4 Sample size for subjective evaluation

How many images and how many raters are needed to reach a reliable conclusion?

### Sample size calculation (power analysis)

To detect a 5% preference-rate difference (e.g. A win rate 50% vs 55%) requires:

$$
n = \left( \frac{z_{1-\alpha/2} + z_{1-\beta}}{p_1 - p_2} \right)^2 \cdot \left( p_1(1-p_1) + p_2(1-p_2) \right)
$$

Plugging in $\alpha = 0.05$ (significance level), $\beta = 0.2$ (power 0.8), $p_1 = 0.55, p_2 = 0.5$:

$$
n \approx 1500 \text{ pairs}
$$

In other words, **to reliably distinguish two close models, each pair needs about 1500 2AFC votes**.

Common compromises in practice:

- **Coarse screening** (gap 10%+): 100-300 pairs
- **Normal comparison** (gap 3-5%): 500-1000 pairs
- **Subtle differences** (gap 1-2%): 3000+ pairs (very expensive)

### Number of raters

How many distinct raters per image pair? Empirical:

- **Academic papers**: 5-10 distinct raters per pair
- **Product A/B**: 100+ independent users

## 12.5 Sources of raters

### Amazon Mechanical Turk (AMT)

The most classic crowdsourcing platform. Pros: many people, cheap ($0.01-0.05/task); cons: variable quality.

### Prolific

A more modern alternative. Quality is higher than AMT, price slightly higher ($0.10-0.20/task).

### Self-built platform

Deploy a rating system internally and have employees / recruited volunteers rate. Quality is controllable but sample size is limited.

### Professional evaluation agencies

Specialized video/image evaluation companies (such as Telecom ParisTech IPI lab). Highest quality, highest price (thousands of dollars per batch).

### Rater quality control

Required no matter the platform:

1. **Catch trial**: include 5-10% "obvious-answer" samples (HR vs an extremely poor output); raters must pick correctly
2. **Test-retest**: rate the same sample twice; remove raters with low consistency
3. **Time check**: remove raters who score too quickly (< 5 sec/image)
4. **Multiple raters per item**: a single rater cannot determine ground truth

## 12.6 Statistical significance

Before claiming "model A is better than B" you must run a significance test.

### Paired t-test (continuous metrics)

If you compare LPIPS of two models on the same set of test images:

```python
from scipy import stats

lpips_a = [...]   # LPIPS of model A on N test images
lpips_b = [...]   # model B
t_stat, p_value = stats.ttest_rel(lpips_a, lpips_b)
print(f"t = {t_stat:.3f}, p = {p_value:.4f}")
```

Only `p < 0.05` allows you to claim statistical significance.

### Wilcoxon Signed-Rank Test (non-parametric)

LPIPS/PSNR don't necessarily follow a normal distribution; Wilcoxon is more robust:

```python
from scipy.stats import wilcoxon
stat, p = wilcoxon(lpips_a, lpips_b)
```

### Binomial Test (2AFC)

Significance for 2AFC preference rate:

```python
from scipy.stats import binomtest

# Model A won 540 out of 1000 pairs
result = binomtest(540, 1000, p=0.5, alternative='two-sided')
print(f"A win rate 54%, p = {result.pvalue:.4f}")
```

### Multiple comparison correction

If you are comparing 5 models (10 pairs), p-values must be Bonferroni-corrected: $\alpha_{\text{corrected}} = 0.05 / 10 = 0.005$.

## 12.7 Correlation between automatic metrics and MOS

Correlation between different metrics and human perception (statistics from multiple IQA benchmarks):

| Metric | Spearman correlation with MOS |
|------|----------------------|
| PSNR | 0.40 - 0.55 |
| SSIM | 0.50 - 0.65 |
| MS-SSIM | 0.55 - 0.70 |
| **LPIPS** | **0.70 - 0.80** |
| **DISTS** | **0.72 - 0.82** |
| **MANIQA** (NR) | **0.75 - 0.85** |
| FID (distribution-level) | 0.40 - 0.60 |

This tells us:

- **PSNR alone is unreliable** (correlation only 0.4-0.55)
- **LPIPS/DISTS are the strongest full-reference metrics** (0.7+)
- **The SOTA NR metric (MANIQA) is already close to LPIPS**—when there's no ground truth in real-world settings, MANIQA is a decent proxy

Engineering practice:

- During training, watch PSNR + LPIPS
- For papers/reports, beyond PSNR + LPIPS also do subjective evaluation
- For production deployment, watch MANIQA + user behavior

## 12.8 Real-world: downstream task evaluation

The most convincing evaluation: **does the enhanced image make a downstream task perform better?**

Examples:

### Enhancement + OCR

```python
# Test set: 1000 low-quality document images
# Ground truth: known text content

original_ocr_acc  = test_ocr(low_quality_images)         # OCR accuracy on raw images: 65%
enhanced_ocr_acc  = test_ocr(model_output(low_quality))  # after enhancement: 88%

# This is what "real contribution of the enhancement model to the OCR task" means
```

### Enhancement + face recognition

```python
# Test set: 1000 pairs of (low-quality image, high-quality reference)
# Ground truth: same person or not

original_recognition = face_recognize(low_quality_images, gallery)  # accuracy: 70%
enhanced_recognition = face_recognize(model_output(...), gallery)   # accuracy: 85%
```

### Enhancement + object detection

```python
# Test set: 1000 low-quality surveillance images, with object annotations
mAP_original = detection_eval(low_quality_images, annotations)  # 0.45
mAP_enhanced = detection_eval(model_output(...), annotations)   # 0.62
```

Pros of downstream task evaluation:

- **Directly reflects value**—what customers/users actually care about
- **No need for human rating**—reuse off-the-shelf downstream models
- **Objective and reproducible**—accuracy/mAP are absolute numbers

Applicable scenarios:

- Document enhancement → OCR
- Face enhancement → recognition / verification
- Surveillance enhancement → detection / person re-identification
- Satellite enhancement → land cover classification / change detection

Inapplicable scenarios:

- Art restoration (no "task")
- Beauty filters (user-preference driven)

## 12.9 A/B testing in production

Deploy the new model to a fraction of users, compare against the old model, observe user-behavior changes.

### A/B design

```
50% of users → old model (control)
50% of users → new model (treatment)
```

Monitored metrics:

- **Direct quality metrics** (active user feedback): like rate, report rate, reprocess rate
- **Behavior metrics**: processing duration, repeat-use rate, subscription conversion rate
- **Exit metrics**: bounce rate, uninstall rate

```python
# Simplified A/B report
def compute_ab_metrics(control_users, treatment_users):
    return {
        'reprocess_rate':  metric_reprocess(treatment) - metric_reprocess(control),
        'subscribe_rate':  metric_subscribe(treatment) - metric_subscribe(control),
        'rating_avg':      rating_avg(treatment) - rating_avg(control),
        'p_value':         ab_significance_test(control, treatment),
    }
```

### Multi-Armed Bandit

A more advanced deployment style: have traffic allocation itself adjust dynamically based on results. Models that perform better automatically receive more traffic.

Engineering implementation: use mature frameworks such as Vowpal Wabbit, Adobe Sensei.

### Engineering caveats for A/B

1. **Long enough duration**: run for at least 7-14 days to avoid single-day fluctuations
2. **Avoid Selection Bias**: assign users randomly, do not let users self-select
3. **Sample Ratio Mismatch**: monitor whether the split is actually 50/50 (infrastructure bugs are common)
4. **Multiple testing**: simultaneously testing several variants requires Bonferroni correction
5. **Segmented analysis**: different devices/regions/user segments may react differently

## 12.10 Failure case bank

A model with good average metrics may completely break in **specific scenarios**. **Specifically maintain a failure case bank**—collect known broken inputs and run every new model against this set.

### How to collect

- From user reports (production)
- Actively constructed (boundary scenarios)
- Failure cases in papers
- Complaints on Twitter/Reddit

Typical failure categories (general for image enhancement):

- Extreme low light
- Extreme high-ISO noise
- Severe motion blur
- Extremely small details (small text, distant faces)
- Multi-face scenes
- Rare objects (animals other than cats and dogs)
- Special viewpoints (fisheye, wide-angle)
- Partial occlusion
- All kinds of stylized imagery (anime, oil painting, pixel art)

### Metrics for the failure case bank

Beyond the average, also look at:

- **Failure rate**: how many images in this set are visually broken?
- **New failure modes**: did it introduce problems that didn't exist before?

Engineering practice: every new model version must be run on the failure bank, results archived. **Long-term maintenance** of this set is far more important than benchmark grinding.

## 12.11 Benchmark design

Chapter 4, Section 4.9 covered the bias of academic benchmarks. Here we discuss how to design **your own** benchmark.

### What the test set should include

At least four sources:

1. **Academic benchmarks** (DIV2K val, Set5/14, etc.): for comparison with published methods
2. **Real degradation data** (RealSR, DRealSR): to test the real distribution
3. **Domain-specific test set** (your business data): to test actual results
4. **Failure case bank**: to test robustness

### Diversity audit

Tag every image in the test set (scene, content, degradation level) to ensure distribution coverage:

| Scene | Share |
|------|------|
| Indoor | 25% |
| Outdoor daylight | 30% |
| Outdoor night | 15% |
| Documents/screenshots | 10% |
| People (including faces) | 15% |
| Other | 5% |

If a category is < 1% in share, failures in that category will not appear in the average metrics.

### The benchmark cannot be optimized

The most important discipline: **your final benchmark must not be used for hyperparameter tuning**.

- Train on the train set
- Tune hyperparameters on the val set
- Final evaluation on the test set

If you tweaked hyperparameters in order to grind the test set, the test set is already contaminated. This is the basic principle that machine learning repeatedly violates.

Engineering practice: **keep one "sealed" test set** to be run only once before final release.

## 12.12 Eval pipeline engineering

In real engineering, eval should be automated.

```python
import json
from pathlib import Path

class EvalPipeline:
    def __init__(self, model, datasets: dict, metrics: dict):
        """
        datasets: {name: dataloader}
        metrics: {name: callable(pred, target) -> float}
        """
        self.model = model
        self.datasets = datasets
        self.metrics = metrics

    @torch.no_grad()
    def run(self, output_dir: Path) -> dict:
        results = {}
        for ds_name, loader in self.datasets.items():
            ds_results = {m: [] for m in self.metrics}
            for batch in loader:
                lr, hr = batch['lr'], batch['hr']
                pred = self.model(lr.cuda())
                for m_name, m_fn in self.metrics.items():
                    score = m_fn(pred, hr.cuda())
                    ds_results[m_name].extend(score.cpu().tolist())
            results[ds_name] = {
                m: {
                    'mean':  np.mean(scores),
                    'std':   np.std(scores),
                    'count': len(scores),
                }
                for m, scores in ds_results.items()
            }

        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        return results
```

Engineering essentials:

- **Reproducible**: fixed random seed, version number, commit ID
- **Persistent**: every eval result is written to disk, kept long-term
- **Comparable**: able to diff eval results across commits
- **Visualized**: automatically generate comparison images and tables

## 12.13 Standards for paper experimental reporting

When writing a paper / technical report, the eval section should include:

1. **Datasets**: list each, including version numbers
2. **Metrics**: PSNR + SSIM + LPIPS + DISTS + at least one NR metric
3. **Comparison methods**: include 5+ public SOTA + a simple baseline
4. **Multiple runs**: run each setup at least 3 times, report mean ± std
5. **Failure cases**: required, otherwise reviewers will request them
6. **Subjective evaluation** (if it is a perception-end method)
7. **Downstream tasks** (if the method targets helping downstream tasks)
8. **Ablation studies**: ablate each design decision

Many papers skip 4, 5, 6, and the result is reviewer skepticism or failed reproductions.

## 12.14 Common evaluation antipatterns

A few to avoid in engineering:

### Antipattern 1: only showing cherry-picked images

"Look how well this old photo is restored!"—what about the failure cases you didn't show?

### Antipattern 2: comparing against outdated methods

Only comparing to ESRGAN (2018), not to SwinIR/HAT/SUPIR.

### Antipattern 3: only reporting PSNR

"Our PSNR is 0.3 dB higher than theirs"—high PSNR does not imply good visual quality.

### Antipattern 4: training data contains the test set

Inadvertently including images similar to the test set in training.

### Antipattern 5: "by feel" hyperparameter selection

Each eval is different, with no traceability to why this number was chosen.

## 12.15 Summary

1. **Three layers of evaluation**: automatic metrics + subjective evaluation + downstream tasks, validating each other
2. **MOS is simple but noisy**, 2AFC's forced comparison has higher SNR, **prefer 2AFC**
3. **Sample size requires power analysis**: 1000+ pairs is the lower bound for a serious experiment
4. **Rater quality control**: catch trial, test-retest, time check
5. **Statistical significance**: t-test/Wilcoxon for continuous metrics, binomial test for 2AFC, Bonferroni for multiple comparisons
6. **Correlation between automatic metrics and MOS**: LPIPS/DISTS/MANIQA ~0.75, PSNR only 0.4-0.55
7. **Downstream task evaluation is the most convincing**: OCR accuracy, face recognition rate, detection mAP
8. **A/B testing is the de facto standard in production**: monitor user behavior metrics
9. **Failure case bank**: maintained long-term, more important than average metrics
10. **Benchmarks must not be used for hyperparameter tuning**: keep a sealed test set

That completes Part III's training and evaluation chapters. Part IV moves into video—video is not just "images plus time"; temporal consistency is an independent problem.

---

> Next chapter [Video is Not Just Images Plus Time](13-video-basics.md) → temporal consistency, optical flow, the specifics of video degradation.
