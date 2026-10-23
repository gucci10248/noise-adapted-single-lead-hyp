# CMPB strengthening results (2026-10-23)

Target: **Computer Methods and Programs in Biomedicine** (Elsevier, SCIE, JCR Q1, IF 6.4; non-OA no mandatory APC). CBM was removed from SCIE on 2025-11-17 and is no longer a valid Q1 target.

## 1. Baselines (clean data, PTB-XL train → test/Chapman/Georgia)

| Model | Params | PTB-XL AUROC | Chapman AUROC | Georgia AUROC |
|---|---:|---:|---:|---:|
| Age + sex logistic | 2 | 0.606 | 0.662 | 0.511 |
| Lead-I feature logistic (39 features) | 39 | 0.757 | 0.817 | 0.702 |
| Lead-I feature trees (HistGradientBoosting) | — | 0.763 | 0.833 | 0.716 |
| 1D ResNet | 437,569 | **0.795** | 0.870 | **0.749** |
| TinyECGCNN (primary, full noise aug) | 52,225 | 0.780 | **0.883** | 0.718 |

AUPRC (same order): age-sex 0.150/0.391/0.125; feature logistic 0.370/0.644/0.273; trees 0.374/0.688/0.304; ResNet 0.433/0.728/0.330; TinyCNN 0.415/0.762/0.284.

Interpretation: ResNet is higher on PTB-XL and Georgia (+0.015/+0.031); TinyCNN is higher on Chapman (+0.013). The compact model is an efficiency/robustness trade-off, not a clean-accuracy winner.

## 2. Augmentation ablation (AUROC at 0× / 4×)

| Training augmentation | PTB-XL 0× | PTB-XL 4× | Chapman 0× | Chapman 4× | Georgia 0× | Georgia 4× |
|---|---:|---:|---:|---:|---:|---:|
| None | 0.787 | 0.661 | 0.885 | 0.727 | 0.711 | 0.586 |
| Gaussian only | 0.787 | 0.664 | 0.885 | 0.733 | 0.713 | 0.590 |
| Wander only | 0.782 | **0.756** | 0.874 | 0.850 | **0.722** | **0.685** |
| Dropout only | 0.785 | 0.743 | **0.893** | 0.860 | 0.719 | 0.643 |
| Full (primary) | 0.780 | **0.760** | 0.883 | **0.868** | 0.718 | 0.682 |

Key finding: **baseline wander is the dominant component**; Gaussian-only is almost inert; dropout helps clean Chapman and intermediate noise; full augmentation gives the best balance on PTB-XL/Chapman. Wander-only recovered 96%, 87%, and 103% of the full gain over no augmentation.

## 3. Test-time augmentation (TTA)

TTA averages 5 additional corruption draws at 1× or 4× training severity. It did **not** improve discrimination:

| Cohort / test severity | Standard | TTA 1× | TTA 4× |
|---|---:|---:|---:|
| PTB-XL 4× | 0.760 | 0.754 | 0.745 |
| Chapman 4× | 0.868 | 0.859 | 0.848 |
| Georgia 4× | 0.682 | 0.676 | 0.661 |

Georgia ECE at 4× improved slightly (0.243 → 0.236 with TTA 4×), but AUROC fell. TTA is a null result and should be reported as such, not sold as a contribution.

## 4. External recalibration (20% calibration / 80% evaluation; 10 splits)

| Cohort | Method | ECE | Slope | Intercept | AUROC |
|---|---|---:|---:|---:|---:|
| Chapman | None | 0.101 | 1.226 | −0.676 | 0.886 |
| Chapman | Temperature | 0.083 | 0.935 | −0.676 | 0.886 |
| Chapman | Platt | **0.035** | **1.016** | **0.024** | 0.886 |
| Georgia | None | 0.276 | 0.546 | −1.884 | 0.718 |
| Georgia | Temperature | 0.302 | 0.883 | −1.884 | 0.718 |
| Georgia | Platt | **0.011** | **1.001** | **−0.007** | 0.718 |

Key finding: the miscalibration is a **shift + scale** problem, not a simple temperature problem. A small labeled external sample (20%) plus Platt scaling restores calibration without changing AUROC.

## 5. ResNet noise robustness

| Cohort | 0× | 1× | 4× |
|---|---:|---:|---:|
| PTB-XL | 0.795 | 0.790 | 0.781 |
| Chapman | 0.870 | 0.867 | 0.869 |
| Georgia | 0.749 | 0.746 | 0.717 |

The ResNet is more robust than the compact model on PTB-XL/Georgia; the compact model's rationale is size (8.4× fewer parameters), not superior accuracy or robustness.

## 6. Manuscript implications

- Keep TinyECGCNN as the primary model but frame it as a **compact, edge-oriented** choice, not a clean-accuracy winner.
- Add a baseline table and an honest discussion of the ResNet trade-off.
- Make **augmentation ablation** and **external recalibration** the main new contributions.
- Report TTA as a negative result; do not overclaim.
- Title: "Noise-robust single-lead AI-ECG for hypertrophy screening: augmentation ablation, external validation, and recalibration across label sources."
- Target: CMPB (Q1), fallback Scientific Reports (Q1/Q2, APC).
