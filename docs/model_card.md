# Model card — TinyECGCNN HYP v1.3

**Model name:** `TinyECGCNN` HYP v1.3 (noise-augmented)
**Version:** 1.3 (tag `hyp_v1_aug`)
**Date:** 2026-10-21
**Checkpoint:** `outputs/hyp_v1_aug/best_model.pt`
**Code:** `src/ai_ecg_hf/model.py`, `scripts/train_hyp.py`

## Intended use

Research-use-only screening model that estimates the probability of the PTB-XL HYP diagnostic superclass from a 10-second, 100 Hz, single-lead (Lead I) ECG. It is intended for retrospective evaluation and as a foundation for prospective studies of wearable or handheld ECG screening for left ventricular hypertrophy.

## Not intended for

- Clinical diagnosis of hypertrophy or heart failure.
- Replacement of echocardiography or 12-lead ECG interpretation.
- Direct deployment on consumer wearables without external validation on the target device.
- Paediatric or pregnant populations (not represented in the training data).
- Any use in which a probability threshold would trigger treatment without confirmatory testing.

## Input

- One-dimensional Lead I signal, 10 s, resampled to 1,000 samples (100 Hz).
- Per-record z-score normalization; no beat detection or filtering.
- Missing or corrupted segments should be handled upstream; the model has no explicit quality gate.

## Output

- A single logit converted to a probability via the sigmoid function.
- Raw probabilities are not calibrated for clinical use; temperature scaling (T = 0.936) reduces but does not eliminate miscalibration.
- Validation-locked thresholds: Youden 0.3577; sensitivity ≥0.90 threshold 0.1574; specificity ≥0.90 threshold 0.6378.

## Training data

- PTB-XL v1.0.3 `records100`, folds 1–8 (17,418 records; HYP 12.17%).
- Validation: fold 9 (2,183 records). Test: fold 10 (2,198 records).
- Labels: PTB-XL HYP diagnostic superclass (LVH, RVH, left atrial overload/enlargement, septal hypertrophy).
- No external data were used for training or fine-tuning.

## Architecture and training

- Three 1D convolutional blocks (32/64/128 filters; kernels 7/5/5), batch normalization, ReLU, max pooling, global average pooling, linear head.
- 52,225 parameters.
- Adam (1 × 10⁻³), batch 64, BCEWithLogits with positive-class weighting, early stopping on validation AUROC (patience 4), seed 42.
- Wearable-like augmentation: Gaussian σ = 0.05, baseline sine amplitude 0.1, random zeroing p = 0.2.

## Performance (v1.3)

| Cohort | n | HYP, n (%) | AUROC (95% CI) | AUPRC (95% CI) | ECE |
|---|---:|---:|---|---|---:|
| PTB-XL test (fold 10) | 2,198 | 262 (11.92) | 0.780 (0.748–0.814) | 0.415 (0.355–0.486) | 0.206 |
| Chapman-Shaoxing (case-control) | 2,769 | 769 (27.77) | 0.883 (0.867–0.896) | 0.762 (0.730–0.793) | 0.100 |
| Georgia (Emory, natural prevalence) | 10,344 | 1,305 (12.62) | 0.718 (0.703–0.732) | 0.284 (0.261–0.307) | 0.274 |

Noise robustness at 4× training severity: PTB-XL 0.760 vs 0.661 (clean), Chapman 0.868 vs 0.727, Georgia 0.682 vs 0.586.

## Secondary analyses

- Calibration slope (ideal 1.0): PTB-XL 0.81, Chapman 1.21, Georgia 0.55; intercepts −1.69, −0.69, and −1.88.
- Decision curve analysis: net benefit over treat-all only at threshold probabilities ≤0.15 in PTB-XL and Georgia.
- Occlusion sensitivity: small, diffuse probability drops without a single dominant 0.5-s segment.
- Seed variability across seeds 42–44: PTB-XL 0.787 ± 0.007; Chapman 0.885 ± 0.005; Georgia 0.725 ± 0.009 AUROC.

## Subgroups

- LVH-only > RVH/RAH-only in both external cohorts (Chapman 0.905 vs 0.765; Georgia 0.726 vs 0.590).
- Sex and age differences were small and inconsistent across cohorts.
- Race/ethnicity and skin tone were not available in the source datasets; fairness across these axes is unassessed.

## Limitations

- Labels are not echo-confirmed; PTB-XL uses a diagnostic superclass, Chapman uses expert SNOMED codes, and Georgia uses report-derived SNOMED codes.
- Single-lead, 100 Hz input discards spatial information from the other 11 leads.
- Simulated noise may not capture real wearable artifacts (motion, poor contact, device filtering).
- External calibration is poor (ECE 0.10–0.27); recalibration on the target population is required before probability-based decisions.
- No prospective clinical utility or outcome data.
- No explainability analysis (saliency/occlusion) is included in v1.3.

## Ethical and regulatory considerations

- Public de-identified data; no individual consent required.
- Not a medical device; research use only.
- Potential for false positives in low-prevalence populations; confirmatory testing is required.
- Performance may differ across device hardware, skin tone, body habitus, and clinical setting.

## Reproducibility

- Training: `python scripts/train_hyp.py --tag hyp_v1_aug --seed 42`
- Internal/external analysis: `python scripts/analyze_augmented_model.py`, `python scripts/eval_georgia_external.py`
- Noise robustness: `python scripts/eval_noise_robustness.py`
- Random seed: 42; deterministic data splits; bootstrap seeds fixed.
