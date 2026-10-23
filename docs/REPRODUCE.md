# Reproducing the analysis

The pipeline was run on an Apple M3 with MPS acceleration, Python 3.12, and the
versions in `requirements.txt`. Training one model takes roughly 10–20 minutes;
the full pipeline (including two extra seeds and external validation) takes a
few hours.

## 0. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_env.py
```

## 1. Data

Follow `docs/DATA.md`. Expected downloads:

```bash
python scripts/download_ptbxl.py
python scripts/download_chapman.py --hea-only --all --workers 32
python scripts/download_chapman.py --from-list data_lists/chapman_records.txt --workers 32
python scripts/download_georgia.py
```

## 2. Train the models

```bash
# Clean comparator (v1)
python scripts/train_hyp.py --no-aug --tag hyp_v1 --seed 42

# Noise-augmented primary model (v1.3)
python scripts/train_hyp.py --tag hyp_v1_aug --seed 42

# Two additional seeds for variability
python scripts/train_hyp.py --tag hyp_v1_aug_seed43 --seed 43
python scripts/train_hyp.py --tag hyp_v1_aug_seed44 --seed 44
```

Each run writes `outputs/<tag>/best_model.pt`, `metrics.json`, and a ROC figure.

## 3. Internal validation and calibration

```bash
python scripts/eval_hyp_v1_metrics.py          # operating points, v1.1
python scripts/calibrate_hyp_temperature.py    # temperature scaling, v1.2
```

These scripts use `outputs/hyp_v1` by default.

## 4. Chapman-Shaoxing external validation

```bash
python scripts/analyze_chapman_external.py     # clean v1 package
python scripts/analyze_augmented_model.py      # v1.3 package
```

`analyze_chapman_external.py` writes `outputs/chapman_ext/` (predictions,
external_analysis.json, tables, figures). `analyze_augmented_model.py` uses
`outputs/hyp_v1_aug` and also reads the clean Chapman results; run the clean
script first.

## 5. Georgia external validation

```bash
python scripts/eval_georgia_external.py
python scripts/eval_georgia_external.py \
  --ckpt outputs/hyp_v1/best_model.pt \
  --tag hyp_v1 \
  --out-dir outputs/georgia_ext_clean \
  --fig-prefix georgia_ext_clean
```

The first command writes `outputs/georgia_ext/` (v1.3); the second writes
`outputs/georgia_ext_clean/` (clean comparator). The clean predictions are
needed for the decision curve analysis.

## 6. Secondary analyses

```bash
python scripts/eval_noise_robustness.py          # 0–4× noise, 3 draws, 3 cohorts
python scripts/decision_curve_analysis.py        # DCA, PTB-XL + Georgia
python scripts/explainability_occlusion.py       # 0.5-s occlusion, PTB-XL + Georgia
python scripts/eval_seed_variability.py          # seeds 42–44
python scripts/plot_seed_variability.py
python scripts/collect_results.py                # combined results table
```

## 7. Expected outputs

| Output | Content |
|---|---|
| `outputs/hyp_v1_aug/metrics.json` | Internal test AUROC/AUPRC, history |
| `outputs/hyp_v1_aug/metrics_v1_1.json` | Thresholds, operating points, ECE |
| `outputs/hyp_v1_aug/external_metrics.json` | Chapman external validation (v1.3) |
| `outputs/chapman_ext/external_analysis.json` | Chapman full package (clean) |
| `outputs/georgia_ext/external_metrics.json` | Georgia external validation (v1.3) |
| `outputs/georgia_ext_clean/external_metrics.json` | Georgia external validation (clean) |
| `outputs/noise_robustness/noise_metrics.json` | Noise robustness |
| `outputs/dca/summary.json` | Net benefit at selected thresholds |
| `outputs/explainability/occlusion_summary.json` | Occlusion profiles |
| `outputs/seed_variability/seed_summary.csv` | Seed-to-seed AUROC/AUPRC |

## Notes

- Random seed: 42 unless otherwise stated.
- PTB-XL splits are patient-wise official `strat_fold` folds.
- Thresholds and temperature scaling are locked on validation fold 9 and applied
  unchanged to test/external cohorts.
- No external data are used for training or fine-tuning.
- The code is research-use-only; it is not a medical device.

## 8. CMPB strengthening analyses

```bash
# Baselines
python scripts/train_baseline_resnet.py --tag hyp_resnet --seed 42
python scripts/eval_baselines.py --n-boot 500

# Augmentation-component ablation
python scripts/train_aug_ablation.py
python scripts/eval_aug_ablation.py

# Test-time augmentation
python scripts/eval_tta.py --n-tta 5

# External recalibration
python scripts/eval_external_recalibration.py --repeats 10

# ResNet noise robustness
python scripts/eval_resnet_noise.py
```

Expected outputs:

| Output | Content |
|---|---|
| `outputs/baselines/baseline_metrics.csv` | Age-sex, feature logistic, feature trees, ResNet, compact CNN |
| `outputs/aug_ablation/ablation_metrics.csv` | None/Gaussian/wander/dropout/full at 0–4× |
| `outputs/tta/tta_metrics.csv` | Standard vs TTA-1× vs TTA-4× |
| `outputs/recalibration/recalibration_summary.csv` | Platt/temperature recalibration |
| `outputs/resnet_noise/resnet_noise.csv` | ResNet noise robustness |
