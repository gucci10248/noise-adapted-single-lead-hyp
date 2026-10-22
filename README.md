# Noise-adapted single-lead AI-ECG for hypertrophy screening

Code and analysis for the manuscript:

> **Noise-adapted single-lead AI-ECG for hypertrophy screening: development and external validation in PTB-XL, Chapman-Shaoxing, and Georgia**

The model is a compact one-dimensional CNN (52,225 parameters) that estimates the probability of the PTB-XL HYP diagnostic superclass from a 10-s, 100-Hz, single-lead (Lead I) ECG. Training uses wearable-like noise augmentation (Gaussian noise, baseline wander, random dropout). The model is evaluated internally on PTB-XL fold 10 and externally on Chapman-Shaoxing and the Georgia 12-Lead ECG Challenge Database.

**Status:** research use only; not a medical device. The manuscript is under review.

## Main results

| Cohort | n | HYP, n (%) | AUROC (95% CI) | AUPRC (95% CI) | ECE |
|---|---:|---:|---|---|---:|
| PTB-XL test (fold 10) | 2,198 | 262 (11.92) | 0.780 (0.748–0.814) | 0.415 (0.355–0.486) | 0.206 |
| Chapman-Shaoxing (case-control) | 2,769 | 769 (27.77) | 0.883 (0.867–0.896) | 0.762 (0.730–0.793) | 0.100 |
| Georgia (Emory, natural prevalence) | 10,344 | 1,305 (12.62) | 0.718 (0.703–0.732) | 0.284 (0.261–0.307) | 0.274 |

Across three training seeds (42–44), AUROC was 0.787 ± 0.007 (PTB-XL), 0.885 ± 0.005 (Chapman), and 0.725 ± 0.009 (Georgia). At 4× training noise severity, AUROC was 0.760 vs 0.661 (PTB-XL, noise-augmented vs clean), 0.868 vs 0.727 (Chapman), and 0.682 vs 0.586 (Georgia).

## Data

No data are redistributed in this repository. All three datasets are publicly available:

| Dataset | Source | License |
|---|---|---|
| PTB-XL v1.0.3 | https://physionet.org/content/ptb-xl/1.0.3/ | CC BY 4.0 |
| Chapman-Shaoxing (`ecg-arrhythmia` 1.0.0) | https://physionet.org/content/ecg-arrhythmia/1.0.0/ | CC BY 4.0 |
| Georgia 12-Lead ECG Challenge Database | https://www.kaggle.com/datasets/physionet/georgia-12lead-ecg-challenge-database | CC0 |

See [`docs/DATA.md`](docs/DATA.md) for the expected directory layout and download commands.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_ptbxl.py
python scripts/train_hyp.py --tag hyp_v1_aug --seed 42
python scripts/analyze_augmented_model.py
```

For the full pipeline, including external validation, noise robustness, DCA, occlusion, and seed variability, see [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Repository structure

```
src/ai_ecg_hf/        # data loaders, model, preprocessing
scripts/              # training, internal/external evaluation, analysis
results/              # summary metrics (CSV/JSON) used in the manuscript
figures/              # main and supplementary figures
docs/                 # data access, reproduction, model card
requirements.txt
pyproject.toml
```

## Key scripts

| Script | Purpose |
|---|---|
| `scripts/train_hyp.py` | Train the CNN on PTB-XL (folds 1–8/9/10); `--no-aug` for the clean comparator |
| `scripts/eval_hyp_v1_metrics.py` | Internal test operating points and calibration |
| `scripts/calibrate_hyp_temperature.py` | Temperature scaling on validation fold 9 |
| `scripts/analyze_augmented_model.py` | Internal + Chapman analysis for the noise-augmented model |
| `scripts/eval_chapman_external.py` | Minimal Chapman external validation (AUROC/AUPRC) |
| `scripts/analyze_chapman_external.py` | Full Chapman package (ROC/PR, calibration, subgroups, baseline) |
| `scripts/eval_georgia_external.py` | Georgia external validation |
| `scripts/eval_noise_robustness.py` | Noise robustness across 0–4× severity |
| `scripts/decision_curve_analysis.py` | Decision curve analysis (PTB-XL, Georgia) |
| `scripts/explainability_occlusion.py` | Occlusion sensitivity analysis |
| `scripts/eval_seed_variability.py` | Seed-to-seed variability (42–44) |

## Model card

See [`docs/model_card.md`](docs/model_card.md) for intended use, training data, performance, limitations, and ethical considerations.

## Citation

If you use this code, please cite the manuscript (under review):

```bibtex
@article{guo2026noiseadapted,
  title   = {Noise-adapted single-lead AI-ECG for hypertrophy screening: development and external validation in PTB-XL, Chapman-Shaoxing, and Georgia},
  author  = {Guo, Jinkai and Chen, Hua and Wang, Yu and Song, Xiaopeng and Chang, Hao and Lv, Yueqi and Li, Mengfan and Chang, Jiaxin and Yang, Chongdan and Zhou, Rongfang and Zhang, Shujie},
  journal = {Journal of Electrocardiology},
  year    = {2026},
  note    = {Under review}
}
```

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). The source datasets remain under their original licenses; no data are redistributed.

## Contact

Hua Chen, MD, PhD  
Department of Cardiology, Inner Mongolia Autonomous Region People's Hospital  
No.20 Zhaowuda Road, Xincheng District, Hohhot, Inner Mongolia Autonomous Region, P.R. China 010017  
Email: Zxcv8521@163.com
