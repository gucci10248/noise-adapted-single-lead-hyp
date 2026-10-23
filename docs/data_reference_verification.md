# Data and reference verification report

**Date:** 2026-10-22  
**Manuscript:** Noise-adapted single-lead AI-ECG for hypertrophy screening: development and external validation in PTB-XL, Chapman-Shaoxing, and Georgia  
**Scope:** all 21 references, three data sources, cohort counts, label definitions, and the numbers used in the manuscript.

## 1. Reference verification

All 21 references were checked against Crossref (journal articles) or DataCite (PhysioNet resources). Every DOI resolves to the title cited in the manuscript.

| # | First author, year | Journal | Volume:pages | DOI | Status |
|---:|---|---|---|---|---|
| 1 | Siontis, 2021 | Nat Rev Cardiol | 18:465–478 | 10.1038/s41569-020-00503-2 | Verified |
| 2 | Attia, 2019 | Nat Med | 25:70–74 | 10.1038/s41591-018-0240-2 | Verified |
| 3 | Yao, 2021 | Nat Med | 27:815–819 | 10.1038/s41591-021-01335-4 | Verified |
| 4 | Haimovich, 2023 | Cardiovasc Digit Health J | 4:48–59 | 10.1016/j.cvdhj.2023.03.001 | Verified |
| 5 | Liu, 2022 | Circ Cardiovasc Qual Outcomes | 15:e008360 | 10.1161/CIRCOUTCOMES.121.008360 | Verified |
| 6 | Zhao, 2022 | Front Cardiovasc Med | 9:952089 | 10.3389/fcvm.2022.952089 | Verified |
| 7 | Siranart, 2024 | Sci Rep | 14:15882 | 10.1038/s41598-024-66247-y | Verified |
| 8 | Sato, 2023 | Circ J | 88:146–156 | 10.1253/circj.CJ-23-0216 | Verified |
| 9 | Aminorroaya, 2025 | Eur Heart J Digit Health | 6:554–566 | 10.1093/ehjdh/ztaf034 | Verified |
| 10 | Croon, 2025 | Circulation | 152:1282–1294 | 10.1161/CIRCULATIONAHA.125.076279 | Verified |
| 11 | Collins, 2024 | BMJ | 385:e078378 | 10.1136/bmj-2023-078378 | Verified |
| 12 | Wagner, 2020 | Sci Data | 7:154 | 10.1038/s41597-020-0495-6 | Verified |
| 13 | Zheng, 2020 | Sci Data | 7:48 | 10.1038/s41597-020-0386-x | Verified |
| 14 | Reyna, 2020 | Computing in Cardiology | 47 | 10.22489/CinC.2020.236 | Verified |
| 15 | Hannun, 2019 | Nat Med | 25:65–69 | 10.1038/s41591-018-0268-3 | Verified |
| 16 | Topol, 2019 | Nat Med | 25:44–56 | 10.1038/s41591-018-0300-7 | Verified |
| 17 | Levy, 1990 | N Engl J Med | 322:1561–1566 | 10.1056/NEJM199005313222203 | Verified |
| 18 | Pewsner, 2007 | BMJ | 335:711 | 10.1136/bmj.39276.636354.ae | Verified |
| 19 | Zheng, 2022 | PhysioNet | version 1.0.0 | 10.13026/wgex-er52 | Verified (DataCite) |
| 20 | Van Calster, 2019 | BMC Med | 17:230 | 10.1186/s12916-019-1466-7 | Verified |
| 21 | Riley, 2020 | BMJ | 368:m441 | 10.1136/bmj.m441 | Verified |

### Reference corrections made

- Added DOIs to references 1–16 and 20–21 (only 14 and 17–18 had DOIs before).
- Added reference 19, the PhysioNet `ecg-arrhythmia` 1.0.0 release. The 45,152-record release is from 2022 and is not the same object as the 2020 Sci Data paper (ref 13), which describes the underlying database. Both are now cited in Methods 2.2.
- Confirmed the correct DOIs for refs 1–3 (initial candidate DOIs were wrong; the final DOIs resolve to the cited titles).

## 2. Data source verification

| Dataset | Version / access | Records | Label definition | Local check | Status |
|---|---|---|---:|---|---|
| PTB-XL | v1.0.3, PhysioNet, CC BY 4.0 | 21,799 (18,869 patients) | HYP diagnostic superclass | 21,799 headers; 18,869 patients | Verified |
| Chapman-Shaoxing | PhysioNet `ecg-arrhythmia` 1.0.0, CC BY 4.0 | 45,152 | SNOMED LVH 164873001 + RVH 89792004 + RAH 446358003 | 45,152 headers; 769 HYP | Verified |
| Georgia | Kaggle `physionet/georgia-12lead-ecg-challenge-database`, CC0 | 10,344 | same SNOMED alignment | 10,344 headers; 1,305 HYP | Verified |

### PTB-XL

- Splits: 17,418 training (folds 1–8), 2,183 validation (fold 9), 2,198 test (fold 10).
- HYP events: 2,119 / 268 / 262; prevalence 12.17% / 12.28% / 11.92%.
- Age median (IQR): 61.0 (50.0–72.0), 63.0 (50.0–73.0), 63.0 (50.0–75.0).
- Male: 52.2% / 51.9% / 51.5%.
- HYP superclass contents (from `scp_statements.csv`): LVH, LAO/LAE, RVH, RAO/RAE, SEHYP.
- Note: the 2020 Sci Data paper abstract reports 21,837 records from 18,885 patients; the v1.0.3 PhysioNet page and the local `records100` metadata contain 21,799 records from 18,869 patients. The manuscript uses the v1.0.3 release numbers.

### Chapman-Shaoxing

- 45,152 headers; 3,234 records with local `.mat`; 3,227 readable after excluding 7 unreadable files.
- HYP positives: 769 (1.70% of 45,152).
  - LVH 647; RVH 110; RAH 36.
  - Overlaps: LVH+RVH 6; LVH+RAH 5; RVH+RAH 16; all three 3.
  - RVH/RAH-only (no LVH): 122.
- Negative library: 45,152 − 769 = **44,383**.
- Case-control sample: 769 positives + 2,000 negatives sampled from 2,458 readable negatives (seed 42) = 2,769.
- Age: 7 missing and 12 recorded as 0 in the 2,769 sample; both were treated as missing for the age-sex baseline.

### Georgia

- 10,344 records; 1,305 HYP (12.62%).
  - LVH 1,232; RVH 86; RAH 0; LVH+RVH overlap 13.
- Age missing in 77 records (0.7%); no age-0 records; 3 records under 18.
- Signals: 500 Hz, 12 × 5,000 int16, gain 4,880/mV.
- The Kaggle mirror is a redistribution of the PhysioNet/Computing in Cardiology Challenge 2020 Georgia subset; no data are redistributed in the code repository.

## 3. Data-related corrections made to the manuscript

1. **Negative library count:** 43,383 → **44,383** (45,152 − 769). Corrected in Methods 2.2, Limitations 4.7, the response template, and the results memo.
2. **PTB-XL HYP definition:** added **right atrial overload/enlargement (RAO/RAE)**. The superclass is LVH, RVH, LAO/LAE, RAO/RAE, and SEHYP.
3. **External label scope:** added a sentence stating that the external cohorts provide LVH/RVH/RAH but not atrial overload/enlargement or septal hypertrophy codes, so the external label is a subset of the PTB-XL HYP superclass.
4. **Subgroup wording:** Chapman “RVH-only” → **RVH/RAH-only** (122), with Georgia RVH-only (73) distinguished.
5. **Age handling:** documented 77 Georgia missing ages, 7 Chapman missing ages, and 12 Chapman age-0 records imputed as 60 for the baseline only.
6. **Chapman release citation:** added ref 19 (PhysioNet 1.0.0, 2022) and cited it alongside the 2020 Sci Data paper.

## 4. Reproducibility checks

- `results/results_table.csv` reproduces the manuscript’s AUROC/AUPRC/ECE values.
- `outputs/seed_variability/seed_summary.csv` reproduces the 3-seed means and SDs.
- `outputs/noise_robustness/table4_noise_robustness.csv` reproduces the 4× noise values.
- `outputs/calibration/calibration_metrics.csv` reproduces the calibration slopes/intercepts.
- `outputs/dca/summary.json` reproduces the DCA net-benefit values.
- The Chapman case-control sample is reproducible from `data_lists/chapman_records.txt` plus seed 42.

## 5. Remaining caveats (already stated in the manuscript)

- HYP labels are not echocardiography-confirmed.
- Chapman is case-control; AUPRC/PPV are inflated and PPV/NPV are projected to 1.70% prevalence.
- Georgia labels are report-derived.
- MIMIC-IV-ECG was unavailable in this region.
- No data are redistributed; the code repository contains public record IDs only.

## 6. Verdict

The reference list is traceable by DOI, the three data sources match their official releases, and the cohort counts and label definitions are reproducible from the local data. The data-related errors found in this check (44,383; RAO/RAE; RVH/RAH wording; age handling; Chapman release citation) have been corrected.
