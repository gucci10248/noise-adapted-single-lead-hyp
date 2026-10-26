# External recalibration correction

26 October 2026. Previous versions, including v1.3.0, assigned the first StratifiedShuffleSplit output to calibration despite test_size=0.2. This used 80% calibration / 20% evaluation, contradicting the documented protocol. The corrected script uses 20% calibration / 80% evaluation, checks disjoint indices and calibration counts, and records actual sample counts per repeat. Primary model weights and discrimination/noise analyses are unchanged.

Ten seed-42 stratified splits were rerun from the frozen predictions. Calibration/evaluation n: Chapman 554/2215; Georgia 2069/8275. Corrected mean ECE: Chapman 0.100→0.029 (Platt); Georgia 0.274→0.006. Mean slopes after Platt: 0.995 and 1.006. Splits overlap and are not independent validation cohorts. Chapman recalibration targets case-control prevalence, not the source-population 1.70% prevalence.

Run scripts/verify_recalibration_split.py after the evaluation to independently reconstruct all splits and verify the Platt coefficients and AUROCs. The verified outputs are in results/recalibration_*.csv. No publication or submission status is inferred from this repository.
