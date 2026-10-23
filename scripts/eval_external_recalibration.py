"""External recalibration experiment on held-out Chapman/Georgia splits.

For each cohort, repeatedly split 20% calibration / 80% evaluation (stratified),
fit Platt scaling or temperature scaling on the calibration split, and evaluate
ECE, calibration slope/intercept, AUROC, and AUPRC on the held-out split.

Usage:
  python scripts/eval_external_recalibration.py --repeats 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_chapman_external import expected_calibration_error  # noqa: E402

OUT_DIR = ROOT / "outputs" / "recalibration"
FIG_DIR = ROOT / "figures"
SEED = 42
COHORTS = {
    "chapman": ROOT / "outputs" / "hyp_v1_aug" / "chapman_predictions.csv",
    "georgia": ROOT / "outputs" / "georgia_ext" / "predictions.csv",
}


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    z = torch.tensor(logits, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    log_t = torch.nn.Parameter(torch.zeros(()))
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=100)
    loss_fn = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = loss_fn(z / torch.exp(log_t), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).detach())


def fit_platt(logits: np.ndarray, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(logits.reshape(-1, 1), y)
    return model


def calibration_slope_intercept(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    eps = 1e-6
    z = np.log(np.clip(scores, eps, 1 - eps) / (1 - np.clip(scores, eps, 1 - eps)))
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(z.reshape(-1, 1), y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def eval_scores(y: np.ndarray, scores: np.ndarray) -> dict:
    slope, intercept = calibration_slope_intercept(y, scores)
    return {
        "auroc": float(roc_auc_score(y, scores)),
        "auprc": float(average_precision_score(y, scores)),
        "ece": float(expected_calibration_error(y, scores)),
        "slope": slope,
        "intercept": intercept,
        "mean_pred": float(np.mean(scores)),
        "obs_rate": float(np.mean(y)),
    }


def reliability_points(y: np.ndarray, scores: np.ndarray, n_bins: int = 10):
    """Equal-count (quantile) bins for a stable reliability curve."""
    order = np.argsort(scores)
    xs, ys = [], []
    for idx in np.array_split(order, n_bins):
        if len(idx) == 0:
            continue
        xs.append(float(np.mean(scores[idx])))
        ys.append(float(np.mean(y[idx])))
    return np.asarray(xs), np.asarray(ys)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    reliability = {}
    for cohort_key, path in COHORTS.items():
        df = pd.read_csv(path)
        y = df["y_hyp"].to_numpy(dtype=int)
        logits = df["logit"].to_numpy(dtype=float)
        print(f"[{cohort_key}] n={len(y)} pos={y.mean():.3f}")

        splitter = StratifiedShuffleSplit(
            n_splits=args.repeats, test_size=0.2, random_state=args.seed
        )
        rows = []
        for rep, (cal_idx, eval_idx) in enumerate(splitter.split(logits, y)):
            z_cal, y_cal = logits[cal_idx], y[cal_idx]
            z_eval, y_eval = logits[eval_idx], y[eval_idx]

            # No recalibration
            rows.append(
                {
                    "cohort": cohort_key,
                    "repeat": rep,
                    "method": "none",
                    **eval_scores(y_eval, sigmoid(z_eval)),
                }
            )
            # Temperature scaling
            T = fit_temperature(z_cal, y_cal)
            rows.append(
                {
                    "cohort": cohort_key,
                    "repeat": rep,
                    "method": "temperature",
                    "T": T,
                    **eval_scores(y_eval, sigmoid(z_eval / T)),
                }
            )
            # Platt scaling
            platt = fit_platt(z_cal, y_cal)
            rows.append(
                {
                    "cohort": cohort_key,
                    "repeat": rep,
                    "method": "platt",
                    "platt_a": float(platt.coef_[0, 0]),
                    "platt_b": float(platt.intercept_[0]),
                    **eval_scores(y_eval, platt.predict_proba(z_eval.reshape(-1, 1))[:, 1]),
                }
            )
            if rep == 0:
                reliability[cohort_key] = {
                    "y": y_eval,
                    "none": sigmoid(z_eval),
                    "temperature": sigmoid(z_eval / T),
                    "platt": platt.predict_proba(z_eval.reshape(-1, 1))[:, 1],
                }
        all_rows.extend(rows)

    raw = pd.DataFrame(all_rows)
    raw.to_csv(OUT_DIR / "recalibration_raw.csv", index=False)

    agg = (
        raw.groupby(["cohort", "method"])
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_sd=("auroc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_sd=("auprc", "std"),
            ece_mean=("ece", "mean"),
            ece_sd=("ece", "std"),
            slope_mean=("slope", "mean"),
            slope_sd=("slope", "std"),
            intercept_mean=("intercept", "mean"),
            intercept_sd=("intercept", "std"),
        )
        .reset_index()
    )
    agg.to_csv(OUT_DIR / "recalibration_summary.csv", index=False)
    (OUT_DIR / "recalibration_metadata.json").write_text(
        json.dumps(
            {
                "repeats": args.repeats,
                "seed": args.seed,
                "calibration_fraction": 0.2,
                "cohorts": {k: str(v) for k, v in COHORTS.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Reliability plot for one held-out split per cohort.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, cohort_key in zip(axes, ["chapman", "georgia"]):
        r = reliability[cohort_key]
        ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=1)
        for method, color, label in [
            ("none", "0.4", "Before"),
            ("temperature", "C0", "Temperature"),
            ("platt", "C3", "Platt"),
        ]:
            xs, ys = reliability_points(r["y"], r[method])
            ax.plot(xs, ys, marker="o", ms=4, lw=1.6, color=color, label=label)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed HYP rate")
        ax.set_title(cohort_key.capitalize())
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "external_recalibration.png", dpi=300)
    fig.savefig(FIG_DIR / "external_recalibration.pdf")
    plt.close(fig)

    print("\n==== recalibration summary ====")
    print(agg.round(3).to_string(index=False))
    print(f"wrote {OUT_DIR / 'recalibration_summary.csv'}")
    print(f"wrote {FIG_DIR / 'external_recalibration.png'}")


if __name__ == "__main__":
    main()
