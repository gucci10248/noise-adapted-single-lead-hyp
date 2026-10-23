"""Test-time augmentation (TTA) for the noise-augmented single-lead model.

At inference, average predictions over K additional wearable-like corruption
draws.  TTA severity is fixed at 1x or 4x the training corruption; test
severity is applied first to the clean signal.

Usage:
  python scripts/eval_tta.py --n-tta 5
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
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_chapman_external import (  # noqa: E402
    expected_calibration_error,
    load_chapman_case_control,
    load_ptbxl_test,
)
from ai_ecg_hf.dataset import add_wearable_like_noise  # noqa: E402
from eval_georgia_external import load_georgia  # noqa: E402
from eval_noise_robustness import (  # noqa: E402
    BASE_DROPOUT,
    BASE_SIGMA,
    BASE_WANDER,
    corrupt,
    load_model,
    predict_logits,
)

OUT_DIR = ROOT / "outputs" / "tta"
FIG_DIR = ROOT / "figures"
TEST_SEVERITIES = [0.0, 0.5, 1.0, 2.0, 4.0]
TTA_SEVERITIES = {"standard": 0.0, "tta_1x": 1.0, "tta_4x": 4.0}
SEED = 42
N_TEST_DRAWS = 3
N_TTA = 5


def tta_scores(model, x, severity: float, n_tta: int, device, rng) -> np.ndarray:
    if severity == 0.0:
        return 1.0 / (1.0 + np.exp(-predict_logits(model, x, device)))
    probs = []
    for _ in range(n_tta):
        x_aug = x.copy()
        for i in range(len(x_aug)):
            x_aug[i] = add_wearable_like_noise(
                x_aug[i],
                rng,
                sigma=BASE_SIGMA * severity,
                wander_amp=BASE_WANDER * severity,
                dropout_p=min(BASE_DROPOUT * severity, 0.9),
            )
        probs.append(1.0 / (1.0 + np.exp(-predict_logits(model, x_aug, device))))
    return np.mean(probs, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-tta", type=int, default=N_TTA)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_model(ROOT / "outputs" / "hyp_v1_aug" / "best_model.pt", device)
    print(f"device={device} n_tta={args.n_tta}")

    cohorts = {
        "ptbxl": load_ptbxl_test(seed=args.seed),
        "chapman": load_chapman_case_control(seed=args.seed),
        "georgia": load_georgia(),
    }

    rows = []
    for cohort_key, c in cohorts.items():
        x, y = c["x"], c["y"]
        for si, s_test in enumerate(TEST_SEVERITIES):
            # Pre-generate the test-condition draws so every TTA method sees
            # exactly the same observed signals.
            test_draws = []
            for draw in range(N_TEST_DRAWS if s_test > 0 else 1):
                rng = np.random.default_rng(args.seed + 1000 * si + draw)
                test_draws.append(corrupt(x, s_test, rng))
            for method, s_tta in TTA_SEVERITIES.items():
                aurocs, auprcs, eces = [], [], []
                for di, x_test in enumerate(test_draws):
                    rng = np.random.default_rng(args.seed + 5000 * si + 100 * di + 7)
                    scores = tta_scores(model, x_test, s_tta, args.n_tta, device, rng)
                    aurocs.append(roc_auc_score(y, scores))
                    auprcs.append(average_precision_score(y, scores))
                    eces.append(expected_calibration_error(y, scores))
                rows.append(
                    {
                        "cohort": cohort_key,
                        "test_severity": s_test,
                        "method": method,
                        "auroc_mean": float(np.mean(aurocs)),
                        "auroc_sd": float(np.std(aurocs)),
                        "auprc_mean": float(np.mean(auprcs)),
                        "auprc_sd": float(np.std(auprcs)),
                        "ece_mean": float(np.mean(eces)),
                        "ece_sd": float(np.std(eces)),
                        "n_test_draws": len(test_draws),
                    }
                )
                print(
                    f"[{cohort_key}] test={s_test:>3} {method:8s}: "
                    f"AUROC={np.mean(aurocs):.4f} ECE={np.mean(eces):.4f}"
                )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "tta_metrics.csv", index=False)
    meta = {
        "test_severities": TEST_SEVERITIES,
        "tta_severities": TTA_SEVERITIES,
        "n_tta": args.n_tta,
        "n_test_draws": N_TEST_DRAWS,
        "noise": {
            "base_sigma": BASE_SIGMA,
            "base_wander_amp": BASE_WANDER,
            "base_dropout_p": BASE_DROPOUT,
        },
        "checkpoint": str(ROOT / "outputs" / "hyp_v1_aug" / "best_model.pt"),
    }
    (OUT_DIR / "tta_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Figure: AUROC and ECE.
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.2))
    specs = [
        ("ptbxl", "PTB-XL test (fold 10)"),
        ("chapman", "Chapman-Shaoxing"),
        ("georgia", "Georgia (Emory)"),
    ]
    styles = {
        "standard": ("Standard", "0.4", "o", "-"),
        "tta_1x": ("TTA 1×", "C0", "s", "--"),
        "tta_4x": ("TTA 4×", "C3", "^", "-."),
    }
    for col, (cohort_key, title) in enumerate(specs):
        for row, metric in enumerate(["auroc_mean", "ece_mean"]):
            ax = axes[row, col]
            for method, (label, color, marker, ls) in styles.items():
                sub = df[(df.cohort == cohort_key) & (df.method == method)].sort_values(
                    "test_severity"
                )
                ax.plot(
                    sub.test_severity,
                    sub[metric],
                    marker=marker,
                    ms=5,
                    lw=1.8,
                    ls=ls,
                    color=color,
                    label=label,
                )
            ax.set_xlabel("Test noise severity (× training augmentation)")
            ax.set_ylabel("AUROC" if metric == "auroc_mean" else "ECE")
            ax.set_title(f"{title} — {'AUROC' if metric == 'auroc_mean' else 'calibration'}")
            ax.grid(alpha=0.3)
            if row == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "test_time_augmentation.png", dpi=300)
    fig.savefig(FIG_DIR / "test_time_augmentation.pdf")
    plt.close(fig)

    print("\n==== AUROC ====")
    print(
        df.pivot_table(
            index=["cohort", "test_severity"],
            columns="method",
            values="auroc_mean",
        ).round(4).to_string()
    )
    print(f"wrote {OUT_DIR / 'tta_metrics.csv'}")
    print(f"wrote {FIG_DIR / 'test_time_augmentation.png'}")


if __name__ == "__main__":
    main()
