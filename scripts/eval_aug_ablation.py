"""Augmentation-component ablation across PTB-XL, Chapman, and Georgia.

Models: clean, Gaussian-only, wander-only, dropout-only, full augmentation.
Evaluation: 0x, 1x, and 4x training corruption, three noise draws.

Usage:
  python scripts/eval_aug_ablation.py
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
    load_chapman_case_control,
    load_ptbxl_test,
)
from eval_georgia_external import load_georgia  # noqa: E402
from eval_noise_robustness import (  # noqa: E402
    BASE_DROPOUT,
    BASE_SIGMA,
    BASE_WANDER,
    corrupt,
    load_model,
    predict_logits,
)

OUT_DIR = ROOT / "outputs" / "aug_ablation"
FIG_DIR = ROOT / "figures"
SEVERITIES = [0.0, 0.5, 1.0, 2.0, 4.0]
N_DRAWS = 3
SEED = 42
CKPTS = {
    "none": ROOT / "outputs" / "hyp_v1" / "best_model.pt",
    "gaussian": ROOT / "outputs" / "hyp_abl_gaussian" / "best_model.pt",
    "wander": ROOT / "outputs" / "hyp_abl_wander" / "best_model.pt",
    "dropout": ROOT / "outputs" / "hyp_abl_dropout" / "best_model.pt",
    "full": ROOT / "outputs" / "hyp_v1_aug" / "best_model.pt",
}
LABELS = {
    "none": "No augmentation",
    "gaussian": "Gaussian only",
    "wander": "Wander only",
    "dropout": "Dropout only",
    "full": "Full augmentation",
}
COLORS = {
    "none": "0.55",
    "gaussian": "C0",
    "wander": "C2",
    "dropout": "C3",
    "full": "C1",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-draws", type=int, default=N_DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}")

    ch = load_chapman_case_control(seed=args.seed)
    pt = load_ptbxl_test(seed=args.seed)
    ga = load_georgia()
    cohorts = {
        "ptbxl": (pt["x"], pt["y"]),
        "chapman": (ch["x"], ch["y"]),
        "georgia": (ga["x"], ga["y"]),
    }

    rows = []
    cache = {}
    for model_key, ckpt in CKPTS.items():
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint {ckpt}")
        print(f"loading {model_key}: {ckpt}")
        model = load_model(ckpt, device)
        for cohort_key, (x, y) in cohorts.items():
            for si, severity in enumerate(SEVERITIES):
                aurocs, auprcs = [], []
                for draw in range(args.n_draws if severity > 0 else 1):
                    rng = np.random.default_rng(args.seed + 1000 * si + draw)
                    x_eval = corrupt(x, severity, rng)
                    scores = 1.0 / (1.0 + np.exp(-predict_logits(model, x_eval, device)))
                    aurocs.append(roc_auc_score(y, scores))
                    auprcs.append(average_precision_score(y, scores))
                rows.append(
                    {
                        "model": model_key,
                        "cohort": cohort_key,
                        "severity": severity,
                        "auroc_mean": float(np.mean(aurocs)),
                        "auroc_sd": float(np.std(aurocs)),
                        "auprc_mean": float(np.mean(auprcs)),
                        "auprc_sd": float(np.std(auprcs)),
                        "n_draws": len(aurocs),
                    }
                )
                print(
                    f"  {model_key:8s} {cohort_key:8s} sev={severity:>3}: "
                    f"AUROC={np.mean(aurocs):.4f}±{np.std(aurocs):.4f}"
                )
        del model

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "ablation_metrics.csv", index=False)
    meta = {
        "severities": SEVERITIES,
        "n_draws": args.n_draws,
        "noise": {
            "base_sigma": BASE_SIGMA,
            "base_wander_amp": BASE_WANDER,
            "base_dropout_p": BASE_DROPOUT,
        },
        "checkpoints": {k: str(v) for k, v in CKPTS.items()},
    }
    (OUT_DIR / "ablation_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Figure: one panel per cohort.
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    specs = [
        ("ptbxl", "PTB-XL test (fold 10)"),
        ("chapman", "Chapman-Shaoxing"),
        ("georgia", "Georgia (Emory)"),
    ]
    for ax, (cohort_key, title) in zip(axes, specs):
        for model_key in CKPTS:
            sub = df[(df.model == model_key) & (df.cohort == cohort_key)].sort_values("severity")
            ax.errorbar(
                sub.severity,
                sub.auroc_mean,
                yerr=sub.auroc_sd,
                marker="o",
                ms=4,
                lw=1.6,
                capsize=2,
                color=COLORS[model_key],
                label=LABELS[model_key],
            )
        ax.axhline(0.5, ls=":", color="0.6", lw=1)
        ax.set_xlabel("Noise severity (× training augmentation)")
        ax.set_ylabel("AUROC")
        ax.set_title(title)
        ax.set_ylim(0.45, 1.0)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "augmentation_ablation.png", dpi=300)
    fig.savefig(FIG_DIR / "augmentation_ablation.pdf")
    plt.close(fig)

    print("\n==== AUROC at 4x ====")
    print(df[df.severity == 4.0].pivot(index="model", columns="cohort", values="auroc_mean").round(4).to_string())
    print(f"wrote {OUT_DIR / 'ablation_metrics.csv'}")
    print(f"wrote {FIG_DIR / 'augmentation_ablation.png'}")


if __name__ == "__main__":
    main()
