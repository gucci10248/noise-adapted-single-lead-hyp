"""Bar chart of baseline AUROC by cohort."""

from __future__ import annotations

from pathlib import Path

import ast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "baselines" / "baseline_metrics.csv"
OUT = ROOT / "figures" / "baseline_comparison.png"

LABELS = {
    "age_sex": "Age + sex",
    "feature_logistic": "Feature logistic",
    "feature_trees": "Feature trees",
    "resnet": "1D ResNet",
    "tiny_cnn_aug": "Compact CNN",
}
ORDER = ["age_sex", "feature_logistic", "feature_trees", "resnet", "tiny_cnn_aug"]
COLORS = ["0.65", "C0", "C2", "C3", "C1"]


def main() -> None:
    df = pd.read_csv(SRC)
    cohorts = [("ptbxl_test", "PTB-XL test"), ("chapman", "Chapman"), ("georgia", "Georgia")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, (key, title) in zip(axes, cohorts):
        sub = df[df.cohort == key].set_index("model").loc[ORDER]
        x = np.arange(len(ORDER))
        vals = sub.auroc.to_numpy()
        lo = np.array([ast.literal_eval(c)[0] for c in sub.auroc_ci95])
        hi = np.array([ast.literal_eval(c)[1] for c in sub.auroc_ci95])
        ax.bar(x, vals, color=COLORS, width=0.7, edgecolor="black", linewidth=0.5)
        ax.errorbar(x, vals, yerr=[vals - lo, hi - vals], fmt="none", ecolor="black", capsize=3, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in ORDER], rotation=35, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0.45, 0.95)
        ax.grid(axis="y", alpha=0.3)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    axes[0].set_ylabel("AUROC")
    fig.tight_layout()
    fig.savefig(OUT, dpi=300)
    fig.savefig(OUT.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
