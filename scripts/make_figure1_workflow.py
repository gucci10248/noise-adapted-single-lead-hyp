"""Generate Figure 1: study workflow (schematic, no data)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "figure1_workflow.png"


def box(ax, x, y, w, h, text, fc="#EAF1FB", ec="#2F5597", fs=9, bold=False):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight="bold" if bold else "normal",
        wrap=True,
    )


def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", lw=1.1, color="0.35", shrinkA=2, shrinkB=2),
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Row 1: development
    box(ax, 0.02, 0.76, 0.20, 0.14, "PTB-XL\nn = 21,799\n100 Hz, Lead I", bold=True)
    box(ax, 0.27, 0.76, 0.20, 0.14, "Training\nfolds 1–8\nn = 17,418", bold=True)
    box(ax, 0.52, 0.86, 0.19, 0.10, "Noise-augmented\ntraining", fc="#FCE4D6", ec="#C55A11")
    box(ax, 0.52, 0.72, 0.19, 0.10, "Clean training\n(comparator)", fc="#F2F2F2", ec="#7F7F7F")
    box(ax, 0.76, 0.76, 0.21, 0.14, "Internal test\nfold 10\nn = 2,198", bold=True)

    arrow(ax, 0.22, 0.83, 0.27, 0.83)
    arrow(ax, 0.47, 0.85, 0.52, 0.90)
    arrow(ax, 0.47, 0.81, 0.52, 0.77)
    arrow(ax, 0.71, 0.90, 0.76, 0.85)
    arrow(ax, 0.71, 0.77, 0.76, 0.81)

    # Row 2: frozen model / external cohorts
    box(ax, 0.27, 0.47, 0.20, 0.12, "Frozen Lead-I CNN\n52,225 parameters", fc="#FFF2CC", ec="#BF9000", bold=True)
    arrow(ax, 0.47, 0.76, 0.37, 0.59)

    box(ax, 0.55, 0.55, 0.19, 0.14, "Chapman-Shaoxing\nn = 2,769\n769 HYP (27.8%)", fc="#E2F0D9", ec="#548235")
    box(ax, 0.77, 0.55, 0.20, 0.14, "Georgia (Emory)\nn = 10,344\n1,305 HYP (12.6%)", fc="#E2F0D9", ec="#548235")
    arrow(ax, 0.47, 0.53, 0.55, 0.62)
    arrow(ax, 0.47, 0.51, 0.77, 0.62)

    # Row 3: analyses
    box(
        ax,
        0.10,
        0.12,
        0.80,
        0.20,
        "Evaluation\nAUROC / AUPRC (bootstrap 95% CI)  ·  calibration (ECE, slope, Platt/temperature)\n"
        "subgroups (LVH-only, RVH/RAH-only, sex, age)  ·  noise robustness (0–4× training severity)\n"
        "baselines (age-sex, features, trees, ResNet)  ·  ablation (none/Gaussian/wander/dropout/full)\n"
        "test-time augmentation  ·  external recalibration  ·  PPV / NPV at cohort prevalence",
        fc="#EDEDED",
        ec="#595959",
        fs=8.5,
    )
    arrow(ax, 0.65, 0.55, 0.55, 0.32)
    arrow(ax, 0.86, 0.55, 0.65, 0.32)

    ax.text(0.02, 0.985, "Model development, benchmarking, and external validation", fontsize=12, fontweight="bold")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
