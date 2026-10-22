"""Plot seed-to-seed AUROC variability."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "seed_variability"
FIG = ROOT / "figures"


def main() -> None:
    df = pd.read_csv(OUT / "seed_metrics.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    cohorts = list(df["cohort"].unique())
    offsets = {c: i for i, c in enumerate(cohorts)}
    colors = {c: f"C{i}" for i, c in enumerate(cohorts)}
    for cohort in cohorts:
        sub = df[df["cohort"] == cohort].sort_values("seed")
        mean = sub["auroc"].mean()
        ax.hlines(mean, offsets[cohort] - 0.18, offsets[cohort] + 0.18, color="0.2", lw=1.5)
        for _, r in sub.iterrows():
            xpos = offsets[cohort] + (r.seed - 43) * 0.07
            ax.scatter(xpos, r.auroc, s=55, color=colors[cohort], label=cohort if r.seed == sub.seed.min() else None, zorder=3)
            ax.annotate(f"{r.seed}", (xpos, r.auroc), textcoords="offset points",
                        xytext=(6, 4), fontsize=7, color="0.35")
    ax.set_xticks(range(len(cohorts)))
    ax.set_xticklabels(cohorts)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.65, 0.92)
    ax.set_title("Seed-to-seed variability (noise-augmented model)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "seed_variability.png", dpi=300)
    fig.savefig(FIG / "seed_variability.pdf")
    fig.savefig(FIG / "submission" / "FigureS5.png", dpi=300)
    from PIL import Image
    Image.open(FIG / "submission" / "FigureS5.png").convert("RGB").save(
        FIG / "submission" / "FigureS5.jpg", quality=95, dpi=(300, 300)
    )
    plt.close(fig)
    print(f"wrote {FIG / 'seed_variability.png'}")


if __name__ == "__main__":
    main()
