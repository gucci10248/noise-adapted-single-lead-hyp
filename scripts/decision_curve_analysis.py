"""Decision curve analysis for the HYP models.

PTB-XL test and Georgia have natural prevalence and are suitable for DCA.
Chapman is case-control and is therefore excluded from the primary DCA.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ai_ecg_hf.model import TinyECGCNN  # noqa: E402
from analyze_chapman_external import load_ptbxl_test, predict_logits  # noqa: E402

OUT = ROOT / "outputs" / "dca"
FIG = ROOT / "figures"
CKPTS = {
    "noise-augmented": ROOT / "outputs" / "hyp_v1_aug" / "best_model.pt",
    "clean": ROOT / "outputs" / "hyp_v1" / "best_model.pt",
}
THRESHOLDS = np.round(np.arange(0.01, 0.601, 0.01), 2)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def net_benefit(y: np.ndarray, p: np.ndarray, pt: float) -> float:
    n = len(y)
    pred = p >= pt
    tp = int(((pred) & (y == 1)).sum())
    fp = int(((pred) & (y == 0)).sum())
    return tp / n - fp / n * (pt / (1 - pt))


def dca_table(y: np.ndarray, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    prev = float(y.mean())
    for pt in THRESHOLDS:
        row = {"threshold": float(pt), "treat_all": prev - (1 - prev) * pt / (1 - pt)}
        for name, p in scores.items():
            row[name] = net_benefit(y, p, pt)
        rows.append(row)
    return pd.DataFrame(rows)


def plot(results: dict[str, pd.DataFrame], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, (cohort, df) in zip(axes, results.items()):
        ax.plot(df["threshold"], df["noise-augmented"], lw=2, color="C1", label="Noise-augmented")
        ax.plot(df["threshold"], df["clean"], lw=2, color="C0", label="Clean")
        ax.plot(df["threshold"], df["treat_all"], lw=1.2, ls="--", color="0.5", label="Treat all")
        ax.axhline(0, lw=1.2, ls=":", color="0.3", label="Treat none")
        ax.set_xlim(0.01, 0.60)
        ax.set_ylim(-0.05, max(0.20, df[["noise-augmented", "clean"]].max().max() + 0.03))
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
        ax.set_title(cohort)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # PTB-XL test: compute scores once, save for reproducibility.
    pt = load_ptbxl_test(seed=42)
    pt_scores = {}
    for name, ckpt in CKPTS.items():
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = TinyECGCNN()
        model.load_state_dict(blob["model"])
        model.to(device).eval()
        pt_scores[name] = sigmoid(predict_logits(model, pt["x"], device))
        del model
    pd.DataFrame(
        {"y": pt["y"].astype(int), **{f"score_{k}": v for k, v in pt_scores.items()}}
    ).to_csv(OUT / "ptbxl_test_scores.csv", index=False)
    pt_y = np.asarray(pt["y"]).astype(int)

    # Georgia predictions
    ga_aug = pd.read_csv(ROOT / "outputs" / "georgia_ext" / "predictions.csv")
    ga_clean = pd.read_csv(ROOT / "outputs" / "georgia_ext_clean" / "predictions.csv")
    ga_y = ga_aug["y_hyp"].to_numpy().astype(int)
    ga_scores = {
        "noise-augmented": ga_aug["score"].to_numpy(),
        "clean": ga_clean["score"].to_numpy(),
    }

    results = {
        "PTB-XL test": dca_table(pt_y, pt_scores),
        "Georgia (Emory)": dca_table(ga_y, ga_scores),
    }
    combined = pd.concat(
        [df.assign(cohort=k) for k, df in results.items()], ignore_index=True
    )
    combined.to_csv(OUT / "dca.csv", index=False)
    plot(results, FIG / "dca.png")

    summary = {}
    for cohort, df in results.items():
        row = {}
        for pt in [0.10, 0.20, 0.30]:
            r = df.loc[np.isclose(df["threshold"], pt)].iloc[0]
            row[f"at_{pt:.2f}"] = {
                "noise-augmented": float(r["noise-augmented"]),
                "clean": float(r["clean"]),
                "treat_all": float(r["treat_all"]),
            }
        summary[cohort] = row
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT / 'dca.csv'}")
    print(f"wrote {FIG / 'dca.png'}")


if __name__ == "__main__":
    main()
