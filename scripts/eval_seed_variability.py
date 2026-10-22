"""Evaluate seed-to-seed variability of the noise-augmented model.

Seeds 42, 43, and 44 are evaluated on PTB-XL test, Chapman, and Georgia with the
same preprocessing and no recalibration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ai_ecg_hf.model import TinyECGCNN  # noqa: E402
from analyze_chapman_external import (  # noqa: E402
    load_chapman_case_control,
    load_ptbxl_test,
    predict_logits,
)
from eval_georgia_external import load_georgia  # noqa: E402

OUT = ROOT / "outputs" / "seed_variability"
CKPTS = {
    42: ROOT / "outputs" / "hyp_v1_aug" / "best_model.pt",
    43: ROOT / "outputs" / "hyp_v1_aug_seed43" / "best_model.pt",
    44: ROOT / "outputs" / "hyp_v1_aug_seed44" / "best_model.pt",
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    cohorts = {
        "PTB-XL test": load_ptbxl_test(seed=42),
        "Chapman": load_chapman_case_control(seed=42),
        "Georgia": load_georgia(),
    }
    for name, c in cohorts.items():
        c["y"] = np.asarray(c["y"]).astype(int)

    rows = []
    for seed, ckpt in CKPTS.items():
        if not ckpt.exists():
            print(f"missing {ckpt}; skipping seed {seed}")
            continue
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = TinyECGCNN()
        model.load_state_dict(blob["model"])
        model.to(device).eval()
        for name, c in cohorts.items():
            logits = predict_logits(model, c["x"], device)
            p = sigmoid(logits)
            rows.append(
                {
                    "seed": seed,
                    "cohort": name,
                    "n": int(len(c["y"])),
                    "n_pos": int(c["y"].sum()),
                    "auroc": float(roc_auc_score(c["y"], p)),
                    "auprc": float(average_precision_score(c["y"], p)),
                }
            )
        del model

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "seed_metrics.csv", index=False)
    summary = (
        df.groupby("cohort")
        .agg(
            n=("n", "first"),
            n_pos=("n_pos", "first"),
            auroc_mean=("auroc", "mean"),
            auroc_sd=("auroc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_sd=("auprc", "std"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    summary.to_csv(OUT / "seed_summary.csv", index=False)
    (OUT / "summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    print(df.to_string(index=False))
    print("\n==== summary ====")
    print(summary.round(4).to_string(index=False))
    print(f"wrote {OUT / 'seed_summary.csv'}")


if __name__ == "__main__":
    main()
