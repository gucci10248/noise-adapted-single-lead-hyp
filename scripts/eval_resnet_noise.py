"""Noise robustness of the 1D ResNet baseline (for the model-choice comparison)."""

from __future__ import annotations

import argparse
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

from ai_ecg_hf.baselines import LeadIResNet  # noqa: E402
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
    predict_logits,
)

OUT_DIR = ROOT / "outputs" / "resnet_noise"
SEVERITIES = [0.0, 1.0, 4.0]
N_DRAWS = 3
SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-draws", type=int, default=N_DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    blob = torch.load(
        ROOT / "outputs" / "hyp_resnet" / "best_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = LeadIResNet()
    model.load_state_dict(blob["model"])
    model.to(device).eval()

    cohorts = {
        "ptbxl": load_ptbxl_test(seed=args.seed),
        "chapman": load_chapman_case_control(seed=args.seed),
        "georgia": load_georgia(),
    }
    rows = []
    for cohort_key, c in cohorts.items():
        x, y = c["x"], c["y"]
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
                    "model": "resnet",
                    "cohort": cohort_key,
                    "severity": severity,
                    "auroc_mean": float(np.mean(aurocs)),
                    "auroc_sd": float(np.std(aurocs)),
                    "auprc_mean": float(np.mean(auprcs)),
                    "auprc_sd": float(np.std(auprcs)),
                }
            )
            print(f"{cohort_key:8s} sev={severity:>3}: {np.mean(aurocs):.4f}")
    pd.DataFrame(rows).to_csv(OUT_DIR / "resnet_noise.csv", index=False)
    (OUT_DIR / "resnet_noise.json").write_text(
        json.dumps(
            {
                "severities": SEVERITIES,
                "n_draws": args.n_draws,
                "noise": {
                    "base_sigma": BASE_SIGMA,
                    "base_wander": BASE_WANDER,
                    "base_dropout": BASE_DROPOUT,
                },
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {OUT_DIR / 'resnet_noise.csv'}")


if __name__ == "__main__":
    main()
