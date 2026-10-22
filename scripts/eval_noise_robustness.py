"""Noise-robustness evaluation: clean-trained vs noise-augmented HYP CNN.

Compares the frozen v1 checkpoint (trained before the augmentation bug was
fixed) with the properly augmented checkpoint (``hyp_v1_aug``) on clean and
corrupted PTB-XL test / Chapman signals.  Corruption severity is expressed as a
multiple of the training augmentation level (sigma 0.05, wander 0.1, dropout
0.2); each level is averaged over several independent noise draws.

Usage:
  python scripts/eval_noise_robustness.py
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
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ai_ecg_hf.dataset import add_wearable_like_noise  # noqa: E402
from ai_ecg_hf.model import TinyECGCNN  # noqa: E402

# Reuse the exact frozen evaluation sets from the external-validation analysis.
from analyze_chapman_external import (  # noqa: E402
    load_chapman_case_control,
    load_ptbxl_test,
)
from eval_georgia_external import load_georgia  # noqa: E402

OUT_DIR = ROOT / "outputs" / "noise_robustness"
FIG_DIR = ROOT / "figures"
CKPTS = {
    "clean": ROOT / "outputs" / "hyp_v1" / "best_model.pt",
    "augmented": ROOT / "outputs" / "hyp_v1_aug" / "best_model.pt",
}
SEVERITIES = [0.0, 0.5, 1.0, 2.0, 4.0]
N_DRAWS = 3
SEED = 42
BASE_SIGMA = 0.05
BASE_WANDER = 0.1
BASE_DROPOUT = 0.2


@torch.no_grad()
def predict_logits(model: torch.nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x[:, None, :])), batch_size=256)
    out = []
    for (xb,) in loader:
        out.append(model(xb.to(device)).detach().cpu().numpy())
    return np.concatenate(out)


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint {path}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = TinyECGCNN()
    model.load_state_dict(blob["model"])
    model.to(device).eval()
    return model


def corrupt(x: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    if severity == 0:
        return x.copy()
    sigma = BASE_SIGMA * severity
    wander = BASE_WANDER * severity
    dropout = min(BASE_DROPOUT * severity, 0.9)
    out = x.copy()
    for i in range(out.shape[0]):
        out[i] = add_wearable_like_noise(
            out[i],
            rng,
            sigma=sigma,
            wander_amp=wander,
            dropout_p=dropout,
        )
    return out


def evaluate(model, x, y, device, n_draws: int, seed: int) -> dict:
    rows = []
    for si, severity in enumerate(SEVERITIES):
        aurocs, auprcs = [], []
        for draw in range(n_draws if severity > 0 else 1):
            rng = np.random.default_rng(seed + 1000 * si + draw)
            x_eval = corrupt(x, severity, rng)
            scores = 1.0 / (1.0 + np.exp(-predict_logits(model, x_eval, device)))
            aurocs.append(roc_auc_score(y, scores))
            auprcs.append(average_precision_score(y, scores))
        rows.append(
            {
                "severity": severity,
                "auroc_mean": float(np.mean(aurocs)),
                "auroc_sd": float(np.std(aurocs)),
                "auprc_mean": float(np.mean(auprcs)),
                "auprc_sd": float(np.std(auprcs)),
                "n_draws": len(aurocs),
            }
        )
        print(
            f"  severity={severity:>3}: AUROC={np.mean(aurocs):.4f} "
            f"±{np.std(aurocs):.4f}  AUPRC={np.mean(auprcs):.4f}"
        )
    return {"rows": rows}


def plot(results: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))
    specs = [
        ("ptbxl", "PTB-XL test (fold 10)"),
        ("chapman", "Chapman-Shaoxing"),
        ("georgia", "Georgia (Emory)"),
    ]
    styles = {
        "clean": ("Clean-trained", "C0", "o"),
        "augmented": ("Noise-augmented", "C1", "s"),
    }
    for ax, (key, title) in zip(axes, specs):
        for model_key, (label, color, marker) in styles.items():
            rows = results["cohorts"][key][model_key]["rows"]
            sev = [r["severity"] for r in rows]
            mean = [r["auroc_mean"] for r in rows]
            sd = [r["auroc_sd"] for r in rows]
            ax.errorbar(
                sev,
                mean,
                yerr=sd,
                marker=marker,
                color=color,
                capsize=3,
                lw=1.8,
                label=label,
            )
        ax.axhline(0.5, ls=":", color="0.6", lw=1)
        ax.set_xlabel("Noise severity (× training augmentation)")
        ax.set_ylabel("AUROC")
        ax.set_title(f"{title} (n={results['cohorts'][key]['n']})")
        ax.set_ylim(0.45, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-draws", type=int, default=N_DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}")

    ch = load_chapman_case_control(seed=args.seed)
    pt = load_ptbxl_test(seed=args.seed)
    ga = load_georgia()
    cohorts = {
        "ptbxl": {"x": pt["x"], "y": pt["y"], "n": int(len(pt["y"]))},
        "chapman": {"x": ch["x"], "y": ch["y"], "n": int(len(ch["y"]))},
        "georgia": {"x": ga["x"], "y": ga["y"], "n": int(len(ga["y"]))},
    }

    results = {
        "severities": SEVERITIES,
        "n_draws": args.n_draws,
        "noise": {
            "base_sigma": BASE_SIGMA,
            "base_wander_amp": BASE_WANDER,
            "base_dropout_p": BASE_DROPOUT,
        },
        "checkpoints": {k: str(v) for k, v in CKPTS.items()},
        "cohorts": {},
    }

    for cohort_key, cohort in cohorts.items():
        results["cohorts"][cohort_key] = {
            "n": cohort["n"],
            "pos_rate": float(cohort["y"].mean()),
            "clean": {},
            "augmented": {},
        }
        for model_key, ckpt in CKPTS.items():
            print(f"[{cohort_key}] {model_key}  ({ckpt.name})")
            model = load_model(ckpt, device)
            results["cohorts"][cohort_key][model_key] = evaluate(
                model, cohort["x"], cohort["y"], device, args.n_draws, args.seed
            )
            del model

    # Deltas (augmented - clean) at each severity.
    table_rows = []
    for cohort_key in cohorts:
        clean = results["cohorts"][cohort_key]["clean"]["rows"]
        aug = results["cohorts"][cohort_key]["augmented"]["rows"]
        for c, a in zip(clean, aug):
            table_rows.append(
                {
                    "cohort": cohort_key,
                    "severity": c["severity"],
                    "auroc_clean": c["auroc_mean"],
                    "auroc_augmented": a["auroc_mean"],
                    "delta_auroc": a["auroc_mean"] - c["auroc_mean"],
                    "auprc_clean": c["auprc_mean"],
                    "auprc_augmented": a["auprc_mean"],
                }
            )
    table = pd.DataFrame(table_rows)
    table.to_csv(OUT_DIR / "table4_noise_robustness.csv", index=False)

    results["deltas"] = table_rows
    (OUT_DIR / "noise_metrics.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8"
    )
    plot(results, FIG_DIR / "noise_robustness.png")

    print("\n==== delta AUROC (augmented - clean) ====")
    print(table.pivot(index="severity", columns="cohort", values="delta_auroc").round(4).to_string())
    print(f"wrote {OUT_DIR / 'noise_metrics.json'}")
    print(f"wrote {FIG_DIR / 'noise_robustness.png'}")


if __name__ == "__main__":
    main()
