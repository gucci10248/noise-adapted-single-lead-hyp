"""Evaluate external baselines for single-lead HYP screening.

Baselines (all trained on PTB-XL folds 1-8 only):
  - age + sex logistic
  - Lead-I feature logistic (39 features)
  - Lead-I feature gradient-boosted trees (sklearn HistGradientBoosting)
  - 1D ResNet (scripts/train_baseline_resnet.py)
  - TinyECGCNN noise-augmented (reference)

Usage:
  python scripts/eval_baselines.py --n-boot 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ai_ecg_hf.baselines import (  # noqa: E402
    FEATURE_NAMES,
    LeadIResNet,
    build_feature_matrix,
    count_parameters,
    fit_feature_logistic,
    fit_feature_trees,
)
from ai_ecg_hf.model import TinyECGCNN  # noqa: E402
from analyze_chapman_external import (  # noqa: E402
    bootstrap_metrics,
    load_chapman_case_control,
    load_ptbxl_test,
)
from eval_georgia_external import load_georgia  # noqa: E402
from train_hyp import preload, predict_logits, set_seed  # noqa: E402

OUT_DIR = ROOT / "outputs" / "baselines"
SEED = 42


def scores_from_logits(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=float)))


def encode_sex(values: np.ndarray) -> np.ndarray:
    """PTB-XL coding: 0 = male, 1 = female; unknown -> 0."""
    out = []
    for v in values:
        s = str(v).strip().lower()
        out.append(1.0 if s in {"f", "female", "1"} else 0.0)
    return np.asarray(out, dtype=float)


def age_fill(values: np.ndarray) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    return np.where(np.isfinite(a) & (a > 0), a, 60.0)


def eval_scores(y, scores, n_boot: int, seed: int) -> dict:
    y = np.asarray(y, dtype=float)
    scores = np.asarray(scores, dtype=float)
    m = bootstrap_metrics(y, scores, n_boot=n_boot, seed=seed)
    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "pos_rate": float(y.mean()),
        "auroc": float(roc_auc_score(y, scores)),
        "auroc_ci95": m["auroc_ci95"],
        "auprc": float(average_precision_score(y, scores)),
        "auprc_ci95": m["auprc_ci95"],
    }


def load_resnet(path: Path, device: torch.device) -> torch.nn.Module:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = LeadIResNet()
    model.load_state_dict(blob["model"])
    return model.to(device).eval()


def load_tiny(path: Path, device: torch.device) -> torch.nn.Module:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = TinyECGCNN()
    model.load_state_dict(blob["model"])
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    set_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("loading PTB-XL train/test ...")
    x_tr, y_tr, m_tr = preload("train", seed=args.seed)
    x_te, y_te, m_te = preload("test", seed=args.seed)
    print("loading Chapman ...")
    ch = load_chapman_case_control(seed=args.seed)
    print("loading Georgia ...")
    ga = load_georgia()

    cohorts = {
        "ptbxl_test": {
            "x": x_te,
            "y": y_te,
            "age": m_te[:, 0],
            "sex": m_te[:, 1],
        },
        "chapman": {
            "x": ch["x"],
            "y": ch["y"],
            "age": ch["age"],
            "sex": encode_sex(ch["sex"]),
        },
        "georgia": {
            "x": ga["x"],
            "y": ga["y"],
            "age": ga["age"],
            "sex": encode_sex(ga["sex"]),
        },
    }

    # Age + sex logistic, trained on PTB-XL train.
    age_sex = LogisticRegression(max_iter=1000, class_weight="balanced")
    age_sex.fit(np.column_stack([m_tr[:, 0], m_tr[:, 1]]), y_tr)

    print("fitting Lead-I feature models ...")
    feat_log = fit_feature_logistic(x_tr, y_tr)
    feat_tree = fit_feature_trees(x_tr, y_tr)

    print("loading deep baselines ...")
    resnet = load_resnet(ROOT / "outputs" / "hyp_resnet" / "best_model.pt", device)
    tiny = load_tiny(ROOT / "outputs" / "hyp_v1_aug" / "best_model.pt", device)

    rows = []
    model_meta = {
        "age_sex": {"params": 2, "features": "age, sex"},
        "feature_logistic": {"params": None, "features": len(FEATURE_NAMES)},
        "feature_trees": {"params": None, "features": len(FEATURE_NAMES)},
        "resnet": {"params": count_parameters(resnet), "features": "raw Lead I"},
        "tiny_cnn_aug": {"params": count_parameters(tiny), "features": "raw Lead I"},
    }
    for cohort_key, c in cohorts.items():
        x = c["x"]
        y = c["y"]
        print(f"\n[{cohort_key}] n={len(y)} pos={y.mean():.3f}")

        # Age + sex
        s = age_sex.decision_function(np.column_stack([age_fill(c["age"]), c["sex"]]))
        rows.append(
            {
                "cohort": cohort_key,
                "model": "age_sex",
                **eval_scores(y, scores_from_logits(s), args.n_boot, args.seed),
            }
        )
        print("  age_sex done")

        # Feature logistic
        s = feat_log.predict_proba(build_feature_matrix(x))[:, 1]
        rows.append(
            {
                "cohort": cohort_key,
                "model": "feature_logistic",
                **eval_scores(y, s, args.n_boot, args.seed),
            }
        )
        print("  feature_logistic done")

        # Feature trees
        s = feat_tree.predict_proba(build_feature_matrix(x))[:, 1]
        rows.append(
            {
                "cohort": cohort_key,
                "model": "feature_trees",
                **eval_scores(y, s, args.n_boot, args.seed),
            }
        )
        print("  feature_trees done")

        # Deep models
        for name, model in [("resnet", resnet), ("tiny_cnn_aug", tiny)]:
            s = scores_from_logits(predict_logits(model, x, device))
            rows.append(
                {
                    "cohort": cohort_key,
                    "model": name,
                    **eval_scores(y, s, args.n_boot, args.seed),
                }
            )
            print(f"  {name} done")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "baseline_metrics.csv", index=False)
    meta = {
        "seed": args.seed,
        "n_boot": args.n_boot,
        "feature_names": FEATURE_NAMES,
        "models": model_meta,
        "cohorts": {k: {"n": int(len(v["y"])), "pos_rate": float(v["y"].mean())} for k, v in cohorts.items()},
    }
    (OUT_DIR / "baseline_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n==== AUROC by cohort/model ====")
    print(df.pivot(index="model", columns="cohort", values="auroc").round(4).to_string())
    print(f"\nwrote {OUT_DIR / 'baseline_metrics.csv'}")


if __name__ == "__main__":
    main()
