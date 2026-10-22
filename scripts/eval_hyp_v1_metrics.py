"""HYP v1.1 metric pack: val-locked thresholds + test operating points + calibration.

Thresholds are chosen ONLY on validation (fold 9), then frozen and applied to test.
Does not retrain the CNN.

Usage:
  source .venv/bin/activate
  python scripts/eval_hyp_v1_metrics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_ecg_hf.dataset import PTBXLLeadIBinary, zscore  # noqa: E402
from ai_ecg_hf.model import TinyECGCNN  # noqa: E402
from ai_ecg_hf.ptbxl import extract_lead_i, read_record  # noqa: E402

SEED = 42
POS = ("HYP",)
OUT_DIR = ROOT / "outputs" / "hyp_v1"
FIG_DIR = ROOT / "figures"
CKPT = OUT_DIR / "best_model.pt"


def preload(split: str) -> tuple[np.ndarray, np.ndarray]:
    ds = PTBXLLeadIBinary(
        split=split,
        positive_superclasses=POS,
        augment=False,
        require_local=True,
        seed=SEED,
    )
    xs, ys = [], []
    for ecg_id in tqdm(ds.ids, desc=f"preload-{split}", leave=False):
        row = ds.df.loc[ecg_id]
        signal, meta = read_record(ds.root, row["filename_lr"])
        xs.append(zscore(extract_lead_i(signal, meta)))
        ys.append(float(row["y"]))
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32)


@torch.no_grad()
def predict_proba(model: torch.nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x[:, None, :])),
        batch_size=256,
        shuffle=False,
    )
    outs = []
    for (xb,) in loader:
        logits = model(xb.to(device)).detach().cpu().numpy()
        outs.append(1.0 / (1.0 + np.exp(-logits)))
    return np.concatenate(outs, axis=0)


def operating_point(y_true: np.ndarray, scores: np.ndarray, thr: float) -> dict[str, float]:
    yhat = (scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    return {
        "threshold": float(thr),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def choose_thresholds(y_val: np.ndarray, s_val: np.ndarray) -> dict[str, float]:
    fpr, tpr, thr = roc_curve(y_val, s_val)
    # sklearn thr has len = len(tpr); last thr is inf-like handling — use finite
    youden_idx = int(np.argmax(tpr - fpr))
    youden = float(thr[youden_idx])

    # target sensitivity >= 0.90 on val: pick highest thr that still meets sens
    sens_thr = 0.0
    for t in np.unique(s_val)[::-1]:
        op = operating_point(y_val, s_val, float(t))
        if op["sensitivity"] >= 0.90:
            sens_thr = float(t)
            break

    # target specificity >= 0.90 on val: pick lowest thr that still meets spec
    # (higher thr → higher spec). Scan ascending scores.
    spec_thr = 1.0
    for t in np.unique(s_val):
        op = operating_point(y_val, s_val, float(t))
        if op["specificity"] >= 0.90:
            spec_thr = float(t)
            break

    return {
        "youden": youden,
        "sens_ge_0.90": sens_thr,
        "spec_ge_0.90": spec_thr,
    }


def expected_calibration_error(
    y_true: np.ndarray, scores: np.ndarray, n_bins: int = 10
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (scores >= bins[i]) & (scores < bins[i + 1] if i < n_bins - 1 else scores <= bins[i + 1])
        if not np.any(m):
            continue
        ece += (m.mean()) * abs(scores[m].mean() - y_true[m].mean())
    return float(ece)


def plot_calibration(y_true: np.ndarray, scores: np.ndarray, path: Path, title: str) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, scores, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1, label="ideal")
    ax.plot(mean_pred, frac_pos, marker="o", lw=2, label="CNN")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_threshold_table(
    rows: list[tuple[str, dict]], path: Path, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.axis("off")
    cell = [
        [
            name,
            f"{r['threshold']:.3f}",
            f"{r['sensitivity']:.3f}",
            f"{r['specificity']:.3f}",
            f"{r['ppv']:.3f}",
            f"{r['npv']:.3f}",
        ]
        for name, r in rows
    ]
    table = ax.table(
        cellText=cell,
        colLabels=["rule", "thr", "sens", "spec", "PPV", "NPV"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)
    ax.set_title(title, pad=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    if not CKPT.exists():
        raise FileNotFoundError(f"Missing {CKPT}; run scripts/train_hyp.py first")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = TinyECGCNN()
    model.load_state_dict(blob["model"])
    model.to(device)

    x_va, y_va = preload("val")
    x_te, y_te = preload("test")
    s_va = predict_proba(model, x_va, device)
    s_te = predict_proba(model, x_te, device)

    thrs = choose_thresholds(y_va, s_va)
    print("val-locked thresholds:", thrs)

    val_ops = {k: operating_point(y_va, s_va, t) for k, t in thrs.items()}
    test_ops = {k: operating_point(y_te, s_te, t) for k, t in thrs.items()}

    ece_va = expected_calibration_error(y_va, s_va)
    ece_te = expected_calibration_error(y_te, s_te)
    print(f"ECE val={ece_va:.4f} test={ece_te:.4f}")
    print("test @ youden:", test_ops["youden"])
    print("test @ sens>=0.90:", test_ops["sens_ge_0.90"])
    print("test @ spec>=0.90:", test_ops["spec_ge_0.90"])

    plot_calibration(
        y_te,
        s_te,
        FIG_DIR / "hyp_v1_test_calibration.png",
        f"PTB-XL Lead I · HYP · test calibration (ECE={ece_te:.3f})",
    )
    plot_threshold_table(
        [(k, test_ops[k]) for k in thrs],
        FIG_DIR / "hyp_v1_test_operating_points.png",
        "Test operating points (thresholds locked on val)",
    )

    base = json.loads((OUT_DIR / "metrics.json").read_text(encoding="utf-8"))
    pack = {
        "version": "v1.1",
        "note": "Thresholds chosen on validation only; applied frozen to test.",
        "auroc": {
            "val": float(roc_auc_score(y_va, s_va)),
            "test": float(roc_auc_score(y_te, s_te)),
        },
        "thresholds_val_locked": thrs,
        "operating_points": {"val": val_ops, "test": test_ops},
        "calibration": {
            "ece_val": ece_va,
            "ece_test": ece_te,
            "n_bins": 10,
            "strategy": "uniform",
        },
        "figures": {
            "calibration": str(FIG_DIR / "hyp_v1_test_calibration.png"),
            "operating_points": str(FIG_DIR / "hyp_v1_test_operating_points.png"),
        },
        "parent_metrics": {
            "cnn_test_auroc": base.get("cnn", {}).get("test", {}).get("auroc"),
            "baseline_test_auroc": base.get("baseline_age_sex", {}).get("test", {}).get("auroc"),
        },
    }
    out = OUT_DIR / "metrics_v1_1.json"
    out.write_text(json.dumps(pack, indent=2), encoding="utf-8")

    # also merge summary key into metrics.json for convenience
    base["v1_1"] = pack
    (OUT_DIR / "metrics.json").write_text(json.dumps(base, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
