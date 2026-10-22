"""Fit temperature scaling on val logits; report calibrated test ECE / operating points.

Temperature T is fit by minimizing NLL on validation only, then frozen for test.

Usage:
  source .venv/bin/activate
  python scripts/calibrate_hyp_temperature.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
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
def predict_logits(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x[:, None, :])),
        batch_size=256,
        shuffle=False,
    )
    outs = []
    for (xb,) in loader:
        outs.append(model(xb.to(device)).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = 0.0
    for i in range(n_bins):
        hi = bins[i + 1]
        m = (p >= bins[i]) & (p < hi if i < n_bins - 1 else p <= hi)
        if np.any(m):
            out += float(m.mean()) * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(out)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Scalar T>0 minimizing BCEWithLogits on val."""
    device = torch.device("cpu")
    z = torch.tensor(logits, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    # optimize log_T for positivity
    log_t = torch.nn.Parameter(torch.zeros((), device=device))
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=50)

    loss_fn = nn.BCEWithLogitsLoss()

    def closure() -> torch.Tensor:
        opt.zero_grad()
        t = torch.exp(log_t)
        loss = loss_fn(z / t, yt)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(torch.exp(log_t).detach())
    return max(t, 1e-3)


def operating_point(y: np.ndarray, scores: np.ndarray, thr: float) -> dict[str, float]:
    yhat = (scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "ppv": float(tp / max(tp + fp, 1)),
        "npv": float(tn / max(tn + fn, 1)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def choose_thresholds(y_val: np.ndarray, s_val: np.ndarray) -> dict[str, float]:
    fpr, tpr, thr = roc_curve(y_val, s_val)
    youden = float(thr[int(np.argmax(tpr - fpr))])
    sens_thr = 0.0
    for t in np.unique(s_val)[::-1]:
        if operating_point(y_val, s_val, float(t))["sensitivity"] >= 0.90:
            sens_thr = float(t)
            break
    spec_thr = 1.0
    for t in np.unique(s_val):
        if operating_point(y_val, s_val, float(t))["specificity"] >= 0.90:
            spec_thr = float(t)
            break
    return {"youden": youden, "sens_ge_0.90": sens_thr, "spec_ge_0.90": spec_thr}


def plot_calibration_compare(
    y: np.ndarray, p_raw: np.ndarray, p_cal: np.ndarray, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1, label="ideal")
    for scores, name in ((p_raw, "raw"), (p_cal, "temp-scaled")):
        frac, mean = calibration_curve(y, scores, n_bins=10, strategy="uniform")
        ax.plot(mean, frac, marker="o", lw=2, label=f"{name} ECE={ece(y, scores):.3f}")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Test calibration: raw vs temperature scaling")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = TinyECGCNN()
    model.load_state_dict(blob["model"])
    model.to(device)

    x_va, y_va = preload("val")
    x_te, y_te = preload("test")
    z_va = predict_logits(model, x_va, device)
    z_te = predict_logits(model, x_te, device)

    t = fit_temperature(z_va, y_va)
    p_va_raw, p_te_raw = sigmoid(z_va), sigmoid(z_te)
    p_va_cal, p_te_cal = sigmoid(z_va / t), sigmoid(z_te / t)

    print(f"temperature T={t:.4f}")
    print(
        f"AUROC raw/cal test={roc_auc_score(y_te, p_te_raw):.4f}/"
        f"{roc_auc_score(y_te, p_te_cal):.4f} (should match)"
    )
    print(
        f"ECE val raw/cal={ece(y_va, p_va_raw):.4f}/{ece(y_va, p_va_cal):.4f} | "
        f"test raw/cal={ece(y_te, p_te_raw):.4f}/{ece(y_te, p_te_cal):.4f}"
    )

    thrs = choose_thresholds(y_va, p_va_cal)
    test_ops = {k: operating_point(y_te, p_te_cal, thr) for k, thr in thrs.items()}
    print("calibrated test @ youden:", test_ops["youden"])

    plot_calibration_compare(
        y_te, p_te_raw, p_te_cal, FIG_DIR / "hyp_v1_test_calibration_tempscale.png"
    )

    pack = {
        "version": "v1.2_temperature_scaling",
        "temperature": t,
        "fit_on": "validation fold 9 only",
        "auroc_test_raw": float(roc_auc_score(y_te, p_te_raw)),
        "auroc_test_cal": float(roc_auc_score(y_te, p_te_cal)),
        "ece": {
            "val_raw": ece(y_va, p_va_raw),
            "val_cal": ece(y_va, p_va_cal),
            "test_raw": ece(y_te, p_te_raw),
            "test_cal": ece(y_te, p_te_cal),
        },
        "thresholds_val_locked_on_calibrated": thrs,
        "operating_points_test_calibrated": test_ops,
        "figure": str(FIG_DIR / "hyp_v1_test_calibration_tempscale.png"),
    }
    out = OUT_DIR / "metrics_v1_2_tempscale.json"
    out.write_text(json.dumps(pack, indent=2), encoding="utf-8")

    base_path = OUT_DIR / "metrics.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    base["v1_2_temperature_scaling"] = pack
    # also store T with checkpoint sidecar
    torch.save(
        {"model": blob["model"], "temperature": t, "seed": SEED, "pos": list(POS)},
        OUT_DIR / "best_model_tempscale.pt",
    )
    base_path.write_text(json.dumps(base, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote {OUT_DIR / 'best_model_tempscale.pt'}")


if __name__ == "__main__":
    main()
