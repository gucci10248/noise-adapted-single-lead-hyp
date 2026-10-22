"""Formal HYP screening experiment on PTB-XL Lead I (folds 1-8/9/10).

Usage:
  source .venv/bin/activate
  python scripts/train_hyp.py
  python scripts/train_hyp.py --no-aug --tag hyp_v1_noaug   # clean ablation
  python scripts/train_hyp.py --seed 43 --tag hyp_v1_seed43
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_ecg_hf.dataset import (  # noqa: E402
    PTBXLLeadIBinary,
    add_wearable_like_noise,
    zscore,
)
from ai_ecg_hf.model import TinyECGCNN  # noqa: E402
from ai_ecg_hf.ptbxl import extract_lead_i, read_record  # noqa: E402

SEED = 42
EPOCHS = 20
BATCH = 64
LR = 1e-3
PATIENCE = 4
POS = ("HYP",)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def preload(
    split: str, augment_seed: int | None = None, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Lead I + labels (+ optional age/sex matrix) into memory once."""
    ds = PTBXLLeadIBinary(
        split=split,
        positive_superclasses=POS,
        augment=False,
        require_local=True,
        seed=seed,
    )
    xs, ys, metas = [], [], []
    rng = np.random.default_rng(augment_seed) if augment_seed is not None else None
    for ecg_id in tqdm(ds.ids, desc=f"preload-{split}", leave=False):
        row = ds.df.loc[ecg_id]
        rel = row["filename_lr"]
        signal, meta = read_record(ds.root, rel)
        lead = zscore(extract_lead_i(signal, meta))
        if rng is not None:
            lead = add_wearable_like_noise(lead, rng)
        xs.append(lead)
        ys.append(float(row["y"]))
        age = row["age"] if np.isfinite(row["age"]) else 60.0
        sex = float(row["sex"])  # PTB-XL: 0/1
        metas.append([age, sex])
    x = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.float32)
    m = np.asarray(metas, dtype=np.float32)
    return x, y, m


def make_loader(
    x: np.ndarray, y: np.ndarray, shuffle: bool, train_aug: bool
) -> DataLoader:
    xt = torch.from_numpy(x[:, None, :])
    yt = torch.from_numpy(y)
    ds = TensorDataset(xt, yt)
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, num_workers=0)


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


def metrics_from_logits(y_true: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    scores = 1.0 / (1.0 + np.exp(-logits))
    out = {
        "auroc": float(roc_auc_score(y_true, scores)) if len(set(y_true.tolist())) > 1 else float("nan"),
        "auprc": float(average_precision_score(y_true, scores))
        if len(set(y_true.tolist())) > 1
        else float("nan"),
        "pos_rate": float(y_true.mean()),
        "n": int(len(y_true)),
    }
    return out


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    x_np: np.ndarray | None = None,
    y_np: np.ndarray | None = None,
    use_live_aug: bool = False,
    seed: int = SEED,
) -> float:
    model.train()
    total, n = 0.0, 0
    rng = np.random.default_rng(seed + int(torch.randint(0, 10_000, ()).item()))
    for xb, yb in loader:
        if use_live_aug:
            # light batch-wise wearable-like noise on already-loaded tensors
            arr = xb.numpy()
            for i in range(arr.shape[0]):
                arr[i, 0] = add_wearable_like_noise(arr[i, 0], rng)
            xb = torch.from_numpy(arr)
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        total += float(loss.detach()) * len(yb)
        n += len(yb)
    return total / max(n, 1)


def plot_roc(y_true: np.ndarray, scores: np.ndarray, path: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUROC={auc:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loss(history: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="train loss")
    ax.plot([h["epoch"] for h in history], [h["val_auroc"] for h in history], label="val AUROC")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PTB-XL HYP Lead-I CNN")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--no-aug",
        action="store_true",
        help="disable wearable-like augmentation (clean baseline)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="hyp_v1",
        help="output subdirectory under outputs/ and figure prefix",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = ROOT / "outputs" / args.tag
    fig_dir = ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device} tag={args.tag} seed={args.seed} aug={not args.no_aug}")
    x_tr, y_tr, m_tr = preload("train", seed=args.seed)
    x_va, y_va, m_va = preload("val", seed=args.seed)
    x_te, y_te, m_te = preload("test", seed=args.seed)
    print(
        f"n train/val/test={len(y_tr)}/{len(y_va)}/{len(y_te)} "
        f"pos={y_tr.mean():.3f}/{y_va.mean():.3f}/{y_te.mean():.3f}"
    )

    # Age+sex logistic baseline
    base = LogisticRegression(max_iter=1000, class_weight="balanced")
    base.fit(m_tr, y_tr)
    base_va = metrics_from_logits(y_va, base.decision_function(m_va))
    base_te = metrics_from_logits(y_te, base.decision_function(m_te))
    # decision_function is logit-scale; metrics_from_logits applies sigmoid — OK for ranking AUROC
    print(f"baseline age+sex  val AUROC={base_va['auroc']:.3f}  test AUROC={base_te['auroc']:.3f}")

    train_loader = make_loader(x_tr, y_tr, shuffle=True, train_aug=True)
    model = TinyECGCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    # pos weight for imbalance
    pos = float(y_tr.sum())
    neg = float(len(y_tr) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history: list[dict] = []
    best_val = -1.0
    best_state = None
    bad = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(
            model,
            train_loader,
            opt,
            loss_fn,
            device,
            use_live_aug=not args.no_aug,
            seed=args.seed,
        )
        va_logits = predict_logits(model, x_va, device)
        va_m = metrics_from_logits(y_va, va_logits)
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_auroc": va_m["auroc"],
                "val_auprc": va_m["auprc"],
            }
        )
        print(
            f"epoch={epoch:02d} loss={tr_loss:.4f} "
            f"val_auroc={va_m['auroc']:.3f} val_auprc={va_m['auprc']:.3f}"
        )
        if va_m["auroc"] > best_val + 1e-4:
            best_val = va_m["auroc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"early stop at epoch={epoch}")
                break

    if best_state is None:
        raise RuntimeError("No successful training epoch")
    model.load_state_dict(best_state)

    te_logits = predict_logits(model, x_te, device)
    te_m = metrics_from_logits(y_te, te_logits)
    va_logits = predict_logits(model, x_va, device)
    va_m = metrics_from_logits(y_va, va_logits)
    te_scores = 1.0 / (1.0 + np.exp(-te_logits))

    roc_path = fig_dir / f"{args.tag}_test_roc.png"
    curves_path = fig_dir / f"{args.tag}_curves.png"
    plot_roc(
        y_te,
        te_scores,
        roc_path,
        f"PTB-XL Lead I · HYP · test ROC ({args.tag})",
    )
    plot_loss(history, curves_path)

    ckpt = out_dir / "best_model.pt"
    torch.save(
        {
            "model": best_state,
            "seed": args.seed,
            "pos": list(POS),
            "augmentation": not args.no_aug,
        },
        ckpt,
    )

    summary = {
        "task": "HYP binary screening",
        "lead": "I",
        "sampling_hz": 100,
        "splits": "train folds1-8 / val fold9 / test fold10",
        "seed": args.seed,
        "augmentation": not args.no_aug,
        "tag": args.tag,
        "n": {"train": int(len(y_tr)), "val": int(len(y_va)), "test": int(len(y_te))},
        "pos_rate": {
            "train": float(y_tr.mean()),
            "val": float(y_va.mean()),
            "test": float(y_te.mean()),
        },
        "baseline_age_sex": {"val": base_va, "test": base_te},
        "cnn": {"val": va_m, "test": te_m, "best_val_auroc": best_val},
        "history": history,
        "artifacts": {
            "checkpoint": str(ckpt),
            "roc": str(roc_path),
            "curves": str(curves_path),
        },
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("==== RESULT ====")
    print(json.dumps({"baseline_test_auroc": base_te["auroc"], "cnn_test": te_m}, indent=2))
    print(f"wrote {metrics_path}")
    print(f"wrote {roc_path}")


if __name__ == "__main__":
    main()
