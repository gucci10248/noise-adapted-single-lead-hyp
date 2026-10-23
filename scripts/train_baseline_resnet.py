"""Train a 1D ResNet baseline on the same PTB-XL Lead-I split.

Usage:
  source .venv/bin/activate
  python scripts/train_baseline_resnet.py --tag hyp_resnet --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ai_ecg_hf.baselines import LeadIResNet, count_parameters  # noqa: E402
from train_hyp import (  # noqa: E402
    EPOCHS,
    PATIENCE,
    SEED,
    make_loader,
    metrics_from_logits,
    plot_loss,
    plot_roc,
    predict_logits,
    preload,
    set_seed,
    train_one_epoch,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tag", type=str, default="hyp_resnet")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--aug-mode", type=str, default="full")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = ROOT / "outputs" / args.tag
    fig_dir = ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    x_tr, y_tr, _ = preload("train", seed=args.seed)
    x_va, y_va, _ = preload("val", seed=args.seed)
    x_te, y_te, _ = preload("test", seed=args.seed)

    model = LeadIResNet().to(device)
    print(f"device={device} tag={args.tag} params={count_parameters(model)}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    pos = float(y_tr.sum())
    neg = float(len(y_tr) - pos)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([neg / max(pos, 1.0)], device=device)
    )

    train_loader = make_loader(x_tr, y_tr, shuffle=True, train_aug=True)
    history = []
    best_val = -1.0
    best_state = None
    bad = 0
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            model,
            train_loader,
            opt,
            loss_fn,
            device,
            use_live_aug=args.aug_mode != "none",
            aug_mode=args.aug_mode,
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
    te_scores = 1.0 / (1.0 + np.exp(-te_logits))

    ckpt = out_dir / "best_model.pt"
    torch.save(
        {
            "model": best_state,
            "arch": "LeadIResNet",
            "seed": args.seed,
            "aug_mode": args.aug_mode,
        },
        ckpt,
    )
    roc_path = fig_dir / f"{args.tag}_test_roc.png"
    plot_roc(y_te, te_scores, roc_path, f"PTB-XL Lead I HYP ResNet ({args.tag})")
    plot_loss(history, fig_dir / f"{args.tag}_curves.png")

    summary = {
        "task": "HYP binary screening",
        "arch": "LeadIResNet",
        "params": count_parameters(model),
        "tag": args.tag,
        "seed": args.seed,
        "aug_mode": args.aug_mode,
        "n": {"train": int(len(y_tr)), "val": int(len(y_va)), "test": int(len(y_te))},
        "test": te_m,
        "best_val_auroc": best_val,
        "history": history,
        "checkpoint": str(ckpt),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"resnet_test": te_m}, indent=2))
    print(f"wrote {ckpt}")


if __name__ == "__main__":
    main()
