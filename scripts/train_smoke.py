"""Train TinyECGCNN for a few steps once PTB-XL records100 is available."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_ecg_hf.dataset import PTBXLLeadIBinary  # noqa: E402
from ai_ecg_hf.model import TinyECGCNN  # noqa: E402


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # require_local=True: only use already-downloaded waveforms (safe during partial sync)
    train_ds = PTBXLLeadIBinary(
        split="train",
        augment=True,
        positive_superclasses=("HYP",),
        require_local=True,
    )
    val_ds = PTBXLLeadIBinary(
        split="val",
        augment=False,
        positive_superclasses=("HYP",),
        require_local=True,
    )
    print(
        f"train={len(train_ds)} val={len(val_ds)} "
        f"pos_rate_train={train_ds.label_frame()['y'].mean():.3f} device={device}"
    )
    if len(train_ds) < 16:
        raise RuntimeError(
            f"Too few local train records ({len(train_ds)}). "
            "Wait for sync or run: nohup zsh scripts/sync_records100.sh &"
        )

    loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    model = TinyECGCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model.train()
    for step, (xb, yb) in enumerate(loader, start=1):
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        if step % 5 == 0:
            print(f"step={step} loss={float(loss.detach()):.4f}")
        if step >= 20:
            break

    # Quick val AUROC on up to 128 samples
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for i in range(min(128, len(val_ds))):
            x, y = val_ds[i]
            logit = model(x.unsqueeze(0).to(device)).cpu().item()
            ys.append(float(y))
            ps.append(logit)
    if len(set(ys)) > 1:
        # logits as scores are fine for ranking AUROC
        auc = roc_auc_score(ys, ps)
        print(f"smoke val AUROC (~128) = {auc:.3f}")
    else:
        print("smoke val: single-class subsample, skip AUROC")
    print("train_smoke OK")


if __name__ == "__main__":
    main()
