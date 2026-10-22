"""External validation of PTB-XL HYP CNN on Chapman-Shaoxing Lead I (LVH SNOMED labels).

Requires:
  - outputs/hyp_v1/best_model.pt (or best_model_tempscale.pt)
  - data/chapman/WFDBRecords/** (see scripts/download_chapman.py)

Usage:
  python scripts/eval_chapman_external.py
  python scripts/eval_chapman_external.py --tempscale --max-n 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_ecg_hf.chapman import list_local_records, read_record  # noqa: E402
from ai_ecg_hf.mimic_ecg import resample_to_length, zscore  # noqa: E402
from ai_ecg_hf.model import TinyECGCNN  # noqa: E402

OUT_DIR = ROOT / "outputs" / "chapman_ext"


def load_xy(records, max_n: int | None, neg_pool: int | None, seed: int):
    # Pre-read labels (hea only, no .mat needed) to allow prevalence-aware sampling
    labeled = []
    for rec in records:
        try:
            _, info = read_record(rec)
        except Exception:
            continue
        labeled.append((rec, info))
    pos = [t for t in labeled if t[1]["y_hyp"] == 1]
    neg = [t for t in labeled if t[1]["y_hyp"] == 0]
    if neg_pool is not None:
        # case-control design: all positives + fixed negative pool
        rng = np.random.RandomState(seed)
        if len(neg) > neg_pool:
            idx = rng.choice(len(neg), size=neg_pool, replace=False)
            neg = [neg[i] for i in sorted(idx)]
        picked = pos + neg
        print(f"design=case-control pos={len(pos)} neg={len(neg)}")
    else:
        picked = labeled
        if max_n is not None and len(picked) > max_n:
            picked = sorted(np.random.RandomState(seed).choice(picked, size=max_n, replace=False).tolist())
        print("design=as-downloaded (prevalence not representative)")
    xs, ys, ages, sexes, missing = [], [], [], [], 0
    for rec, info in tqdm(picked, desc="load-chapman"):
        if not Path(str(rec) + ".mat").exists():
            missing += 1
            continue
        try:
            lead, info2 = read_record(rec)
        except Exception:
            missing += 1
            continue
        lead = resample_to_length(lead, 1000)
        xs.append(zscore(lead))
        ys.append(float(info["y_hyp"]))
        ages.append(info.get("age", np.nan))
        sexes.append(info.get("sex", ""))
    used = len(xs)
    if used == 0:
        raise FileNotFoundError("No local records loaded. Run scripts/download_chapman.py first.")
    return (
        np.stack(xs).astype(np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(ages, dtype=np.float32),
        sexes,
        used,
        missing,
    )


@torch.no_grad()
def predict_proba(model, x, device, temperature: float = 1.0):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x[:, None, :])), batch_size=256)
    outs = []
    for (xb,) in loader:
        logits = model(xb.to(device)) / temperature
        outs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tempscale", action="store_true")
    parser.add_argument("--max-n", type=int, default=None)
    parser.add_argument(
        "--neg-pool",
        type=int,
        default=None,
        help="case-control design: keep ALL positives + this many negatives",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = list_local_records()
    print(f"local records: {len(records)}")
    x, y, ages, sexes, used, missing = load_xy(records, args.max_n, args.neg_pool, args.seed)
    print(f"used={used} missing={missing} pos_rate={y.mean():.4f}")

    ckpt_path = (
        ROOT / "outputs/hyp_v1/best_model_tempscale.pt"
        if args.tempscale
        else ROOT / "outputs/hyp_v1/best_model.pt"
    )
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    temperature = float(blob.get("temperature", 1.0)) if args.tempscale else 1.0
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = TinyECGCNN()
    model.load_state_dict(blob["model"])
    model.to(device)

    scores = predict_proba(model, x, device, temperature=temperature)
    has_pos = len(set(y.tolist())) > 1
    auroc = float(roc_auc_score(y, scores)) if has_pos else float("nan")
    auprc = float(average_precision_score(y, scores)) if has_pos else float("nan")

    auroc_ci = auprc_ci = None
    if has_pos and used >= 50:
        rng = np.random.RandomState(args.seed)
        boot_auroc, boot_auprc = [], []
        for _ in range(1000):
            idx = rng.randint(0, used, size=used)
            if len(set(y[idx].tolist())) < 2:
                continue
            boot_auroc.append(roc_auc_score(y[idx], scores[idx]))
            boot_auprc.append(average_precision_score(y[idx], scores[idx]))
        auroc_ci = [float(np.percentile(boot_auroc, 2.5)), float(np.percentile(boot_auroc, 97.5))]
        auprc_ci = [float(np.percentile(boot_auprc, 2.5)), float(np.percentile(boot_auprc, 97.5))]

    result = {
        "task": "external validation PTB-XL HYP CNN -> Chapman-Shaoxing HYP-aligned (SNOMED LVH/RVH/RAH)",
        "n_used": used,
        "n_missing_files": missing,
        "pos_rate": float(y.mean()),
        "auroc": auroc,
        "auroc_ci95": auroc_ci,
        "auprc": auprc,
        "auprc_ci95": auprc_ci,
        "temperature": temperature,
        "checkpoint": str(ckpt_path),
        "mean_age": float(np.nanmean(ages)) if used else None,
        "pct_male": float(np.mean([s.lower().startswith("m") for s in sexes])) if used else None,
        "caveat": (
            "Positive = expert-labeled SNOMED LVH/RVH/RAH, aligned to the PTB-XL HYP superclass "
            "(no left-atrial overload / septal hypertrophy codes exist in Chapman); "
            "no echo confirmation; interpret as cross-population transportability check."
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "external_metrics.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
