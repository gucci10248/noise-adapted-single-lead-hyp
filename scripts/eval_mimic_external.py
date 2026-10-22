"""External validation of PTB-XL HYP CNN on MIMIC-IV-ECG Lead I (LVH proxy labels).

Requires:
  - outputs/hyp_v1/best_model.pt (or best_model_tempscale.pt)
  - data/mimic-iv-ecg/record_list.csv + machine_measurements.csv
  - local waveform files referenced by record paths

Usage:
  python scripts/eval_mimic_external.py
  python scripts/eval_mimic_external.py --tempscale --max-n 5000
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

from ai_ecg_hf.mimic_ecg import (  # noqa: E402
    build_labeled_index,
    default_mimic_root,
    extract_lead_i_from_record,
    resample_to_length,
    zscore,
)
from ai_ecg_hf.model import TinyECGCNN  # noqa: E402

OUT_DIR = ROOT / "outputs" / "mimic_ext"


def resolve_path_col(df) -> str:
    if "path" in df.columns:
        return "path"
    for c in df.columns:
        if "path" in c.lower():
            return c
    raise RuntimeError(f"No path column in {list(df.columns)}")


def load_xy(df, root: Path, max_n: int | None, seed: int):
    path_col = resolve_path_col(df)
    if max_n is not None and len(df) > max_n:
        df = df.sample(n=max_n, random_state=seed)
    xs, ys, used = [], [], 0
    missing = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="load-mimic"):
        rel = str(row[path_col]).rstrip("/")
        if rel.endswith(".hea") or rel.endswith(".dat"):
            rel = rel.rsplit(".", 1)[0]
        local = root / rel
        if not (local.with_suffix(".hea").exists() or Path(str(local) + ".hea").exists()):
            # wfdb expects path without extension
            hea = Path(str(local) + ".hea")
            if not hea.exists():
                missing += 1
                continue
        try:
            lead, meta = extract_lead_i_from_record(local)
        except Exception:
            missing += 1
            continue
        fs = float(meta.get("fs") or 500.0)
        # map to ~10s @ 100 Hz = 1000 samples
        target_len = 1000
        # if length corresponds to 10s at fs, resample to 1000
        lead = resample_to_length(lead, target_len)
        xs.append(zscore(lead))
        ys.append(float(row["y_lvh"]))
        used += 1
    if used == 0:
        raise FileNotFoundError(
            f"No local waveforms found under {root} (missing≈{missing}). "
            "Download a sample with scripts/download_mimic_iv_ecg.py --sample-n ..."
        )
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32), used, missing


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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = default_mimic_root()
    df = build_labeled_index(root)
    print("label summary:", {k: (float(df[k].mean()) if k.startswith("y_") else None) for k in ["y_lvh", "y_lvh_definite"]})
    print("n rows", len(df), "pos_rate", float(df["y_lvh"].mean()))

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

    x, y, used, missing = load_xy(df, root, args.max_n, args.seed)
    scores = predict_proba(model, x, device, temperature=temperature)
    auroc = float(roc_auc_score(y, scores)) if len(set(y.tolist())) > 1 else float("nan")
    auprc = float(average_precision_score(y, scores)) if len(set(y.tolist())) > 1 else float("nan")

    result = {
        "task": "external validation PTB-XL HYP CNN -> MIMIC LVH proxy",
        "n_used": used,
        "n_missing_files": missing,
        "pos_rate": float(y.mean()),
        "auroc": auroc,
        "auprc": auprc,
        "temperature": temperature,
        "checkpoint": str(ckpt_path),
        "caveat": (
            "Label is machine-report LVH text proxy, not identical to PTB-XL HYP superclass "
            "and not echo-confirmed; interpret as transportability check."
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "external_metrics.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
