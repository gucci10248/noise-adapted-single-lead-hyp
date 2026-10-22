"""Plot one PTB-XL Lead I trace to figures/lead_i_demo.png."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_ecg_hf.ptbxl import load_lead_i  # noqa: E402


def main() -> None:
    lead_i, meta, row = load_lead_i(ecg_id=1, sampling_rate=100)
    fs = float(meta.get("fs") or 100.0)
    t = np.arange(lead_i.shape[0], dtype=np.float32) / fs

    out = ROOT / "figures" / "lead_i_demo.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, lead_i, lw=0.8, color="#1f4e79")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Lead I (mV)")
    ax.set_title(
        f"PTB-XL ecg_id=1  patient_id={int(row['patient_id'])}  fs={fs:.0f} Hz"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved: {out}")
    print(f"shape={lead_i.shape}  mean={lead_i.mean():.4f}  std={lead_i.std():.4f}")


if __name__ == "__main__":
    main()
