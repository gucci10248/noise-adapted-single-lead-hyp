"""Chapman-Shaoxing 12-lead ECG helpers (PhysioNet ecg-arrhythmia v1.0.0).

Records: WFDB .hea + .mat, 500 Hz, 5000 samples (10 s), 12 leads, mV scale.
Labels: SNOMED CT codes in the `#Dx:` header line.
  LVH = 164873001 (left ventricle hypertrophy).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import wfdb

SNOMED_LVH = "164873001"
SNOMED_RVH = "89792004"
SNOMED_RAH = "446358003"  # right atrial hypertrophy

# Alignment with the PTB-XL HYP superclass = {LVH, RVH, RAH/O, LAO/LAE, SEHYP}
# (scp_statements.csv, diagnostic_class == 'HYP'). Chapman has no left-atrial
# overload or septal hypertrophy codes, so the aligned positive set is:
HYP_POSITIVE_SNOMED = {SNOMED_LVH, SNOMED_RVH, SNOMED_RAH}


def default_chapman_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "chapman"


def list_local_records(root: Path | None = None) -> list[Path]:
    """Return all local record paths (without extension) under WFDBRecords."""
    root = root or default_chapman_root()
    base = root / "WFDBRecords"
    if not base.exists():
        raise FileNotFoundError(f"Missing {base}; run scripts/download_chapman.py first.")
    recs = []
    for hea in sorted(base.rglob("*.hea")):
        recs.append(hea.with_suffix(""))
    return recs


def read_record(rec_path_no_ext: Path) -> tuple[np.ndarray, dict]:
    """Return Lead I signal and metadata {fs, age, sex, snomed, raw_dx}."""
    rec = str(rec_path_no_ext)
    # wfdb header comments carry Age/Sex/Dx; rdsamp reads them via meta['comments']
    signal, meta = wfdb.rdsamp(rec)
    signal = np.asarray(signal, dtype=np.float32)
    names = list(meta.get("sig_name") or [])
    idx = names.index("I") if "I" in names else 0

    info: dict = {"fs": float(meta.get("fs") or 500.0)}
    dx: list[str] = []
    for c in meta.get("comments") or []:
        c = str(c).lstrip("#").strip()  # wfdb strips the leading '#'
        low = c.lower()
        if low.startswith("age:"):
            try:
                info["age"] = float(c.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif low.startswith("sex:"):
            info["sex"] = c.split(":", 1)[1].strip()
        elif low.startswith("dx:"):
            dx = re.split(r"[,\s]+", c.split(":", 1)[1].strip())
    info["snomed"] = dx
    info["y_hyp"] = int(any(d in HYP_POSITIVE_SNOMED for d in dx))
    info["y_lvh"] = int(SNOMED_LVH in dx)
    return signal[:, idx], info
