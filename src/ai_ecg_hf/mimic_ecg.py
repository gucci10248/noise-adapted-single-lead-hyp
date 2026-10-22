"""MIMIC-IV-ECG helpers: metadata, LVH proxy labels, Lead I loading."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb


def default_mimic_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "mimic-iv-ecg"


# Machine-report phrases used as HYP/LVH proxy (case-insensitive).
LVH_POSITIVE_PATTERNS = (
    r"\bleft ventricular hypertrophy\b",
    r"\blvh\b",
    r"\bventricular hypertrophy\b",
)
# Softer / uncertain mentions still count as positive for screening-style proxy,
# but we keep a separate "definite" flag for sensitivity analyses.
LVH_DEFINITE_PATTERNS = (
    r"\bleft ventricular hypertrophy\b",
    r"(?<!possible )(?<!cannot rule out )\blvh\b",
)


def _compile(pats: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in pats), flags=re.IGNORECASE)


_POS_RE = _compile(LVH_POSITIVE_PATTERNS)
_DEF_RE = _compile(LVH_DEFINITE_PATTERNS)


def load_record_list(root: Path | None = None) -> pd.DataFrame:
    root = root or default_mimic_root()
    path = root / "record_list.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download MIMIC-IV-ECG metadata first "
            "(see docs/MIMIC_external_validation.md)."
        )
    return pd.read_csv(path)


def load_machine_measurements(root: Path | None = None) -> pd.DataFrame:
    root = root or default_mimic_root()
    path = root / "machine_measurements.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path, low_memory=False)


def combine_reports(row: pd.Series) -> str:
    parts: list[str] = []
    for col in row.index:
        if str(col).startswith("report_"):
            val = row[col]
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())
    return " | ".join(parts)


def label_lvh_from_reports(mm: pd.DataFrame) -> pd.DataFrame:
    """Add proxy labels from machine-generated report text."""
    out = mm.copy()
    out["report_text"] = out.apply(combine_reports, axis=1)
    out["y_lvh"] = out["report_text"].apply(lambda t: int(bool(_POS_RE.search(t or ""))))
    out["y_lvh_definite"] = out["report_text"].apply(
        lambda t: int(bool(_DEF_RE.search(t or "")) and "possible" not in (t or "").lower())
    )
    return out


def build_labeled_index(root: Path | None = None) -> pd.DataFrame:
    """Merge record paths with LVH proxy labels (one row per study_id)."""
    root = root or default_mimic_root()
    records = load_record_list(root)
    mm = label_lvh_from_reports(load_machine_measurements(root))
    # keep essential columns
    keep = [c for c in mm.columns if c in {"subject_id", "study_id", "ecg_time", "y_lvh", "y_lvh_definite", "report_text"}]
    labeled = records.merge(mm[keep], on=["subject_id", "study_id"], how="inner", suffixes=("", "_mm"))
    return labeled


def extract_lead_i_from_record(record_path: Path) -> tuple[np.ndarray, dict]:
    """Read WFDB record and return Lead I as float32 vector."""
    signal, meta = wfdb.rdsamp(str(record_path))
    signal = np.asarray(signal, dtype=np.float32)
    names = list(meta.get("sig_name") or [])
    if "I" in names:
        idx = names.index("I")
    else:
        idx = 0
    return signal[:, idx], meta


def zscore(x: np.ndarray) -> np.ndarray:
    mu = float(x.mean())
    sd = float(x.std())
    if sd < 1e-6:
        return x - mu
    return ((x - mu) / sd).astype(np.float32)


def resample_to_length(x: np.ndarray, target_len: int = 1000) -> np.ndarray:
    """Linear resample to fixed length (default 10s @ 100 Hz to match PTB-XL)."""
    if len(x) == target_len:
        return x.astype(np.float32)
    old = np.linspace(0.0, 1.0, num=len(x), dtype=np.float64)
    new = np.linspace(0.0, 1.0, num=target_len, dtype=np.float64)
    return np.interp(new, old, x.astype(np.float64)).astype(np.float32)


def summarize_labels(df: pd.DataFrame) -> dict[str, float | int]:
    return {
        "n": int(len(df)),
        "n_subjects": int(df["subject_id"].nunique()) if "subject_id" in df else -1,
        "pos_rate_lvh": float(df["y_lvh"].mean()) if len(df) else float("nan"),
        "pos_rate_definite": float(df["y_lvh_definite"].mean()) if len(df) else float("nan"),
    }
