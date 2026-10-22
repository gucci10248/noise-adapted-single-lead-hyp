"""PTB-XL loaders: metadata, Lead I extraction, official fold splits."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb


LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def default_ptbxl_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "ptb-xl"


def load_metadata(root: Path | None = None) -> pd.DataFrame:
    root = root or default_ptbxl_root()
    meta_path = root / "ptbxl_database.csv"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {meta_path}. Run: python scripts/download_ptbxl.py"
        )
    df = pd.read_csv(meta_path, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    return df


def load_scp_statements(root: Path | None = None) -> pd.DataFrame:
    root = root or default_ptbxl_root()
    path = root / "scp_statements.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path, index_col=0)


def aggregate_diagnostic_superclass(
    scp_codes: dict, agg_df: pd.DataFrame
) -> list[str]:
    labels: list[str] = []
    for key in scp_codes:
        if key in agg_df.index:
            labels.append(str(agg_df.loc[key, "diagnostic_class"]))
    return sorted(set(labels))


def add_diagnostic_superclass(df: pd.DataFrame, root: Path | None = None) -> pd.DataFrame:
    agg = load_scp_statements(root)
    agg = agg[agg["diagnostic"] == 1]
    out = df.copy()
    out["diagnostic_superclass"] = out["scp_codes"].apply(
        lambda d: aggregate_diagnostic_superclass(d, agg)
    )
    return out


def read_record(
    root: Path,
    relative_stem: str,
) -> tuple[np.ndarray, dict]:
    """Read one WFDB record. relative_stem like 'records100/00000/00001_lr'."""
    signal, meta = wfdb.rdsamp(str(root / relative_stem))
    return np.asarray(signal, dtype=np.float32), meta


def extract_lead_i(signal: np.ndarray, meta: dict) -> np.ndarray:
    names = list(meta.get("sig_name") or LEAD_NAMES)
    if "I" in names:
        idx = names.index("I")
    else:
        idx = 0
    return signal[:, idx].astype(np.float32)


def load_lead_i(
    root: Path | None = None,
    ecg_id: int = 1,
    sampling_rate: int = 100,
) -> tuple[np.ndarray, dict, pd.Series]:
    """Load a single Lead I trace + row metadata."""
    root = root or default_ptbxl_root()
    df = load_metadata(root)
    if ecg_id not in df.index:
        raise KeyError(f"ecg_id={ecg_id} not in metadata")
    row = df.loc[ecg_id]
    rel = row["filename_lr"] if sampling_rate == 100 else row["filename_hr"]
    signal, meta = read_record(root, rel)
    lead_i = extract_lead_i(signal, meta)
    return lead_i, meta, row


def fold_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Official PTB-XL split: folds 1-8 train, 9 val, 10 test."""
    folds = df["strat_fold"].to_numpy()
    return {
        "train": folds <= 8,
        "val": folds == 9,
        "test": folds == 10,
    }
