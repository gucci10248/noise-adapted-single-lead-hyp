"""PyTorch dataset for PTB-XL Lead I binary labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ai_ecg_hf.ptbxl import (
    add_diagnostic_superclass,
    default_ptbxl_root,
    extract_lead_i,
    fold_masks,
    load_metadata,
    read_record,
)


def zscore(x: np.ndarray) -> np.ndarray:
    mu = float(x.mean())
    sd = float(x.std())
    if sd < 1e-6:
        return x - mu
    return (x - mu) / sd


def add_wearable_like_noise(
    x: np.ndarray,
    rng: np.random.Generator,
    sigma: float = 0.05,
    wander_amp: float = 0.1,
    dropout_p: float = 0.2,
    use_gaussian: bool = True,
    use_wander: bool = True,
    use_dropout: bool = True,
) -> np.ndarray:
    """Add wearable-like corruption (Gaussian, baseline wander, dropout).

    Defaults reproduce the v1 training augmentation.  The keyword arguments
    allow severity sweeps and component ablations while keeping the
    random-draw sequence identical to the original implementation when all
    components are enabled.
    """
    y = x.copy()
    if use_gaussian:
        y = y + rng.normal(0.0, sigma, size=y.shape).astype(np.float32)
    if use_wander:
        t = np.arange(y.shape[0], dtype=np.float32)
        y = y + (wander_amp * np.sin(2 * np.pi * t / max(y.shape[0], 1))).astype(np.float32)
    if use_dropout and rng.random() < dropout_p:
        a, b = sorted(rng.integers(0, y.shape[0], size=2))
        if b > a:
            y[a:b] = 0.0
    return y


class PTBXLLeadIBinary(Dataset):
    """Binary task: positive if diagnostic superclass intersects target set."""

    def __init__(
        self,
        split: str = "train",
        root: Path | None = None,
        sampling_rate: int = 100,
        positive_superclasses: tuple[str, ...] = ("HYP",),
        augment: bool = False,
        seed: int = 0,
        require_local: bool = True,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(split)
        self.root = root or default_ptbxl_root()
        self.sampling_rate = sampling_rate
        self.positive = set(positive_superclasses)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        df = add_diagnostic_superclass(load_metadata(self.root), self.root)
        mask = fold_masks(df)[split]
        self.df = df.loc[mask].copy()
        self.df["y"] = self.df["diagnostic_superclass"].apply(
            lambda labs: int(bool(self.positive.intersection(labs)))
        )
        rel_col = "filename_lr" if sampling_rate == 100 else "filename_hr"
        if require_local:
            exists = self.df[rel_col].apply(
                lambda rel: (self.root / f"{rel}.hea").exists()
                and (self.root / f"{rel}.dat").exists()
            )
            n_before = len(self.df)
            self.df = self.df.loc[exists].copy()
            n_after = len(self.df)
            if n_after == 0:
                raise FileNotFoundError(
                    f"No local waveforms for split={split} under {self.root}. "
                    "Run: python scripts/download_ptbxl.py"
                )
            if n_after < n_before:
                print(
                    f"[PTBXLLeadIBinary] split={split}: using {n_after}/{n_before} "
                    "records with local files"
                )
        self.ids = list(self.df.index)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ecg_id = self.ids[idx]
        row = self.df.loc[ecg_id]
        rel = row["filename_lr"] if self.sampling_rate == 100 else row["filename_hr"]
        signal, meta = read_record(self.root, rel)
        lead = zscore(extract_lead_i(signal, meta))
        if self.augment:
            lead = add_wearable_like_noise(lead, self.rng)
        x = torch.from_numpy(lead[None, :].astype(np.float32))
        y = torch.tensor(float(row["y"]), dtype=torch.float32)
        return x, y

    def label_frame(self) -> pd.DataFrame:
        return self.df[["patient_id", "strat_fold", "diagnostic_superclass", "y"]]
