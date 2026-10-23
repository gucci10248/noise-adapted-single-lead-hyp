"""Baselines for the single-lead HYP task: features, trees, and a 1D ResNet."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def extract_features(lead: np.ndarray) -> np.ndarray:
    """Compact interpretable features from a 10-s, 100-Hz Lead I signal."""
    x = np.asarray(lead, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.ndim != 1:
        raise ValueError(f"expected 1D signal, got {x.shape}")
    n = len(x)
    dx = np.diff(x)
    rms = float(np.sqrt(np.mean(x**2)))
    energy = float(np.mean(x**2))
    med = float(np.median(x))
    q25, q75 = np.percentile(x, [25, 75])
    iqr = float(q75 - q25)
    pos_area = float(np.sum(np.clip(x, 0, None)))
    neg_area = float(-np.sum(np.clip(x, None, 0)))
    r_amp = float(np.max(x))
    s_amp = float(np.min(x))
    rs_ratio = r_amp / max(abs(s_amp), 1e-6)

    # Spectral features on the 100-Hz signal.
    freqs = np.fft.rfftfreq(n, d=0.01)
    power = np.abs(np.fft.rfft(x - x.mean())) ** 2
    total_power = float(power.sum()) + 1e-12
    bands = [(0.5, 4.0), (4.0, 8.0), (8.0, 16.0), (16.0, 32.0), (32.0, 50.0)]
    band_power = []
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        band_power.append(float(power[m].sum() / total_power))
    p = power / total_power
    spectral_entropy = float(-np.sum(p[p > 0] * np.log(p[p > 0]))) / max(np.log(len(p)), 1e-12)
    dominant_freq = float(freqs[int(np.argmax(power))])

    # R-peak proxies; lead I is z-scored, so prominence is in SD units.
    peaks, props = find_peaks(x, distance=20, prominence=0.3)
    n_peaks = int(len(peaks))
    hr = float(60.0 * n_peaks / (n * 0.01))
    if n_peaks >= 2:
        rr = np.diff(peaks) * 0.01
        rr_mean = float(rr.mean())
        rr_std = float(rr.std())
        rr_cv = rr_std / max(rr_mean, 1e-6)
        peak_amp_mean = float(np.mean(props.get("prominences", [0.0])))
        peak_amp_max = float(np.max(props.get("prominences", [0.0])))
    else:
        rr_mean = rr_std = rr_cv = peak_amp_mean = peak_amp_max = 0.0

    feats = [
        float(x.mean()),
        float(x.std()),
        float(x.min()),
        float(x.max()),
        float(np.ptp(x)),
        med,
        iqr,
        float(skew(x)),
        float(kurtosis(x)),
        *[float(v) for v in np.percentile(x, [5, 25, 75, 95])],
        float(np.mean(x > 0)),
        float(np.mean(np.abs(dx) > 0.1)),
        rms,
        energy,
        float(np.mean(np.abs(x))),
        float(np.max(np.abs(x))),
        float(np.mean(np.abs(dx))),
        float(np.std(dx)),
        float(np.max(np.abs(dx))),
        *band_power,
        spectral_entropy,
        dominant_freq,
        float(n_peaks),
        hr,
        rr_mean,
        rr_std,
        rr_cv,
        peak_amp_mean,
        peak_amp_max,
        pos_area,
        neg_area,
        rs_ratio,
    ]
    return np.asarray(feats, dtype=np.float32)


FEATURE_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "ptp",
    "median",
    "iqr",
    "skew",
    "kurtosis",
    "p05",
    "p25",
    "p75",
    "p95",
    "frac_positive",
    "frac_large_derivative",
    "rms",
    "energy",
    "mean_abs",
    "max_abs",
    "mean_abs_derivative",
    "std_derivative",
    "max_abs_derivative",
    "band_0p5_4",
    "band_4_8",
    "band_8_16",
    "band_16_32",
    "band_32_50",
    "spectral_entropy",
    "dominant_freq",
    "n_peaks",
    "heart_rate",
    "rr_mean",
    "rr_std",
    "rr_cv",
    "peak_prominence_mean",
    "peak_prominence_max",
    "positive_area",
    "negative_area",
    "rs_ratio",
]


def build_feature_matrix(x: np.ndarray) -> np.ndarray:
    return np.stack([extract_features(row) for row in x]).astype(np.float32)


def fit_feature_logistic(
    x_train: np.ndarray, y_train: np.ndarray
) -> Pipeline:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000, class_weight="balanced", C=1.0, solver="lbfgs"
                ),
            ),
        ]
    )
    model.fit(build_feature_matrix(x_train), y_train)
    return model


def fit_feature_trees(x_train: np.ndarray, y_train: np.ndarray):
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
    )
    model.fit(build_feature_matrix(x_train), y_train)
    return model


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.short = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.act(h + self.short(x))


class LeadIResNet(nn.Module):
    """A deliberately stronger 1D ResNet baseline (~1.1M parameters)."""

    def __init__(self, n_classes: int = 1) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.layer1 = ResidualBlock(64, 64)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        return self.head(h).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
