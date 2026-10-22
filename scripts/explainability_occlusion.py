"""Occlusion-based explainability for the noise-augmented HYP model.

For each cohort, 100 HYP-positive and 100 HYP-negative records are randomly
selected. A 1-s window is zeroed at 10 positions, and the drop in predicted
probability is averaged by class. The analysis is descriptive and is intended
to show which parts of the Lead I waveform drive the model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ai_ecg_hf.model import TinyECGCNN  # noqa: E402
from analyze_chapman_external import load_ptbxl_test  # noqa: E402
from eval_georgia_external import load_georgia  # noqa: E402

OUT = ROOT / "outputs" / "explainability"
FIG = ROOT / "figures"
CKPT = ROOT / "outputs" / "hyp_v1_aug" / "best_model.pt"
SEED = 42
N_PER_CLASS = 100
WINDOW = 50  # 0.5 s at 100 Hz
N_WINDOWS = 20


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@torch.no_grad()
def occlusion_profile(model, x, idx, device):
    """Return [n_records, n_windows] probability drops."""
    model.eval()
    drops = np.zeros((len(idx), N_WINDOWS), dtype=np.float32)
    for row, i in enumerate(idx):
        base = x[i].copy()
        variants = np.repeat(base[None, :], N_WINDOWS, axis=0)
        for w in range(N_WINDOWS):
            a = w * WINDOW
            b = a + WINDOW
            variants[w, a:b] = 0.0
        xb = torch.from_numpy(variants[:, None, :]).to(device)
        p = torch.sigmoid(model(xb)).cpu().numpy()
        base_p = p.mean()  # all windows share the same base except the occluded one
        # Recompute the unoccluded baseline once per record (cheap).
        xb0 = torch.from_numpy(base[None, None, :]).to(device)
        p0 = float(torch.sigmoid(model(xb0)).cpu().numpy()[0])
        drops[row] = p0 - p
    return drops


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = TinyECGCNN()
    model.load_state_dict(blob["model"])
    model.to(device).eval()

    cohorts = {}
    pt = load_ptbxl_test(seed=SEED)
    cohorts["PTB-XL test"] = (pt["x"], np.asarray(pt["y"]).astype(int))
    ga = load_georgia()
    cohorts["Georgia (Emory)"] = (ga["x"], np.asarray(ga["y"]).astype(int))

    rng = np.random.default_rng(SEED)
    payload = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, (name, (x, y)) in zip(axes, cohorts.items()):
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        pos = rng.choice(pos, size=min(N_PER_CLASS, len(pos)), replace=False)
        neg = rng.choice(neg, size=min(N_PER_CLASS, len(neg)), replace=False)
        d_pos = occlusion_profile(model, x, pos, device)
        d_neg = occlusion_profile(model, x, neg, device)
        centers = (np.arange(N_WINDOWS) + 0.5) * (WINDOW / 100.0)
        mp, sp = d_pos.mean(0), d_pos.std(0) / np.sqrt(len(pos))
        mn, sn = d_neg.mean(0), d_neg.std(0) / np.sqrt(len(neg))
        ax.plot(centers, mp, marker="o", lw=2, color="C1", label="HYP positive")
        ax.fill_between(centers, mp - sp, mp + sp, color="C1", alpha=0.15)
        ax.plot(centers, mn, marker="o", lw=2, color="C0", label="HYP negative")
        ax.fill_between(centers, mn - sn, mn + sn, color="C0", alpha=0.15)
        ax.axhline(0, lw=1, ls=":", color="0.5")
        ax.set_xlabel("Occlusion window centre (s)")
        ax.set_ylabel("Mean probability drop")
        ax.set_title(name)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        payload[name] = {
            "n_pos": int(len(pos)),
            "n_neg": int(len(neg)),
            "window_centres_s": [float(c) for c in centers],
            "pos_mean": [float(v) for v in mp],
            "pos_sem": [float(v) for v in sp],
            "neg_mean": [float(v) for v in mn],
            "neg_sem": [float(v) for v in sn],
        }
        print(f"{name}: n_pos={len(pos)} n_neg={len(neg)}")
        print("  pos drop:", np.round(mp, 4))
        print("  neg drop:", np.round(mn, 4))

    fig.tight_layout()
    fig.savefig(FIG / "explainability_occlusion.png", dpi=300)
    fig.savefig(FIG / "explainability_occlusion.pdf")
    plt.close(fig)
    (OUT / "occlusion_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"wrote {OUT / 'occlusion_summary.json'}")
    print(f"wrote {FIG / 'explainability_occlusion.png'}")


if __name__ == "__main__":
    main()
