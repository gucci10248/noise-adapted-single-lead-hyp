"""Train the three single-component augmentation models for the ablation.

The clean (``hyp_v1``) and full-augmentation (``hyp_v1_aug``) checkpoints already
exist; this script trains the remaining Gaussian-only, wander-only, and
dropout-only models with the same split, seed, and architecture.

Usage:
  python scripts/train_aug_ablation.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = ["gaussian", "wander", "dropout"]


def main() -> None:
    for mode in MODES:
        tag = f"hyp_abl_{mode}"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train_hyp.py"),
            "--aug-mode",
            mode,
            "--tag",
            tag,
            "--seed",
            "42",
        ]
        print("RUN", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
