"""Smoke-check the local AI-ECG environment (Mac MPS / packages)."""

from __future__ import annotations


def main() -> None:
    import platform

    import matplotlib
    import numpy as np
    import pandas as pd
    import sklearn
    import torch
    import wfdb

    print("=== ai_ecg_hf environment ===")
    print(f"platform : {platform.platform()}")
    print(f"python   : {platform.python_version()}")
    print(f"torch    : {torch.__version__}")
    print(f"mps      : {torch.backends.mps.is_available()}")
    print(f"wfdb     : {wfdb.__version__}")
    print(f"pandas   : {pd.__version__}")
    print(f"numpy    : {np.__version__}")
    print(f"sklearn  : {sklearn.__version__}")
    print(f"matplotlib: {matplotlib.__version__}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    x = torch.randn(4, 1, 1000, device=device)
    y = torch.nn.Conv1d(1, 8, kernel_size=7, padding=3).to(device)(x)
    print(f"device   : {device}")
    print(f"conv1d ok: {tuple(y.shape)}")
    print("ALL OK")


if __name__ == "__main__":
    main()
