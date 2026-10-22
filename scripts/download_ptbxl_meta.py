"""Quick-fetch PTB-XL metadata CSVs only (seconds), for offline coding."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "ptb-xl"
BASE = "https://physionet.org/files/ptb-xl/1.0.3"
FILES = ("ptbxl_database.csv", "scp_statements.csv", "example_physionet.py")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        out = DEST / name
        url = f"{BASE}/{name}"
        print(f"GET {url}")
        subprocess.run(
            ["curl", "-L", "--fail", "-o", str(out), url],
            check=True,
        )
    print(f"Meta ready in {DEST}")


if __name__ == "__main__":
    main()
