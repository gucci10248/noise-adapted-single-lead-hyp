"""Download PTB-XL v1.0.3 via AWS Open Data (fast) or PhysioNet zip fallback.

Preferred:
  aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.3/ data/ptb-xl/

Usage:
  source .venv/bin/activate
  python scripts/download_ptbxl.py              # meta + records100
  python scripts/download_ptbxl.py --full       # also records500
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "ptb-xl"
S3 = "s3://physionet-open/ptb-xl/1.0.3"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def aws_bin() -> str:
    candidates = [
        str(ROOT / ".venv" / "bin" / "aws"),
        shutil.which("aws") or "",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise RuntimeError("aws CLI not found; run: uv pip install awscli")


def sync(rel: str) -> None:
    aws = aws_bin()
    target = DEST if rel == "" else DEST / rel
    target.mkdir(parents=True, exist_ok=True)
    src = S3 if rel == "" else f"{S3}/{rel}"
    dst = str(DEST) if rel == "" else str(DEST / rel)
    run([aws, "s3", "sync", "--no-sign-request", src, dst])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Also sync records500")
    args = parser.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    # Always sync top-level metadata files and records100
    for name in (
        "ptbxl_database.csv",
        "scp_statements.csv",
        "example_physionet.py",
        "RECORDS",
        "LICENSE.txt",
        "SHA256SUMS.txt",
    ):
        run(
            [
                aws_bin(),
                "s3",
                "cp",
                "--no-sign-request",
                f"{S3}/{name}",
                str(DEST / name),
            ]
        )

    print("Syncing records100 (this can take a while)...")
    sync("records100")
    if args.full:
        print("Syncing records500...")
        sync("records500")
    print(f"Ready: {DEST}")
    print("Next: python scripts/plot_lead_i.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(exc, file=sys.stderr)
        sys.exit(1)
