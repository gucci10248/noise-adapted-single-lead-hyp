"""Download PTB-XL records100 folder-by-folder via aws s3 sync (resumable).

Usage:
  source .venv/bin/activate
  nohup caffeinate -i python scripts/sync_records100.py >> data/s3_sync_wrapper.log 2>&1 &
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "ptb-xl" / "records100"
LOG = ROOT / "data" / "s3_sync_records100.log"
AWS = ROOT / ".venv" / "bin" / "aws"
S3 = "s3://physionet-open/ptb-xl/1.0.3/records100"
FOLDERS = [f"{i:05d}" for i in range(0, 22000, 1000)]
TARGET = 21700


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def dat_count(path: Path | None = None) -> int:
    root = path or DEST
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob("*.dat"))


def sync_folder(folder: str) -> int:
    local = DEST / folder
    local.mkdir(parents=True, exist_ok=True)
    before = dat_count(local)
    cmd = [
        str(AWS),
        "s3",
        "sync",
        "--no-sign-request",
        "--only-show-errors",
        f"{S3}/{folder}/",
        str(local),
    ]
    for attempt in range(1, 4):
        try:
            subprocess.run(cmd, check=True, timeout=3600)
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            log(f"folder={folder} attempt={attempt} error={exc}")
            time.sleep(3 * attempt)
    after = dat_count(local)
    log(f"folder={folder} dat={after} (+{after - before})")
    return after


def main() -> None:
    if not AWS.exists():
        raise SystemExit(f"aws CLI missing: {AWS}")
    DEST.mkdir(parents=True, exist_ok=True)
    log("start aws folder-wise sync_records100")
    round_id = 1
    while True:
        log(f"round={round_id} local_dat={dat_count()}")
        for folder in FOLDERS:
            try:
                sync_folder(folder)
            except Exception as exc:  # noqa: BLE001
                log(f"ERROR folder={folder}: {exc}")
        n = dat_count()
        log(f"TOTAL dat={n} / 21799")
        if n >= TARGET:
            log("DONE")
            return
        round_id += 1
        time.sleep(5)


if __name__ == "__main__":
    main()
