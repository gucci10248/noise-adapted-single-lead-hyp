"""Download the Georgia 12-Lead ECG Challenge Database (Kaggle, CC0).

The Kaggle mirror is a 506 MB zip containing 10,344 WFDB-style records
(``.hea`` + Matlab v4 ``.mat``) from Emory University.  No Kaggle account is
needed for this public dataset.

Usage:
  python scripts/download_georgia.py
  python scripts/download_georgia.py --force
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "georgia"
ZIP = DEST / "georgia.zip"
EXTRACT = DEST / "extracted"
URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "physionet/georgia-12lead-ecg-challenge-database"
)


def download(force: bool = False) -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    if ZIP.exists() and ZIP.stat().st_size > 400_000_000 and not force:
        print(f"using existing {ZIP} ({ZIP.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"downloading {URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "ai_ecg_hf/1.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(ZIP, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    print(f"wrote {ZIP} ({ZIP.stat().st_size / 1e6:.1f} MB)")


def extract(force: bool = False) -> None:
    target = EXTRACT / "Georgia"
    if target.exists() and any(target.glob("*.hea")) and not force:
        print(f"using existing {target}")
        return
    if EXTRACT.exists() and force:
        shutil.rmtree(EXTRACT)
    EXTRACT.mkdir(parents=True, exist_ok=True)
    print(f"extracting {ZIP}")
    with zipfile.ZipFile(ZIP) as zf:
        zf.extractall(EXTRACT)
    print(f"extracted to {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download(args.force)
    extract(args.force)
    n = len(list((EXTRACT / "Georgia").glob("*.hea")))
    print(f"records: {n}")
    if n != 10344:
        raise RuntimeError(f"expected 10,344 headers, found {n}")


if __name__ == "__main__":
    main()
