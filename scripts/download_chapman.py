"""Download Chapman-Shaoxing 12-lead ECG database (PhysioNet ecg-arrhythmia 1.0.0).

Open access (CC BY 4.0). Uses the S3 mirror (physionet-open bucket), which is
much faster than the web frontend. Records live under
WFDBRecords/<46 groups>/<10 sub>/<~100 recs> as .hea + .mat pairs.

Usage:
  # Phase 1: header-only crawl of all subfolders (fast, builds the label index)
  python scripts/download_chapman.py --hea-only --all --workers 32

  # Phase 2: full waveforms for specific subfolders
  python scripts/download_chapman.py --subfolders 10 --workers 32

  # Phase 2b: waveforms for an explicit record list (built from hea labels)
  python scripts/download_chapman.py --from-list data/chapman/eval_sample.txt --workers 32
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_ecg_hf.chapman import default_chapman_root  # noqa: E402

S3_BUCKET = "https://physionet-open.s3.amazonaws.com"
PREFIX = "ecg-arrhythmia/1.0.0"
RECORDS_TXT = f"{S3_BUCKET}/{PREFIX}/RECORDS"
LIST_URL = f"{S3_BUCKET}?list-type=2&prefix={PREFIX}/&delimiter=/"


def http_get(url: str, dest: Path, retries: int = 3) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True  # resume: skip
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ai_ecg_hf/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                f.write(r.read())
            return True
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return False


def fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "ai_ecg_hf/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode("utf-8", "ignore")


def fetch_subfolder_names() -> list[str]:
    text = fetch_text(RECORDS_TXT).lstrip("\ufeff")
    return [ln.strip().rstrip("/") for ln in text.splitlines() if ln.strip()]


def fetch_record_names(subdir: str) -> list[str]:
    """List record basenames in a subfolder via S3 ListObjectsV2."""
    prefix = f"{PREFIX}/{subdir}/"
    url = f"{S3_BUCKET}?list-type=2&prefix={urllib.parse.quote(prefix)}&max-keys=1000"
    xml = fetch_text(url)
    root = ET.fromstring(xml)
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    names = set()
    for key in root.findall("s3:Contents/s3:Key", ns):
        k = key.text or ""
        if k.endswith(".hea"):
            names.add(Path(k).stem)
    return sorted(names)


def download_subfolder(sub: str, root: Path, workers: int, hea_only: bool) -> tuple[int, int]:
    names = fetch_record_names(sub)
    dest_dir = root / sub
    dest_dir.mkdir(parents=True, exist_ok=True)
    exts = (".hea",) if hea_only else (".hea", ".mat")
    tasks = []
    for name in names:
        for ext in exts:
            tasks.append((f"{S3_BUCKET}/{PREFIX}/{sub}/{name}{ext}", dest_dir / f"{name}{ext}"))
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda t: http_get(*t), tasks))
    ok = sum(results)
    return ok, len(results) - ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="all subfolders")
    ap.add_argument("--subfolders", type=int, default=1, help="first N subfolders")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--hea-only", action="store_true", help="download only .hea headers")
    ap.add_argument("--from-list", type=str, default=None, help="file with one record path per line, e.g. WFDBRecords/01/011/JS00105")
    args = ap.parse_args()

    root = default_chapman_root()
    root.mkdir(parents=True, exist_ok=True)

    if args.from_list:
        recs = [ln.strip() for ln in Path(args.from_list).read_text().splitlines() if ln.strip()]
        tasks = []
        for rel in recs:
            for ext in (".hea", ".mat"):
                dest = root / f"{rel}{ext}"
                tasks.append((f"{S3_BUCKET}/{PREFIX}/{rel}{ext}", dest))
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(lambda t: http_get(*t), tasks))
        print(f"from-list: requested={len(recs)} files_ok={sum(results)} files_fail={len(results)-sum(results)}")
        sys.exit(0 if sum(results) == len(results) else 1)

    subs = fetch_subfolder_names()
    if not args.all:
        subs = subs[: args.subfolders]
    print(f"subfolders: {len(subs)} (hea_only={args.hea_only})")

    n_ok = n_fail = 0
    for i, sub in enumerate(subs, 1):
        try:
            ok, fail = download_subfolder(sub, root, args.workers, args.hea_only)
        except Exception as e:
            print(f"[skip] {sub}: {e}")
            continue
        n_ok += ok
        n_fail += fail
        if i % 10 == 0 or i == len(subs):
            print(f"[{i}/{len(subs)}] cum_ok={n_ok} cum_fail={n_fail}")
    print(f"TOTAL ok={n_ok} fail={n_fail}")


if __name__ == "__main__":
    main()
