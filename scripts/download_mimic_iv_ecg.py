"""Download MIMIC-IV-ECG metadata (+ optional waveform subset).

PhysioNet currently returns 403 without a logged-in account for this project,
even though the license is open. Prefer one of:

1) Browser: log into physionet.org, download files into data/mimic-iv-ecg/
2) CLI with credentials:
   export PHYSIONET_USERNAME=...
   export PHYSIONET_PASSWORD=...
   python scripts/download_mimic_iv_ecg.py --meta-only

Usage:
  python scripts/download_mimic_iv_ecg.py --meta-only
  python scripts/download_mimic_iv_ecg.py --sample-n 2000 --seed 42
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "mimic-iv-ecg"
BASE = "https://physionet.org/files/mimic-iv-ecg/1.0"
META = (
    "record_list.csv",
    "machine_measurements.csv",
    "machine_measurements_data_dictionary.csv",
    "waveform_note_links.csv",
    "LICENSE.txt",
    "SHA256SUMS.txt",
)


def curl_auth(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    user = os.environ.get("PHYSIONET_USERNAME", "").strip()
    password = os.environ.get("PHYSIONET_PASSWORD", "").strip()
    cmd = ["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-o", str(out), url]
    if user and password:
        cmd[1:1] = ["-u", f"{user}:{password}"]
    print("+", " ".join(cmd[:6]), "...", url.split("/")[-1])
    subprocess.run(cmd, check=True)


def download_meta() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in META:
        curl_auth(f"{BASE}/{name}", DEST / name)
    print(f"Meta ready under {DEST}")


def download_sample_waveforms(sample_n: int, seed: int) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from ai_ecg_hf.mimic_ecg import build_labeled_index

    df = build_labeled_index(DEST)
    # subject-level stratified-ish sample: take up to sample_n studies
    import pandas as pd

    pos = df[df["y_lvh"] == 1]
    neg = df[df["y_lvh"] == 0]
    n_pos = min(len(pos), max(sample_n // 5, 1))  # enrich positives a bit but keep realism
    n_neg = min(len(neg), sample_n - n_pos)
    take = pd.concat(
        [pos.sample(n=n_pos, random_state=seed), neg.sample(n=n_neg, random_state=seed)],
        axis=0,
    ).sample(frac=1.0, random_state=seed)

    # path column name in record_list is typically `path` or similar
    path_col = "path" if "path" in take.columns else None
    if path_col is None:
        for c in take.columns:
            if "path" in c.lower() or "file" in c.lower():
                path_col = c
                break
    if path_col is None:
        raise RuntimeError(f"Cannot find waveform path column in {list(take.columns)}")

    out_list = DEST / "sample_study_paths.txt"
    paths = take[path_col].astype(str).tolist()
    out_list.write_text("\n".join(paths) + "\n", encoding="utf-8")
    print(f"Sampling {len(paths)} studies -> {out_list}")

    for i, rel in enumerate(paths, start=1):
        rel = rel.rstrip("/")
        # WFDB typically needs .hea + .dat without extension in path field
        for ext in (".hea", ".dat"):
            # path in CSV may already be directory/stem
            stem = rel
            if stem.endswith(".hea") or stem.endswith(".dat"):
                stem = stem.rsplit(".", 1)[0]
            url = f"{BASE}/{stem}{ext}"
            local = DEST / f"{stem}{ext}"
            if local.exists() and local.stat().st_size > 0:
                continue
            try:
                curl_auth(url, local)
            except subprocess.CalledProcessError as exc:
                print(f"WARN failed {url}: {exc}")
        if i % 50 == 0:
            print(f"waveforms progress {i}/{len(paths)}")
    print("Sample waveform download pass finished.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-only", action="store_true")
    parser.add_argument("--sample-n", type=int, default=0, help="Also download N labeled waveforms")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.environ.get("PHYSIONET_USERNAME") or not os.environ.get("PHYSIONET_PASSWORD"):
        print(
            "NOTE: PHYSIONET_USERNAME / PHYSIONET_PASSWORD not set.\n"
            "If curl gets 403, log into https://physionet.org in a browser and\n"
            "manually download record_list.csv + machine_measurements.csv into:\n"
            f"  {DEST}\n"
        )

    try:
        download_meta()
    except subprocess.CalledProcessError:
        print("Meta download failed (likely 403). Use browser download or set credentials.")
        sys.exit(2)

    if args.sample_n > 0:
        download_sample_waveforms(args.sample_n, args.seed)
    elif not args.meta-only:
        print("Meta done. Re-run with --sample-n 2000 after credentials work to fetch waveforms.")


if __name__ == "__main__":
    main()
