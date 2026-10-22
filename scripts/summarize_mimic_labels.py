"""Summarize MIMIC-IV-ECG LVH proxy labels from machine reports (no waveforms needed)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_ecg_hf.mimic_ecg import build_labeled_index, summarize_labels  # noqa: E402

OUT = ROOT / "outputs" / "mimic_ext" / "label_summary.json"


def main() -> None:
    df = build_labeled_index()
    summary = summarize_labels(df)
    # examples
    pos_ex = (
        df.loc[df["y_lvh"] == 1, "report_text"].dropna().head(3).tolist()
        if "report_text" in df.columns
        else []
    )
    payload = {
        "label_definition": (
            "Positive if machine report text matches LVH/left ventricular hypertrophy "
            "(proxy for PTB-XL HYP external check; not echo-confirmed)."
        ),
        "summary": summary,
        "positive_report_examples": pos_ex,
        "columns": list(df.columns),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
