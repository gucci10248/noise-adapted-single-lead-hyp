"""Collect the frozen results into one comparison table.

Reads the JSON outputs of the internal test, Chapman and Georgia evaluations and
writes ``outputs/results_table.csv`` + ``outputs/results_table.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def row(cohort: str, model: str, m: dict, ece: float | None, note: str) -> dict:
    n = int(m.get("n", m.get("n_used")))
    n_pos = int(m.get("n_pos", round(m["pos_rate"] * n)))
    return {
        "cohort": cohort,
        "model": model,
        "n": n,
        "n_pos": n_pos,
        "prevalence": round(m["pos_rate"], 4),
        "auroc": round(m["auroc"], 4),
        "auroc_95ci": f"{m['auroc_ci95'][0]:.3f}-{m['auroc_ci95'][1]:.3f}",
        "auprc": round(m["auprc"], 4),
        "auprc_95ci": f"{m['auprc_ci95'][0]:.3f}-{m['auprc_ci95'][1]:.3f}",
        "ece": None if ece is None else round(ece, 4),
        "note": note,
    }


def main() -> None:
    rows = []

    # Internal PTB-XL test
    ch_clean = load(OUT / "chapman_ext" / "external_analysis.json")
    clean_internal = ch_clean["discrimination"]["ptbxl_test"]
    aug_v11 = load(OUT / "hyp_v1_aug" / "metrics_v1_1.json")
    aug_internal = aug_v11["discrimination"]["test_raw"]
    for m, label, ece in [
        (clean_internal, "clean v1", ch_clean["calibration"]["ece_internal_test_raw"]),
        (aug_internal, "noise-aug v1.3", aug_v11["calibration"]["ece_test"]),
    ]:
        rows.append(
            row(
                "PTB-XL test",
                label,
                m,
                ece,
                "internal, official fold 10",
            )
        )

    # Chapman
    ch_clean = load(OUT / "chapman_ext" / "external_analysis.json")
    ch_aug = load(OUT / "hyp_v1_aug" / "external_metrics.json")
    rows.append(
        row(
            "Chapman-Shaoxing",
            "clean v1",
            ch_clean["discrimination"]["chapman"],
            ch_clean["calibration"]["ece_raw"],
            "case-control, expert SNOMED labels",
        )
    )
    rows.append(
        row(
            "Chapman-Shaoxing",
            "noise-aug v1.3",
            ch_aug,
            ch_aug["ece_raw"],
            "case-control, expert SNOMED labels",
        )
    )

    # Georgia
    ga_clean = load(OUT / "georgia_ext_clean" / "external_metrics.json")
    ga_aug = load(OUT / "georgia_ext" / "external_metrics.json")
    rows.append(
        row(
            "Georgia (Emory)",
            "clean v1",
            ga_clean,
            ga_clean["ece_raw"],
            "natural prevalence, report-derived SNOMED labels",
        )
    )
    rows.append(
        row(
            "Georgia (Emory)",
            "noise-aug v1.3",
            ga_aug,
            ga_aug["ece_raw"],
            "natural prevalence, report-derived SNOMED labels",
        )
    )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results_table.csv", index=False)

    # markdown
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in cols) + " |")
    (OUT / "results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(df.to_string(index=False))
    print(f"wrote {OUT / 'results_table.csv'}")


if __name__ == "__main__":
    main()
