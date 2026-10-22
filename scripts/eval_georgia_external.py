"""External validation on the Georgia 12-Lead ECG Challenge Database (Emory).

The Kaggle mirror (``physionet/georgia-12lead-ecg-challenge-database``, CC0) is
a 10,344-record subset of the PhysioNet/Computing in Cardiology Challenge 2020.
Headers carry SNOMED CT ``#Dx`` codes, age and sex.  Signals are Matlab v4
``.mat`` files (12 x 5000 int16, gain 4880/mV).

Primary label alignment (same as Chapman): LVH 164873001 + RVH 89792004 +
RAH 446358003 -> HYP.

Usage:
  python scripts/eval_georgia_external.py
  python scripts/eval_georgia_external.py --ckpt outputs/hyp_v1/best_model.pt --tag hyp_v1_clean
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ai_ecg_hf.mimic_ecg import resample_to_length, zscore  # noqa: E402
from ai_ecg_hf.model import TinyECGCNN  # noqa: E402

from analyze_chapman_external import (  # noqa: E402
    bootstrap_metrics,
    expected_calibration_error,
    fit_age_sex_baseline,
    operating_point,
    paired_delta_auroc,
    plot_calibration,
    plot_roc_pr,
    plot_subgroups,
    ppv_npv_at_prevalence,
    predict_logits,
)

DATA_DIR = ROOT / "data" / "georgia" / "extracted" / "Georgia"
FIG_DIR = ROOT / "figures"
HYP_CODES = {"164873001", "89792004", "446358003"}
LVH_CODE = "164873001"
RVH_CODE = "89792004"
RAH_CODE = "446358003"
GAIN_PER_MV = 4880.0
SEED = 42


def parse_header(path: Path) -> dict:
    info = {"record": path.stem, "age": np.nan, "sex": "", "dx": []}
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("#Age:"):
            try:
                info["age"] = float(line.split(":", 1)[1])
            except ValueError:
                pass
        elif line.startswith("#Sex:"):
            info["sex"] = line.split(":", 1)[1].strip()
        elif line.startswith("#Dx:"):
            info["dx"] = [d for d in re.split(r"[,\s]+", line.split(":", 1)[1].strip()) if d]
    info["y_hyp"] = int(bool(HYP_CODES & set(info["dx"])))
    info["y_lvh"] = int(LVH_CODE in info["dx"])
    info["y_rvh"] = int(RVH_CODE in info["dx"])
    info["y_rah"] = int(RAH_CODE in info["dx"])
    return info


def load_georgia() -> dict:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Missing {DATA_DIR}; download the Kaggle Georgia dataset first.")
    xs, ys, lvh, rvh, ages, sexes, names = [], [], [], [], [], [], []
    missing = 0
    for hea in tqdm(sorted(DATA_DIR.glob("*.hea")), desc="load-georgia", leave=False):
        info = parse_header(hea)
        mat = hea.with_suffix(".mat")
        if not mat.exists():
            missing += 1
            continue
        try:
            arr = loadmat(str(mat))["val"]
        except Exception:
            missing += 1
            continue
        if arr.shape[0] != 12 and arr.shape[1] == 12:
            arr = arr.T
        lead = arr[0].astype(np.float32) / GAIN_PER_MV
        lead = zscore(resample_to_length(lead, 1000))
        xs.append(lead.astype(np.float32))
        ys.append(float(info["y_hyp"]))
        lvh.append(float(info["y_lvh"]))
        rvh.append(float(info["y_rvh"]))
        ages.append(float(info["age"]))
        sexes.append(str(info["sex"]))
        names.append(info["record"])
    print(f"georgia loaded={len(xs)} missing={missing}")
    return {
        "x": np.stack(xs).astype(np.float32),
        "y": np.asarray(ys, dtype=np.float32),
        "y_lvh": np.asarray(lvh, dtype=np.float32),
        "y_rvh": np.asarray(rvh, dtype=np.float32),
        "age": np.asarray(ages, dtype=np.float64),
        "sex": np.asarray(sexes),
        "record": np.asarray(names),
        "missing": missing,
    }


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=ROOT / "outputs" / "hyp_v1_aug" / "best_model.pt")
    parser.add_argument("--tag", type=str, default="hyp_v1_aug")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "georgia_ext")
    parser.add_argument("--fig-prefix", type=str, default="georgia_ext")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = TinyECGCNN()
    model.load_state_dict(blob["model"])
    model.to(device).eval()

    # Temperature / thresholds are model-version specific.
    metrics_path = ROOT / "outputs" / args.tag / "metrics_v1_1.json"
    temp_path = ROOT / "outputs" / args.tag / "metrics_v1_2_tempscale.json"
    if metrics_path.exists():
        version_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        thresholds = version_metrics["thresholds_val_locked"]
    else:
        thresholds = {
            "youden": 0.42832863330841064,
            "sens_ge_0.90": 0.18672962486743927,
            "spec_ge_0.90": 0.6423748135566711,
        }
    if temp_path.exists():
        temperature = float(json.loads(temp_path.read_text(encoding="utf-8"))["temperature"])
    else:
        temperature = 1.0
    print(f"tag={args.tag} temperature={temperature:.4f} thresholds={thresholds}")

    ga = load_georgia()
    logits = predict_logits(model, ga["x"], device)
    score = sigmoid(logits)
    score_cal = sigmoid(logits / temperature)

    overall = bootstrap_metrics(ga["y"], score, n_boot=args.n_boot, seed=SEED)
    overall_cal = bootstrap_metrics(ga["y"], score_cal, n_boot=args.n_boot, seed=SEED)
    ece_raw = expected_calibration_error(ga["y"], score)
    ece_cal = expected_calibration_error(ga["y"], score_cal)

    ops = {}
    for name, thr in thresholds.items():
        op = operating_point(ga["y"], score, thr)
        op["ppv_at_prevalence"], op["npv_at_prevalence"] = ppv_npv_at_prevalence(
            op["sensitivity"], op["specificity"], overall["pos_rate"]
        )
        ops[name] = op

    # subgroups
    sex = np.char.lower(ga["sex"].astype(str))
    is_male = np.char.startswith(sex, "m")
    is_female = np.char.startswith(sex, "f")
    age_ok = np.isfinite(ga["age"]) & (ga["age"] > 0)
    negatives = ga["y"] == 0
    subgroup_defs = [
        ("All", np.ones_like(ga["y"], dtype=bool), ga["y"]),
        (
            "LVH only vs negatives",
            negatives | ((ga["y"] == 1) & (ga["y_lvh"] == 1)),
            ((ga["y"] == 1) & (ga["y_lvh"] == 1)).astype(float),
        ),
        (
            "RVH only vs negatives",
            negatives | ((ga["y"] == 1) & (ga["y_lvh"] == 0)),
            ((ga["y"] == 1) & (ga["y_lvh"] == 0)).astype(float),
        ),
        ("Male", is_male, ga["y"]),
        ("Female", is_female, ga["y"]),
        ("Age <65", age_ok & (ga["age"] < 65), ga["y"]),
        ("Age >=65", age_ok & (ga["age"] >= 65), ga["y"]),
    ]
    subgroup_rows = []
    for label, mask, y_sub in subgroup_defs:
        if mask.sum() < 20 or len(np.unique(y_sub[mask])) < 2:
            continue
        m = bootstrap_metrics(y_sub[mask], score[mask], n_boot=args.n_boot, seed=SEED)
        m.update({"label": label, "is_baseline": False})
        subgroup_rows.append(m)

    # age+sex baseline transported from PTB-XL
    base_model, _ = fit_age_sex_baseline()
    age_fill = np.where(age_ok, ga["age"], 60.0)
    sex_num = np.where(is_male, 0.0, np.where(is_female, 1.0, 0.0))
    base_scores = sigmoid(base_model.decision_function(np.column_stack([age_fill, sex_num])))
    base_metrics = bootstrap_metrics(ga["y"], base_scores, n_boot=args.n_boot, seed=SEED)
    base_metrics.update({"label": "Age+sex baseline", "is_baseline": True})
    subgroup_rows.append(base_metrics)
    delta = paired_delta_auroc(ga["y"], score, base_scores, n_boot=args.n_boot, seed=SEED)

    print(
        f"Georgia AUROC={overall['auroc']:.4f} "
        f"({overall['auroc_ci95'][0]:.4f}-{overall['auroc_ci95'][1]:.4f}) "
        f"AUPRC={overall['auprc']:.4f} ECE={ece_raw:.4f}/{ece_cal:.4f}"
    )
    print(f"baseline AUROC={base_metrics['auroc']:.4f}; delta={delta['delta_auroc']:+.4f}")

    # figures
    plot_roc_pr(
        ga["y"],
        score,
        None,
        None,
        overall,
        FIG_DIR / f"{args.fig_prefix}_roc_pr.png",
        dataset_label="Georgia",
    )
    plot_calibration(
        ga["y"],
        score,
        score_cal,
        ece_raw,
        ece_cal,
        FIG_DIR / f"{args.fig_prefix}_calibration.png",
        dataset_label="Georgia",
    )
    ptbxl_path = ROOT / "outputs" / args.tag / "metrics.json"
    ptbxl_auroc = (
        json.loads(ptbxl_path.read_text(encoding="utf-8"))["cnn"]["test"]["auroc"]
        if ptbxl_path.exists()
        else 0.5
    )
    plot_subgroups(
        subgroup_rows,
        ptbxl_auroc,
        FIG_DIR / f"{args.fig_prefix}_subgroups.png",
        title="Georgia subgroup discrimination",
    )

    # predictions
    pd.DataFrame(
        {
            "record": ga["record"],
            "y_hyp": ga["y"].astype(int),
            "y_lvh": ga["y_lvh"].astype(int),
            "y_rvh": ga["y_rvh"].astype(int),
            "age": ga["age"],
            "sex": ga["sex"],
            "logit": logits,
            "score": score,
            "score_cal": score_cal,
            "baseline_age_sex": base_scores,
        }
    ).to_csv(args.out_dir / "predictions.csv", index=False)

    pd.DataFrame(
        [
            {
                "subgroup": r["label"],
                "n": r["n"],
                "n_pos": r["n_pos"],
                "prevalence": round(r["pos_rate"], 4),
                "auroc": round(r["auroc"], 4),
                "auroc_95ci": f"{r['auroc_ci95'][0]:.3f}-{r['auroc_ci95'][1]:.3f}",
                "auprc": round(r["auprc"], 4),
                "is_baseline": bool(r.get("is_baseline", False)),
            }
            for r in subgroup_rows
        ]
    ).to_csv(args.out_dir / "table3_subgroups.csv", index=False)

    payload = {
        "dataset": "Georgia 12-Lead ECG Challenge Database (Emory, Kaggle CC0)",
        "n": overall["n"],
        "n_pos": overall["n_pos"],
        "pos_rate": overall["pos_rate"],
        "n_missing": ga["missing"],
        "auroc": overall["auroc"],
        "auroc_ci95": overall["auroc_ci95"],
        "auprc": overall["auprc"],
        "auprc_ci95": overall["auprc_ci95"],
        "ece_raw": ece_raw,
        "ece_cal": ece_cal,
        "temperature": temperature,
        "thresholds_val_locked": thresholds,
        "operating_points": ops,
        "subgroups": subgroup_rows,
        "baseline_age_sex": base_metrics,
        "delta_auroc_cnn_minus_baseline": delta,
        "caveat": (
            "Labels are SNOMED CT codes from the Challenge 2020 Georgia subset "
            "(report-derived), not echo-confirmed; natural prevalence, not case-control."
        ),
    }
    (args.out_dir / "external_metrics.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8"
    )
    print(f"wrote {args.out_dir / 'external_metrics.json'}")


if __name__ == "__main__":
    main()
