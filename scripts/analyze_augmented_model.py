"""Complete internal + external analysis for the noise-augmented HYP CNN.

The v1 checkpoint (``outputs/hyp_v1``) was trained before the augmentation bug
was fixed and is therefore a clean baseline.  This script analyses the properly
augmented model (``outputs/hyp_v1_aug``):

  * temperature scaling fitted on validation fold 9
  * val-locked thresholds and test operating points
  * PTB-XL test AUROC/AUPRC with bootstrap CIs and ECE
  * Chapman-Shaoxing external validation (same case-control sample)
  * age+sex baseline comparison
  * a compact clean-vs-augmented comparison table

Usage:
  python scripts/analyze_augmented_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ai_ecg_hf.dataset import PTBXLLeadIBinary, zscore  # noqa: E402
from ai_ecg_hf.model import TinyECGCNN  # noqa: E402
from ai_ecg_hf.ptbxl import extract_lead_i, read_record  # noqa: E402

from analyze_chapman_external import (  # noqa: E402
    CHAPMAN_FULL_PREV,
    bootstrap_metrics,
    expected_calibration_error,
    fit_age_sex_baseline,
    load_chapman_case_control,
    operating_point,
    paired_delta_auroc,
    plot_calibration,
    plot_roc_pr,
    plot_subgroups,
    ppv_npv_at_prevalence,
    predict_logits,
)

OUT_DIR = ROOT / "outputs" / "hyp_v1_aug"
FIG_DIR = ROOT / "figures"
CKPT = OUT_DIR / "best_model.pt"
SEED = 42


def load_ptbxl_split(split: str) -> dict:
    ds = PTBXLLeadIBinary(
        split=split,
        positive_superclasses=("HYP",),
        augment=False,
        require_local=True,
        seed=SEED,
    )
    xs, ys = [], []
    for ecg_id in tqdm(ds.ids, desc=f"load-ptbxl-{split}", leave=False):
        row = ds.df.loc[ecg_id]
        signal, meta = read_record(ds.root, row["filename_lr"])
        xs.append(zscore(extract_lead_i(signal, meta)).astype(np.float32))
        ys.append(float(row["y"]))
    return {"x": np.stack(xs).astype(np.float32), "y": np.asarray(ys, dtype=np.float32)}


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    import torch.nn as nn

    z = torch.tensor(logits, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    log_t = torch.nn.Parameter(torch.zeros(()))
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=50)
    loss_fn = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = loss_fn(z / torch.exp(log_t), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(max(torch.exp(log_t).detach(), 1e-3))


def choose_thresholds(y_val: np.ndarray, s_val: np.ndarray) -> dict[str, float]:
    fpr, tpr, thr = roc_curve(y_val, s_val)
    youden = float(thr[int(np.argmax(tpr - fpr))])

    sens_thr = 0.0
    for t in np.unique(s_val)[::-1]:
        if operating_point(y_val, s_val, float(t))["sensitivity"] >= 0.90:
            sens_thr = float(t)
            break

    spec_thr = 1.0
    for t in np.unique(s_val):
        if operating_point(y_val, s_val, float(t))["specificity"] >= 0.90:
            spec_thr = float(t)
            break

    return {
        "youden": youden,
        "sens_ge_0.90": sens_thr,
        "spec_ge_0.90": spec_thr,
    }


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = TinyECGCNN()
    model.load_state_dict(blob["model"])
    model.to(device).eval()

    # ------------------------- PTB-XL val/test ---------------------------- #
    va = load_ptbxl_split("val")
    te = load_ptbxl_split("test")
    z_va = predict_logits(model, va["x"], device)
    z_te = predict_logits(model, te["x"], device)
    s_va, s_te = sigmoid(z_va), sigmoid(z_te)

    temperature = fit_temperature(z_va, va["y"])
    s_te_cal = sigmoid(z_te / temperature)

    thrs = choose_thresholds(va["y"], s_va)
    val_ops = {k: operating_point(va["y"], s_va, t) for k, t in thrs.items()}
    test_ops = {k: operating_point(te["y"], s_te, t) for k, t in thrs.items()}
    ece_test_raw = expected_calibration_error(te["y"], s_te)
    ece_test_cal = expected_calibration_error(te["y"], s_te_cal)
    internal = bootstrap_metrics(te["y"], s_te, n_boot=1000, seed=SEED)
    internal_cal = bootstrap_metrics(te["y"], s_te_cal, n_boot=1000, seed=SEED)

    print(f"T={temperature:.4f}")
    print(f"internal test AUROC={internal['auroc']:.4f} AUPRC={internal['auprc']:.4f}")
    print(f"ECE raw={ece_test_raw:.4f} cal={ece_test_cal:.4f}")
    print("thresholds:", thrs)

    # --------------------------- Chapman ---------------------------------- #
    ch = load_chapman_case_control(seed=SEED)
    z_ch = predict_logits(model, ch["x"], device)
    s_ch = sigmoid(z_ch)
    s_ch_cal = sigmoid(z_ch / temperature)
    ch_metrics = bootstrap_metrics(ch["y"], s_ch, n_boot=1000, seed=SEED)
    ch_metrics_cal = bootstrap_metrics(ch["y"], s_ch_cal, n_boot=1000, seed=SEED)
    ece_ch_raw = expected_calibration_error(ch["y"], s_ch)
    ece_ch_cal = expected_calibration_error(ch["y"], s_ch_cal)

    ch_ops = {}
    for name, thr in thrs.items():
        op = operating_point(ch["y"], s_ch, thr)
        op["ppv_at_full_prevalence"], op["npv_at_full_prevalence"] = ppv_npv_at_prevalence(
            op["sensitivity"], op["specificity"], CHAPMAN_FULL_PREV
        )
        ch_ops[name] = op

    # age+sex baseline transported to Chapman
    base_model, base = fit_age_sex_baseline()
    sex = np.char.lower(ch["sex"].astype(str))
    is_male = np.char.startswith(sex, "m")
    is_female = np.char.startswith(sex, "f")
    age_fill = np.where(np.isfinite(ch["age"]) & (ch["age"] > 0), ch["age"], 60.0)
    sex_num = np.where(is_male, 0.0, np.where(is_female, 1.0, 0.0))
    base_ch_scores = sigmoid(
        base_model.decision_function(np.column_stack([age_fill, sex_num]))
    )
    base_ch = bootstrap_metrics(ch["y"], base_ch_scores, n_boot=1000, seed=SEED)
    delta_ch = paired_delta_auroc(
        ch["y"], s_ch, base_ch_scores, n_boot=1000, seed=SEED
    )

    print(
        f"Chapman AUROC={ch_metrics['auroc']:.4f} "
        f"({ch_metrics['auroc_ci95'][0]:.4f}-{ch_metrics['auroc_ci95'][1]:.4f})"
    )
    print(f"Chapman ECE raw={ece_ch_raw:.4f} cal={ece_ch_cal:.4f}")
    print(f"baseline AUROC={base_ch['auroc']:.4f}; delta={delta_ch['delta_auroc']:+.4f}")

    # --------------------------- subgroups -------------------------------- #
    negatives = ch["y"] == 0
    age_ok = np.isfinite(ch["age"]) & (ch["age"] > 0)
    subgroup_defs = [
        ("All", np.ones_like(ch["y"], dtype=bool), ch["y"]),
        (
            "LVH only vs negatives",
            negatives | ((ch["y"] == 1) & (ch["y_lvh"] == 1)),
            ((ch["y"] == 1) & (ch["y_lvh"] == 1)).astype(float),
        ),
        (
            "RVH/RAH only vs negatives",
            negatives | ((ch["y"] == 1) & (ch["y_lvh"] == 0)),
            ((ch["y"] == 1) & (ch["y_lvh"] == 0)).astype(float),
        ),
        ("Male", is_male, ch["y"]),
        ("Female", is_female, ch["y"]),
        ("Age <65", age_ok & (ch["age"] < 65), ch["y"]),
        ("Age ≥65", age_ok & (ch["age"] >= 65), ch["y"]),
    ]
    subgroup_rows = []
    for label, mask, y_sub in subgroup_defs:
        if mask.sum() < 20 or len(np.unique(y_sub[mask])) < 2:
            continue
        m = bootstrap_metrics(y_sub[mask], s_ch[mask], n_boot=1000, seed=SEED)
        m.update({"label": label, "is_baseline": False})
        subgroup_rows.append(m)
    base_ch.update({"label": "Age+sex baseline", "is_baseline": True})
    subgroup_rows.append(base_ch)
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
    ).to_csv(OUT_DIR / "table3_subgroups.csv", index=False)

    # ---------------------------- figures --------------------------------- #
    plot_subgroups(
        subgroup_rows,
        internal["auroc"],
        FIG_DIR / "hyp_v1_aug_chapman_subgroups.png",
    )
    plot_roc_pr(
        ch["y"],
        s_ch,
        te["y"],
        s_te,
        ch_metrics,
        FIG_DIR / "hyp_v1_aug_chapman_roc_pr.png",
    )
    plot_calibration(
        ch["y"],
        s_ch,
        s_ch_cal,
        ece_ch_raw,
        ece_ch_cal,
        FIG_DIR / "hyp_v1_aug_chapman_calibration.png",
    )

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for scores, label, ece in [
        (s_te, "raw", ece_test_raw),
        (s_te_cal, "temperature-scaled", ece_test_cal),
    ]:
        from sklearn.calibration import calibration_curve

        frac, mean = calibration_curve(te["y"], scores, n_bins=10, strategy="uniform")
        ax.plot(mean, frac, marker="o", lw=2, label=f"{label} (ECE {ece:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=1, label="ideal")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("PTB-XL test calibration (augmented)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hyp_v1_aug_test_calibration.png", dpi=300)
    plt.close(fig)

    # ----------------------------- JSON ----------------------------------- #
    metrics_v11 = {
        "version": "v1.3_noise_augmented",
        "note": "Primary model after fixing the augmentation path; thresholds locked on validation fold 9.",
        "temperature": temperature,
        "auroc": {
            "val": float(roc_auc_score(va["y"], s_va)),
            "test": float(roc_auc_score(te["y"], s_te)),
        },
        "thresholds_val_locked": thrs,
        "operating_points": {"val": val_ops, "test": test_ops},
        "calibration": {
            "ece_val": expected_calibration_error(va["y"], s_va),
            "ece_test": ece_test_raw,
            "ece_test_calibrated": ece_test_cal,
        },
        "discrimination": {"test_raw": internal, "test_calibrated": internal_cal},
    }
    (OUT_DIR / "metrics_v1_1.json").write_text(
        json.dumps(metrics_v11, indent=2, default=float), encoding="utf-8"
    )

    metrics_v12 = {
        "version": "v1.3_temperature_scaling",
        "temperature": temperature,
        "fit_on": "validation fold 9 only",
        "auroc_test_raw": internal["auroc"],
        "auroc_test_cal": internal_cal["auroc"],
        "ece": {
            "val_raw": expected_calibration_error(va["y"], s_va),
            "val_cal": expected_calibration_error(va["y"], sigmoid(z_va / temperature)),
            "test_raw": ece_test_raw,
            "test_cal": ece_test_cal,
        },
        "thresholds_val_locked_on_calibrated": choose_thresholds(va["y"], sigmoid(z_va / temperature)),
    }
    (OUT_DIR / "metrics_v1_2_tempscale.json").write_text(
        json.dumps(metrics_v12, indent=2, default=float), encoding="utf-8"
    )

    external = {
        "task": "external validation of noise-augmented PTB-XL HYP CNN -> Chapman-Shaoxing",
        "n_used": ch_metrics["n"],
        "pos_rate": ch_metrics["pos_rate"],
        "auroc": ch_metrics["auroc"],
        "auroc_ci95": ch_metrics["auroc_ci95"],
        "auprc": ch_metrics["auprc"],
        "auprc_ci95": ch_metrics["auprc_ci95"],
        "ece_raw": ece_ch_raw,
        "ece_cal": ece_ch_cal,
        "temperature": temperature,
        "thresholds_val_locked": thrs,
        "operating_points": ch_ops,
        "subgroups": subgroup_rows,
        "baseline_age_sex": base_ch,
        "delta_auroc_cnn_minus_baseline": delta_ch,
        "caveat": (
            "Positive = expert-labeled SNOMED LVH/RVH/RAH; case-control sample "
            "(769 positives + 2,000 negatives, seed 42); no echo confirmation."
        ),
    }
    (OUT_DIR / "external_metrics.json").write_text(
        json.dumps(external, indent=2, default=float), encoding="utf-8"
    )

    # save predictions for downstream analyses
    pd.DataFrame(
        {
            "record": ch["record"],
            "y_hyp": ch["y"].astype(int),
            "y_lvh": ch["y_lvh"].astype(int),
            "age": ch["age"],
            "sex": ch["sex"],
            "logit": z_ch,
            "score": s_ch,
            "score_cal": s_ch_cal,
            "baseline_age_sex": base_ch_scores,
        }
    ).to_csv(OUT_DIR / "chapman_predictions.csv", index=False)

    # -------------------- clean vs augmented table ------------------------ #
    clean_ext = json.loads(
        (ROOT / "outputs" / "chapman_ext" / "external_analysis.json").read_text(encoding="utf-8")
    )
    clean_internal = clean_ext["discrimination"]["ptbxl_test"]
    clean_ch = clean_ext["discrimination"]["chapman"]
    clean_cal = clean_ext["calibration"]
    clean_v11 = json.loads(
        (ROOT / "outputs" / "hyp_v1" / "metrics_v1_1.json").read_text(encoding="utf-8")
    )
    noise = pd.read_csv(ROOT / "outputs" / "noise_robustness" / "table4_noise_robustness.csv")

    def op_of(ops, key="sens_ge_0.90"):
        return ops[key]["sensitivity"], ops[key]["specificity"]

    rows = []
    for model_name, internal_m, ch_m, ece_internal, ece_ch, ops in [
        (
            "Clean-trained",
            clean_internal,
            clean_ch,
            clean_cal["ece_internal_test_raw"],
            clean_cal["ece_raw"],
            clean_v11["operating_points"]["test"],
        ),
        (
            "Noise-augmented",
            internal,
            ch_metrics,
            ece_test_raw,
            ece_ch_raw,
            test_ops,
        ),
    ]:
        sens, spec = op_of(ops)
        rows.append(
            {
                "model": model_name,
                "ptbxl_test_n": internal_m["n"],
                "ptbxl_test_auroc": round(internal_m["auroc"], 4),
                "ptbxl_test_auroc_ci": f"{internal_m['auroc_ci95'][0]:.3f}-{internal_m['auroc_ci95'][1]:.3f}",
                "ptbxl_test_auprc": round(internal_m["auprc"], 4),
                "ptbxl_test_ece": round(ece_internal, 4),
                "chapman_n": ch_m["n"],
                "chapman_auroc": round(ch_m["auroc"], 4),
                "chapman_auroc_ci": f"{ch_m['auroc_ci95'][0]:.3f}-{ch_m['auroc_ci95'][1]:.3f}",
                "chapman_auprc": round(ch_m["auprc"], 4),
                "chapman_ece": round(ece_ch, 4),
                "chapman_sens_at_sens90": round(sens, 4),
                "chapman_spec_at_sens90": round(spec, 4),
            }
        )
    comparison = pd.DataFrame(rows)

    for cohort, label in [("ptbxl", "PTB-XL"), ("chapman", "Chapman")]:
        sub = noise[noise["cohort"] == cohort].set_index("severity")
        comparison.loc[
            comparison["model"] == "Clean-trained", f"{cohort}_auroc_noise4x"
        ] = round(float(sub.loc[4.0, "auroc_clean"]), 4)
        comparison.loc[
            comparison["model"] == "Noise-augmented", f"{cohort}_auroc_noise4x"
        ] = round(float(sub.loc[4.0, "auroc_augmented"]), 4)

    comparison.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    print("\n==== clean vs augmented ====")
    print(comparison.to_string(index=False))
    print(f"wrote {OUT_DIR / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
