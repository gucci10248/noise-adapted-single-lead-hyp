"""Complete analysis package for the Chapman-Shaoxing external validation.

Reuses the frozen PTB-XL HYP Lead-I CNN (v1) and the same case-control sample as
``eval_chapman_external.py`` (all 769 positives + 2,000 negatives, seed 42), but
adds everything needed for a results section:

  * per-record predictions (CSV)
  * bootstrap CIs for AUROC/AUPRC
  * calibration (raw vs temperature scaling), ECE and Brier score
  * operating points at the PTB-XL-val-locked thresholds, with PPV/NPV projected
    to the full Chapman prevalence
  * subgroups: LVH-only, RVH/RAH-only, sex, age
  * age+sex logistic baseline transported from PTB-XL, with paired delta-AUROC
  * figures: ROC/PR, calibration, subgroup forest plot

Usage:
  python scripts/analyze_chapman_external.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_ecg_hf.chapman import list_local_records, read_record  # noqa: E402
from ai_ecg_hf.dataset import PTBXLLeadIBinary, zscore  # noqa: E402
from ai_ecg_hf.mimic_ecg import resample_to_length  # noqa: E402
from ai_ecg_hf.model import TinyECGCNN  # noqa: E402
from ai_ecg_hf.ptbxl import (  # noqa: E402
    add_diagnostic_superclass,
    extract_lead_i,
    fold_masks,
    load_metadata,
    read_record as read_ptbxl_record,
)

OUT_DIR = ROOT / "outputs" / "chapman_ext"
FIG_DIR = ROOT / "figures"
CKPT = ROOT / "outputs" / "hyp_v1" / "best_model.pt"
TEMPERATURE = 0.8841570615768433
NEG_POOL = 2000
SEED = 42
N_BOOT = 1000

# Thresholds locked on PTB-XL validation fold 9 (v1.1 raw / v1.2 calibrated).
THRESHOLDS_RAW = {
    "youden": 0.42832863330841064,
    "sens_ge_0.90": 0.18672962486743927,
    "spec_ge_0.90": 0.6423748135566711,
}
THRESHOLDS_CAL = {
    "youden": 0.41909345984458923,
    "sens_ge_0.90": 0.15920086205005646,
    "spec_ge_0.90": 0.6598045229911804,
}

# Full Chapman-Shaoxing library prevalence (769 / 45,152), not the case-control
# prevalence.  Used only for prevalence-adjusted PPV/NPV projections.
CHAPMAN_FULL_PREV = 769.0 / 45152.0
CHAPMAN_LVH_FULL_PREV = 647.0 / 45152.0
CHAPMAN_RVH_RAH_FULL_PREV = 122.0 / 45152.0


# --------------------------------------------------------------------------- #
# metrics helpers
# --------------------------------------------------------------------------- #
def bootstrap_metrics(
    y: np.ndarray, scores: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED
) -> dict:
    """AUROC/AUPRC with the same percentile bootstrap used by the v1 eval."""
    y = np.asarray(y, dtype=float)
    scores = np.asarray(scores, dtype=float)
    out = {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "pos_rate": float(y.mean()),
        "auroc": float(roc_auc_score(y, scores)),
        "auprc": float(average_precision_score(y, scores)),
    }
    rng = np.random.RandomState(seed)
    boot_auroc, boot_auprc = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        boot_auroc.append(roc_auc_score(y[idx], scores[idx]))
        boot_auprc.append(average_precision_score(y[idx], scores[idx]))
    out["auroc_ci95"] = [
        float(np.percentile(boot_auroc, 2.5)),
        float(np.percentile(boot_auroc, 97.5)),
    ]
    out["auprc_ci95"] = [
        float(np.percentile(boot_auprc, 2.5)),
        float(np.percentile(boot_auprc, 97.5)),
    ]
    return out


def paired_delta_auroc(
    y: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    """Bootstrap CNN-minus-baseline AUROC on the same records."""
    y = np.asarray(y, dtype=float)
    delta = float(roc_auc_score(y, scores_a) - roc_auc_score(y, scores_b))
    rng = np.random.RandomState(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        boot.append(
            roc_auc_score(y[idx], scores_a[idx])
            - roc_auc_score(y[idx], scores_b[idx])
        )
    return {
        "delta_auroc": delta,
        "delta_ci95": [
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)),
        ],
    }


def expected_calibration_error(y: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (scores >= bins[i]) & (
            scores < bins[i + 1] if i < n_bins - 1 else scores <= bins[i + 1]
        )
        if not np.any(m):
            continue
        ece += (m.mean()) * abs(scores[m].mean() - y[m].mean())
    return float(ece)


def operating_point(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {
        "threshold": float(threshold),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(tp / max(tp + fp, 1)),
        "npv": float(tn / max(tn + fn, 1)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def ppv_npv_at_prevalence(sens: float, spec: float, prevalence: float) -> tuple[float, float]:
    ppv = prevalence * sens / max(
        prevalence * sens + (1.0 - prevalence) * (1.0 - spec), 1e-12
    )
    npv = (1.0 - prevalence) * spec / max(
        (1.0 - prevalence) * spec + prevalence * (1.0 - sens), 1e-12
    )
    return float(ppv), float(npv)


def fmt_ci(est: float, ci: list[float], digits: int = 3) -> str:
    return f"{est:.{digits}f} ({ci[0]:.{digits}f}–{ci[1]:.{digits}f})"


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def load_chapman_case_control(neg_pool: int = NEG_POOL, seed: int = SEED) -> dict:
    """Reproduce the exact sample behind external_metrics.json.

    The reference run labelled every local record with ``read_record``; records
    without a readable ``.mat`` were skipped before sampling.  We keep that
    order and skip the seven known-unreadable waveforms, which reproduces the
    reported mean age and sex distribution exactly.
    """
    records = list_local_records()
    labeled = []
    unreadable = 0
    for rec in tqdm(records, desc="scan-chapman", leave=False):
        if not Path(str(rec) + ".mat").exists():
            continue
        try:
            _, info = read_record(rec)
        except Exception:
            unreadable += 1
            continue
        labeled.append((rec, info))

    pos = [t for t in labeled if t[1]["y_hyp"] == 1]
    neg = [t for t in labeled if t[1]["y_hyp"] == 0]
    rng = np.random.RandomState(seed)
    if len(neg) > neg_pool:
        idx = rng.choice(len(neg), size=neg_pool, replace=False)
        neg = [neg[i] for i in sorted(idx)]
    picked = pos + neg
    print(
        f"chapman readable={len(labeled)} (unreadable={unreadable}) "
        f"pos={len(pos)} neg={len(neg)} used={len(picked)}"
    )

    xs, ys, lvh, ages, sexes, names = [], [], [], [], [], []
    for rec, info in tqdm(picked, desc="load-chapman", leave=False):
        lead, _ = read_record(rec)
        lead = zscore(resample_to_length(lead, 1000))
        xs.append(np.asarray(lead, dtype=np.float32))
        ys.append(float(info["y_hyp"]))
        lvh.append(float(info["y_lvh"]))
        ages.append(float(info.get("age", np.nan)))
        sexes.append(str(info.get("sex", "")))
        names.append(str(rec.relative_to(ROOT / "data" / "chapman")))

    return {
        "x": np.stack(xs).astype(np.float32),
        "y": np.asarray(ys, dtype=np.float32),
        "y_lvh": np.asarray(lvh, dtype=np.float32),
        "age": np.asarray(ages, dtype=np.float64),
        "sex": np.asarray(sexes),
        "record": np.asarray(names),
        "n_labeled": len(labeled),
        "n_unreadable": unreadable,
        "n_positive_total": len(pos),
    }


def load_ptbxl_test(seed: int = SEED) -> dict:
    """Load the frozen fold-10 test set as Lead I arrays."""
    ds = PTBXLLeadIBinary(
        split="test",
        positive_superclasses=("HYP",),
        augment=False,
        require_local=True,
        seed=seed,
    )
    xs, ys = [], []
    for ecg_id in tqdm(ds.ids, desc="load-ptbxl-test", leave=False):
        row = ds.df.loc[ecg_id]
        signal, meta = read_ptbxl_record(ds.root, row["filename_lr"])
        xs.append(zscore(extract_lead_i(signal, meta)).astype(np.float32))
        ys.append(float(row["y"]))
    return {"x": np.stack(xs).astype(np.float32), "y": np.asarray(ys, dtype=np.float32)}


@torch.no_grad()
def predict_logits(model: torch.nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x[:, None, :])), batch_size=256)
    out = []
    for (xb,) in loader:
        out.append(model(xb.to(device)).detach().cpu().numpy())
    return np.concatenate(out)


def fit_age_sex_baseline() -> tuple[LogisticRegression, dict]:
    """Fit the same age+sex logistic baseline on PTB-XL folds 1-8."""
    df = add_diagnostic_superclass(load_metadata(), ROOT / "data" / "ptb-xl")
    df["y"] = df["diagnostic_superclass"].apply(lambda labs: int("HYP" in labs))
    masks = fold_masks(df)
    train = df.loc[masks["train"]]

    def design(frame: pd.DataFrame) -> np.ndarray:
        age = frame["age"].fillna(60.0).to_numpy(dtype=float)
        sex = frame["sex"].to_numpy(dtype=float)
        return np.column_stack([age, sex])

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(design(train), train["y"].to_numpy(dtype=float))
    test = df.loc[masks["test"]]
    return model, {
        "test_logits": model.decision_function(design(test)),
        "test_y": test["y"].to_numpy(dtype=float),
    }


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def plot_roc_pr(
    y, scores, internal_y, internal_scores, metrics, path: Path, dataset_label: str = "Chapman"
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    ax = axes[0]
    fpr, tpr, _ = roc_curve(y, scores)
    ax.plot(
        fpr,
        tpr,
        lw=2,
        color="C0",
        label=f"{dataset_label} (AUROC {fmt_ci(metrics['auroc'], metrics['auroc_ci95'])})",
    )
    if internal_y is not None:
        fpr_i, tpr_i, _ = roc_curve(internal_y, internal_scores)
        ax.plot(
            fpr_i,
            tpr_i,
            lw=1.5,
            ls="--",
            color="0.45",
            label=f"PTB-XL test (AUROC {roc_auc_score(internal_y, internal_scores):.3f})",
        )
    ax.plot([0, 1], [0, 1], ls=":", color="0.7", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    prec, rec, _ = precision_recall_curve(y, scores)
    ax.plot(
        rec,
        prec,
        lw=2,
        color="C1",
        label=f"{dataset_label} (AUPRC {fmt_ci(metrics['auprc'], metrics['auprc_ci95'])})",
    )
    ax.axhline(y.mean(), ls=":", color="0.5", lw=1, label=f"prevalence {y.mean():.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_calibration(
    y, raw, cal, ece_raw, ece_cal, path: Path, dataset_label: str = "Chapman"
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    ax = axes[0]
    for scores, label, ece in [
        (raw, "raw", ece_raw),
        (cal, "temperature-scaled", ece_cal),
    ]:
        frac, mean = calibration_curve(y, scores, n_bins=10, strategy="uniform")
        ax.plot(mean, frac, marker="o", lw=2, label=f"{label} (ECE {ece:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=1, label="ideal")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration ({dataset_label})")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    bins = np.linspace(0, 1, 31)
    ax.hist(raw[y == 0], bins=bins, alpha=0.55, density=True, label="negative")
    ax.hist(raw[y == 1], bins=bins, alpha=0.55, density=True, label="positive")
    ax.set_xlabel("Predicted probability (raw)")
    ax.set_ylabel("Density")
    ax.set_title("Score distribution")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_subgroups(
    rows: list[dict],
    internal_auroc: float,
    path: Path,
    title: str = "Chapman subgroup discrimination",
) -> None:
    rows = [r for r in rows if r.get("auroc") is not None]
    labels = [r["label"] for r in rows]
    est = np.array([r["auroc"] for r in rows])
    lo = np.array([r["auroc_ci95"][0] for r in rows])
    hi = np.array([r["auroc_ci95"][1] for r in rows])
    ypos = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(rows) + 1.6))
    colors = ["C3" if r.get("is_baseline") else "C0" for r in rows]
    ax.errorbar(
        est,
        ypos,
        xerr=[est - lo, hi - est],
        fmt="o",
        ecolor="0.45",
        capsize=3,
        lw=1.2,
        ms=5,
        color="C0",
    )
    for i, (x, yv, c) in enumerate(zip(est, ypos, colors)):
        ax.plot(x, yv, "o", color=c, ms=6)
    ax.axvline(0.5, ls=":", color="0.6", lw=1)
    ax.axvline(internal_auroc, ls="--", color="C1", lw=1.2, label="PTB-XL test")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("AUROC (95% CI)")
    ax.set_xlim(0.45, 1.0)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = TinyECGCNN()
    model.load_state_dict(blob["model"])
    model.to(device)

    # ------------------------------ data ---------------------------------- #
    ch = load_chapman_case_control(seed=args.seed)
    pt = load_ptbxl_test(seed=args.seed)
    ch_logits = predict_logits(model, ch["x"], device)
    pt_logits = predict_logits(model, pt["x"], device)
    ch["score"] = 1.0 / (1.0 + np.exp(-ch_logits))
    ch["score_cal"] = 1.0 / (1.0 + np.exp(-ch_logits / TEMPERATURE))
    pt["score"] = 1.0 / (1.0 + np.exp(-pt_logits))

    # sanity: must reproduce the frozen reference numbers
    overall = bootstrap_metrics(ch["y"], ch["score"], n_boot=args.n_boot, seed=args.seed)
    internal = bootstrap_metrics(pt["y"], pt["score"], n_boot=args.n_boot, seed=args.seed)
    print(f"Chapman AUROC={overall['auroc']:.6f} (reference 0.884670)")
    print(f"PTB-XL  AUROC={internal['auroc']:.6f} (reference 0.787399)")
    if abs(overall["auroc"] - 0.8846703511053315) > 1e-6:
        raise RuntimeError("Chapman sample does not reproduce the frozen reference AUROC")

    # -------------------------- calibration ------------------------------- #
    ece_raw = expected_calibration_error(ch["y"], ch["score"])
    ece_cal = expected_calibration_error(ch["y"], ch["score_cal"])
    cal = {
        "temperature": TEMPERATURE,
        "temperature_fit_on": "PTB-XL validation fold 9",
        "ece_raw": ece_raw,
        "ece_cal": ece_cal,
        "brier_raw": float(brier_score_loss(ch["y"], ch["score"])),
        "brier_cal": float(brier_score_loss(ch["y"], ch["score_cal"])),
        "ece_internal_test_raw": expected_calibration_error(pt["y"], pt["score"]),
    }

    # ------------------------ operating points ---------------------------- #
    ops = {}
    for name, thr in THRESHOLDS_RAW.items():
        op = operating_point(ch["y"], ch["score"], thr)
        op["ppv_at_full_prevalence"], op["npv_at_full_prevalence"] = ppv_npv_at_prevalence(
            op["sensitivity"], op["specificity"], CHAPMAN_FULL_PREV
        )
        ops[f"raw_{name}"] = op
    for name, thr in THRESHOLDS_CAL.items():
        op = operating_point(ch["y"], ch["score_cal"], thr)
        op["ppv_at_full_prevalence"], op["npv_at_full_prevalence"] = ppv_npv_at_prevalence(
            op["sensitivity"], op["specificity"], CHAPMAN_FULL_PREV
        )
        ops[f"calibrated_{name}"] = op

    # ---------------------------- subgroups ------------------------------- #
    sex = np.char.lower(ch["sex"].astype(str))
    is_male = np.char.startswith(sex, "m")
    is_female = np.char.startswith(sex, "f")
    age_ok = np.isfinite(ch["age"]) & (ch["age"] > 0)
    negatives = ch["y"] == 0

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
        m = bootstrap_metrics(y_sub[mask], ch["score"][mask], n_boot=args.n_boot, seed=args.seed)
        m.update({"label": label, "is_baseline": False})
        subgroup_rows.append(m)

    # ----------------------- age+sex baseline ----------------------------- #
    base_model, base = fit_age_sex_baseline()
    age_fill = np.where(np.isfinite(ch["age"]) & (ch["age"] > 0), ch["age"], 60.0)
    sex_num = np.where(is_male, 0.0, np.where(is_female, 1.0, 0.0))
    base_ch_scores = 1.0 / (
        1.0 + np.exp(-base_model.decision_function(np.column_stack([age_fill, sex_num])))
    )
    base_ch = bootstrap_metrics(ch["y"], base_ch_scores, n_boot=args.n_boot, seed=args.seed)
    base_ch.update({"label": "Age+sex baseline", "is_baseline": True})
    subgroup_rows.append(base_ch)

    delta_ch = paired_delta_auroc(
        ch["y"], ch["score"], base_ch_scores, n_boot=args.n_boot, seed=args.seed
    )
    base_pt = bootstrap_metrics(
        base["test_y"], base["test_logits"], n_boot=args.n_boot, seed=args.seed
    )
    delta_pt = paired_delta_auroc(
        base["test_y"], pt["score"], base["test_logits"], n_boot=args.n_boot, seed=args.seed
    )

    # ------------------------------ figures ------------------------------- #
    plot_roc_pr(
        ch["y"],
        ch["score"],
        pt["y"],
        pt["score"],
        overall,
        FIG_DIR / "chapman_ext_roc_pr.png",
    )
    plot_calibration(
        ch["y"],
        ch["score"],
        ch["score_cal"],
        ece_raw,
        ece_cal,
        FIG_DIR / "chapman_ext_calibration.png",
    )
    plot_subgroups(
        subgroup_rows,
        internal["auroc"],
        FIG_DIR / "chapman_ext_subgroups.png",
    )

    # ---------------------------- predictions ----------------------------- #
    preds = pd.DataFrame(
        {
            "record": ch["record"],
            "y_hyp": ch["y"].astype(int),
            "y_lvh": ch["y_lvh"].astype(int),
            "age": ch["age"],
            "sex": ch["sex"],
            "logit": ch_logits,
            "score": ch["score"],
            "score_cal": ch["score_cal"],
            "baseline_age_sex": base_ch_scores,
        }
    )
    preds.to_csv(OUT_DIR / "predictions.csv", index=False)

    # ------------------------------ tables -------------------------------- #
    def op_for(sens_name: str = "raw_sens_ge_0.90", spec_name: str = "raw_spec_ge_0.90"):
        hi = ops[sens_name]
        sp = ops[spec_name]
        return hi["sensitivity"], hi["specificity"], sp["specificity"], sp["sensitivity"]

    def table_row(label, m, target_prev, sens_thr, spec_thr, report_ops: bool = True):
        row = {
            "cohort": label,
            "n": m["n"],
            "n_pos": m["n_pos"],
            "prevalence_sample": round(m["pos_rate"], 4),
            "auroc": round(m["auroc"], 4),
            "auroc_95ci": f"{m['auroc_ci95'][0]:.3f}-{m['auroc_ci95'][1]:.3f}",
            "auprc": round(m["auprc"], 4),
            "auprc_95ci": f"{m['auprc_ci95'][0]:.3f}-{m['auprc_ci95'][1]:.3f}",
            "sens_at_sens90": "—",
            "spec_at_sens90": "—",
            "spec_at_spec90": "—",
            "sens_at_spec90": "—",
            "target_prevalence": round(target_prev, 5),
            "ppv_target_prev": "—",
            "npv_target_prev": "—",
        }
        if report_ops:
            row.update(
                {
                    "sens_at_sens90": round(sens_thr, 4),
                    "spec_at_sens90": round(spec_thr[0], 4),
                    "spec_at_spec90": round(spec_thr[1], 4),
                    "sens_at_spec90": round(spec_thr[2], 4),
                    "ppv_target_prev": round(
                        ppv_npv_at_prevalence(sens_thr, spec_thr[0], target_prev)[0], 4
                    ),
                    "npv_target_prev": round(
                        ppv_npv_at_prevalence(sens_thr, spec_thr[0], target_prev)[1], 4
                    ),
                }
            )
        return row

    ch_sens, ch_spec_hi, ch_spec_lo, ch_sens_lo = op_for()
    internal_op = operating_point(pt["y"], pt["score"], THRESHOLDS_RAW["sens_ge_0.90"])
    internal_op_spec = operating_point(pt["y"], pt["score"], THRESHOLDS_RAW["spec_ge_0.90"])
    pt_prev = float(pt["y"].mean())
    table2 = pd.DataFrame(
        [
            {
                "cohort": "PTB-XL test (fold 10)",
                "n": int(len(pt["y"])),
                "n_pos": int(pt["y"].sum()),
                "prevalence_sample": round(pt_prev, 4),
                "auroc": round(internal["auroc"], 4),
                "auroc_95ci": f"{internal['auroc_ci95'][0]:.3f}-{internal['auroc_ci95'][1]:.3f}",
                "auprc": round(internal["auprc"], 4),
                "auprc_95ci": f"{internal['auprc_ci95'][0]:.3f}-{internal['auprc_ci95'][1]:.3f}",
                "sens_at_sens90": round(internal_op["sensitivity"], 4),
                "spec_at_sens90": round(internal_op["specificity"], 4),
                "spec_at_spec90": round(internal_op_spec["specificity"], 4),
                "sens_at_spec90": round(internal_op_spec["sensitivity"], 4),
                "target_prevalence": round(pt_prev, 4),
                "ppv_target_prev": round(
                    ppv_npv_at_prevalence(
                        internal_op["sensitivity"], internal_op["specificity"], pt_prev
                    )[0],
                    4,
                ),
                "npv_target_prev": round(
                    ppv_npv_at_prevalence(
                        internal_op["sensitivity"], internal_op["specificity"], pt_prev
                    )[1],
                    4,
                ),
            },
            table_row(
                "Chapman-Shaoxing (case-control)",
                overall,
                CHAPMAN_FULL_PREV,
                ch_sens,
                (ch_spec_hi, ch_spec_lo, ch_sens_lo),
            ),
            table_row(
                "Chapman age+sex baseline",
                base_ch,
                CHAPMAN_FULL_PREV,
                None,
                (None, None, None),
                report_ops=False,
            ),
        ]
    )
    table2.to_csv(OUT_DIR / "table2_external_validation.csv", index=False)

    table3 = pd.DataFrame(
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
    )
    table3.to_csv(OUT_DIR / "table3_subgroups.csv", index=False)

    # ------------------------------ markdown ------------------------------ #
    def md_table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)

    summary = f"""# Chapman-Shaoxing external validation — result package

## Cohort
- 45,152 header records scanned; 3,227 readable 12-lead waveforms with local `.mat`.
- All **769** HYP positives (LVH 647, RVH/RAH 122) + **2,000** negatives sampled with seed 42.
- Case-control prevalence {overall['pos_rate']:.1%}; full-library prevalence {CHAPMAN_FULL_PREV:.2%}.
- Mean age {np.nanmean(ch['age']):.1f} y; male {np.mean(is_male):.1%}.

## Discrimination
- AUROC **{overall['auroc']:.3f}** (95% CI {overall['auroc_ci95'][0]:.3f}–{overall['auroc_ci95'][1]:.3f}).
- AUPRC {overall['auprc']:.3f} (95% CI {overall['auprc_ci95'][0]:.3f}–{overall['auprc_ci95'][1]:.3f});
  not directly comparable to the PTB-XL AUPRC because of case-control sampling.
- Age+sex baseline AUROC {base_ch['auroc']:.3f}; ΔAUROC CNN−baseline
  **{delta_ch['delta_auroc']:+.3f}** (95% CI {delta_ch['delta_ci95'][0]:+.3f} to {delta_ch['delta_ci95'][1]:+.3f}).

## Calibration
- ECE raw {ece_raw:.3f} → temperature-scaled {ece_cal:.3f} (T={TEMPERATURE:.3f}).
- Brier raw {cal['brier_raw']:.3f} → {cal['brier_cal']:.3f}.

## Operating points (PTB-XL-val-locked, applied frozen)
- Sensitivity ≥0.90 threshold {THRESHOLDS_RAW['sens_ge_0.90']:.3f}: sens
  {ops['raw_sens_ge_0.90']['sensitivity']:.3f}, spec
  {ops['raw_sens_ge_0.90']['specificity']:.3f}.
- Specificity ≥0.90 threshold {THRESHOLDS_RAW['spec_ge_0.90']:.3f}: sens
  {ops['raw_spec_ge_0.90']['sensitivity']:.3f}, spec
  {ops['raw_spec_ge_0.90']['specificity']:.3f}.
- Projected at full prevalence {CHAPMAN_FULL_PREV:.2%}: PPV
  {ops['raw_sens_ge_0.90']['ppv_at_full_prevalence']:.3f}, NPV
  {ops['raw_sens_ge_0.90']['npv_at_full_prevalence']:.3f} (high-sensitivity point).

## Table 2 — overall
{md_table(table2)}

## Table 3 — subgroups
{md_table(table3)}

## Figures
- `figures/chapman_ext_roc_pr.png`
- `figures/chapman_ext_calibration.png`
- `figures/chapman_ext_subgroups.png`
"""
    (OUT_DIR / "results_summary.md").write_text(summary, encoding="utf-8")

    # ------------------------------ JSON ---------------------------------- #
    payload = {
        "cohort": {
            "n_labeled": ch["n_labeled"],
            "n_unreadable": ch["n_unreadable"],
            "n_used": overall["n"],
            "n_positive_total": ch["n_positive_total"],
            "n_lvh_positive": int(ch["y_lvh"].sum()),
            "n_rvh_rah_positive": int(((ch["y"] == 1) & (ch["y_lvh"] == 0)).sum()),
            "case_control_prevalence": overall["pos_rate"],
            "full_library_prevalence": CHAPMAN_FULL_PREV,
            "mean_age": float(np.nanmean(ch["age"])),
            "pct_male": float(np.mean(is_male)),
            "age_missing_or_zero": int((~age_ok).sum()),
        },
        "discrimination": {"chapman": overall, "ptbxl_test": internal},
        "calibration": cal,
        "operating_points": ops,
        "subgroups": subgroup_rows,
        "baseline_age_sex": {
            "chapman": base_ch,
            "ptbxl_test": base_pt,
            "delta_chapman": delta_ch,
            "delta_ptbxl_test": delta_pt,
        },
        "thresholds_raw": THRESHOLDS_RAW,
        "thresholds_calibrated": THRESHOLDS_CAL,
    }
    (OUT_DIR / "external_analysis.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8"
    )

    print("\n==== Chapman external analysis ====")
    print(f"AUROC {overall['auroc']:.4f} ({overall['auroc_ci95'][0]:.4f}-{overall['auroc_ci95'][1]:.4f})")
    print(f"AUPRC {overall['auprc']:.4f} ({overall['auprc_ci95'][0]:.4f}-{overall['auprc_ci95'][1]:.4f})")
    print(f"ECE raw={ece_raw:.3f} cal={ece_cal:.3f}")
    print(f"baseline AUROC={base_ch['auroc']:.4f}; delta={delta_ch['delta_auroc']:+.4f}")
    print(f"wrote {OUT_DIR / 'external_analysis.json'}")
    print(f"wrote {OUT_DIR / 'results_summary.md'}")
    print(f"wrote {FIG_DIR / 'chapman_ext_roc_pr.png'}")


if __name__ == "__main__":
    main()
