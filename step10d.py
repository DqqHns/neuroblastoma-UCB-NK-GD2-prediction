#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve

# =========================
# ====== USER CONFIG ======
# =========================
OOF_CSV = r"G:/Dqq_code/nk_model(1.8)/compare_methods_results_new_COMPLETE_CASE = True/oof_pred_mean.csv"
OOF_PROBA_COL = "M14_LogReg_L2"
PATIENT_ID_COL = "patient_id"

FEATURE_XLSX = r"G:/Dqq_code/nk_model(1.8)/data/DATA.xlsx"
FEATURE_SHEET = "patient_features"

POS_LABEL = 1
N_BOOT = 2000
RANDOM_SEED = 42

FIG_DPI = 180
CALIBRATION_BINS = 5
OUT_DIR = r"G:/Dqq_code/nk_model(1.8)/outputs_step12d"
# =========================
# ====== END CONFIG =======
# =========================

LABEL_CANDIDATES = [
    "y_true", "label", "target", "outcome", "event", "Y",
    "Status", "status", "NK", "nk", "case", "Case",
    "class", "Class"
]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _find_label_col(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)
    for c in LABEL_CANDIDATES:
        if c in cols:
            return c
    lower = {c.lower(): c for c in cols}
    for cand in LABEL_CANDIDATES:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for lc, orig in lower.items():
        if any(k in lc for k in ["y_true", "label", "target", "outcome", "event", "class"]):
            return orig
    return None


def _as_binary(y: pd.Series, pos_label=1) -> pd.Series:
    if y.dtype == object:
        y2 = y.astype(str).str.strip().str.lower()
        if pos_label in [1, "1", "pos", "positive", "case", "yes", "true"]:
            pos_set = {"1", "pos", "positive", "case", "yes", "true"}
        else:
            pos_set = {str(pos_label).strip().lower()}
        return y2.isin(pos_set).astype(int)
    y_num = pd.to_numeric(y, errors="coerce")
    if y_num.isna().any():
        return (y == pos_label).astype(int)
    return (y_num == pos_label).astype(int)


@dataclass
class Metrics:
    auc: float
    ap: float
    acc: float
    f1: float
    sens: float
    spec: float
    brier: float


def _compute_metrics(y_true: np.ndarray, p: np.ndarray, thr: float = 0.5) -> Metrics:
    y_true = y_true.astype(int)
    p = np.asarray(p, dtype=float)

    auc = np.nan
    ap = np.nan
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, p)
        ap = average_precision_score(y_true, p)

    y_pred = (p >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    brier = brier_score_loss(y_true, p)
    return Metrics(auc=auc, ap=ap, acc=acc, f1=f1, sens=sens, spec=spec, brier=brier)


def _ci(arr: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float]:
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    med = float(np.median(arr))
    lo = float(np.quantile(arr, alpha / 2))
    hi = float(np.quantile(arr, 1 - alpha / 2))
    return med, lo, hi


def _plot_metric_ci(summary: pd.DataFrame, out_png: str) -> None:
    plt.figure(figsize=(8.5, 4.8), dpi=FIG_DPI)
    y = np.arange(len(summary))
    x = summary["median"].values
    xerr = np.vstack([x - summary["lo"].values, summary["hi"].values - x])
    plt.errorbar(x, y, xerr=xerr, fmt="o", capsize=3)
    plt.yticks(y, summary["metric"].values)
    plt.xlabel("Bootstrap median with 95% CI")
    plt.title("Performance metrics (bootstrap)")
    plt.grid(True, axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def _plot_roc_pr(y_true: np.ndarray, p: np.ndarray, out_roc: str, out_pr: str) -> None:
    y_true = y_true.astype(int)
    if len(np.unique(y_true)) != 2:
        return

    fpr, tpr, _ = roc_curve(y_true, p)
    auc = roc_auc_score(y_true, p)

    plt.figure(figsize=(5.6, 5.0), dpi=FIG_DPI)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC curve (AUC={auc:.3f})")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_roc)
    plt.close()

    prec, rec, _ = precision_recall_curve(y_true, p)
    ap = average_precision_score(y_true, p)

    plt.figure(figsize=(5.6, 5.0), dpi=FIG_DPI)
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR curve (AP={ap:.3f})")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_pr)
    plt.close()


def _plot_calibration_with_bootstrap_band(
    y_true: np.ndarray,
    p: np.ndarray,
    boot_ps: List[np.ndarray],
    out_png: str,
    n_bins: int = 5,
) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy="quantile")

    xs = np.linspace(0.0, 1.0, 101)
    curves = []
    for pb in boot_ps:
        try:
            fp_b, mp_b = calibration_curve(y_true, pb, n_bins=n_bins, strategy="quantile")
            curves.append(np.interp(xs, mp_b, fp_b, left=np.nan, right=np.nan))
        except Exception:
            continue

    curves = np.array(curves, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        lo = np.nanquantile(curves, 0.025, axis=0) if curves.size else None
        hi = np.nanquantile(curves, 0.975, axis=0) if curves.size else None

    plt.figure(figsize=(5.8, 5.2), dpi=FIG_DPI)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(mean_pred, frac_pos, marker="o")
    if lo is not None and hi is not None:
        plt.fill_between(xs, lo, hi, alpha=0.2)

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraction of positives")
    plt.title("Calibration curve (with bootstrap 95% band)")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main() -> None:
    np.random.seed(RANDOM_SEED)
    _ensure_dir(OUT_DIR)

    print("[1/6] Loading OOF predictions ...")
    oof = pd.read_csv(OOF_CSV)
    if PATIENT_ID_COL not in oof.columns:
        raise ValueError(f"OOF CSV must contain '{PATIENT_ID_COL}' column.")
    if OOF_PROBA_COL not in oof.columns:
        raise ValueError(f"OOF CSV must contain probability column '{OOF_PROBA_COL}'.")

    label_col = _find_label_col(oof)
    if label_col is None:
        print("  - y_true not found in OOF; trying patient_features sheet ...")
        feat = pd.read_excel(FEATURE_XLSX, sheet_name=FEATURE_SHEET)
        if PATIENT_ID_COL not in feat.columns:
            raise ValueError(f"patient_features must contain '{PATIENT_ID_COL}' column.")
        label_col2 = _find_label_col(feat)
        if label_col2 is None:
            raise ValueError(
                "Could not auto-detect label column in OOF or patient_features. "
                "Please add a y_true/label/target/outcome column."
            )
        feat = feat[[PATIENT_ID_COL, label_col2]].copy()
        oof = oof.merge(feat, on=PATIENT_ID_COL, how="inner")
        label_col = label_col2

    df = oof[[PATIENT_ID_COL, label_col, OOF_PROBA_COL]].copy()
    df = df.dropna(subset=[label_col, OOF_PROBA_COL])
    y = _as_binary(df[label_col], pos_label=POS_LABEL).to_numpy()
    p = df[OOF_PROBA_COL].astype(float).to_numpy()

    print(f"  - N used: {len(df)} ; positives: {int(y.sum())} ; pos_label={POS_LABEL}")

    print("[2/6] Baseline metrics ...")
    base = _compute_metrics(y, p, thr=0.5)
    base_df = pd.DataFrame([base.__dict__])
    base_df.to_csv(os.path.join(OUT_DIR, "step12d_baseline_metrics.csv"), index=False)

    print("[3/6] Bootstrap metrics ...")
    n = len(y)
    rng = np.random.RandomState(RANDOM_SEED)
    rows = []
    boot_ps_for_cal = []
    for b in range(N_BOOT):
        idx = rng.randint(0, n, size=n)
        yb = y[idx]
        pb = p[idx]
        m = _compute_metrics(yb, pb, thr=0.5)
        rows.append(m.__dict__)
        if b < 400:
            boot_ps_for_cal.append(pb)

    boot = pd.DataFrame(rows)

    print("[4/6] Summaries + save tables ...")
    summary_rows = []
    for metric in ["auc", "ap", "acc", "f1", "sens", "spec", "brier"]:
        med, lo, hi = _ci(boot[metric].to_numpy())
        summary_rows.append({"metric": metric, "median": med, "lo": lo, "hi": hi})
    summary = pd.DataFrame(summary_rows)

    boot.to_csv(os.path.join(OUT_DIR, "step12d_metrics_bootstrap.csv"), index=False)

    xlsx_path = os.path.join(OUT_DIR, "step12d_metrics_bootstrap.xlsx")
    with pd.ExcelWriter(xlsx_path) as w:
        base_df.to_excel(w, sheet_name="baseline", index=False)
        summary.to_excel(w, sheet_name="bootstrap_summary", index=False)
        boot.to_excel(w, sheet_name="bootstrap_samples", index=False)

    print("[5/6] Plots ...")
    _plot_metric_ci(summary, os.path.join(OUT_DIR, "fig_metric_ci.png"))
    _plot_roc_pr(y, p, os.path.join(OUT_DIR, "fig_roc.png"), os.path.join(OUT_DIR, "fig_pr.png"))
    _plot_calibration_with_bootstrap_band(
        y_true=y,
        p=p,
        boot_ps=boot_ps_for_cal,
        out_png=os.path.join(OUT_DIR, "fig_calibration.png"),
        n_bins=CALIBRATION_BINS,
    )

    print("[6/6] Done. Outputs saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
