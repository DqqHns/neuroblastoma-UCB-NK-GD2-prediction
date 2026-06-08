# -*- coding: utf-8 -*-
"""
step5d_threshold_confusion.py

Purpose:
- Choose a decision threshold from patient-level OOF probabilities (y_prob_mean)
  using Youden's J (max TPR - FPR) on the ALL cohort.
- Compute confusion matrices + key classification metrics for:
    * all
    * group0
    * group1
- Save:
    * threshold_summary.csv
    * confusion_metrics.csv
    * confusion_*.png (for each subset & threshold)

Notes:
- This is a post-hoc visualization/reporting step based on patient-level mean OOF predictions.
- It does NOT change any modeling logic.

Run:
    python step5d_threshold_confusion.py

Optional env vars:
    DETAILS_XLSX : path to strict_group_compare_details_v2.xlsx
    OUTDIR       : output directory
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, confusion_matrix


DETAILS_XLSX = os.environ.get("DETAILS_XLSX", "compare_groups_results_strict/strict_group_compare_details_v2.xlsx")
OUTDIR = os.environ.get("OUTDIR", "compare_groups_results_strict/plots_step5d")
PATIENT_SHEET_CANDIDATES = ["patient_level_mean_oof", "patient_level", "patient_oof"]


def _ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def _set_chinese_font():
    try:
        import matplotlib
        matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _pick_patient_sheet(sheets: dict):
    lower_map = {k.lower(): k for k in sheets.keys()}
    for cand in PATIENT_SHEET_CANDIDATES:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for k in sheets.keys():
        if "patient" in k.lower():
            return k
    return None


def _require_columns(df: pd.DataFrame, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[ERROR] {name} missing columns: {missing}. Available: {list(df.columns)}")


def youden_threshold(y_true, y_prob):
    """
    Choose threshold that maximizes Youden's J = TPR - FPR.
    Returns threshold, TPR, FPR.
    """
    y_true = np.asarray(y_true, int)
    y_prob = np.asarray(y_prob, float)

    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan

    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    i = int(np.nanargmax(j))
    return float(thr[i]), float(tpr[i]), float(fpr[i])


def metrics_from_cm(cm):
    """
    cm = [[tn, fp],
          [fn, tp]]
    """
    tn, fp, fn, tp = cm.ravel()
    n = tn + fp + fn + tp
    acc = (tp + tn) / n if n else np.nan
    sens = tp / (tp + fn) if (tp + fn) else np.nan  # recall/TPR
    spec = tn / (tn + fp) if (tn + fp) else np.nan  # TNR
    ppv = tp / (tp + fp) if (tp + fp) else np.nan   # precision
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    return {
        "n": int(n),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": float(acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(ppv),
        "npv": float(npv),
        "f1": float(f1),
    }


def plot_confusion(cm, title, out_png):
    """
    Plot a confusion matrix with counts and row-normalized percentages.
    """
    tn, fp, fn, tp = cm.ravel()
    cm_counts = np.array([[tn, fp], [fn, tp]], dtype=float)

    row_sums = cm_counts.sum(axis=1, keepdims=True)
    cm_row = np.divide(cm_counts, row_sums, out=np.zeros_like(cm_counts), where=row_sums != 0)

    fig = plt.figure(figsize=(5.2, 4.6))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm_row)  # default colormap

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            cnt = int(cm_counts[i, j])
            pct = cm_row[i, j] * 100.0 if row_sums[i, 0] else 0.0
            ax.text(j, i, f"{cnt}\n{pct:.1f}%", ha="center", va="center")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close(fig)


def run_for_threshold(pat: pd.DataFrame, thr: float, thr_name: str):
    results = []
    for subset_name, df_sub in [
        ("all", pat),
        ("group0", pat[pat["group01"] == 0]),
        ("group1", pat[pat["group01"] == 1]),
    ]:
        y_true = df_sub["y_true"].to_numpy(int)
        y_prob = df_sub["y_prob_mean"].to_numpy(float)
        y_pred = (y_prob >= thr).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape != (2, 2):
            full = np.zeros((2, 2), dtype=int)
            full[:cm.shape[0], :cm.shape[1]] = cm
            cm = full

        m = metrics_from_cm(cm)
        m.update({"subset": subset_name, "threshold_name": thr_name, "threshold": float(thr)})
        results.append((subset_name, cm, m))
    return results


def main():
    _set_chinese_font()
    _ensure_outdir(OUTDIR)

    if not os.path.exists(DETAILS_XLSX):
        raise FileNotFoundError(f"[ERROR] Cannot find DETAILS_XLSX: {DETAILS_XLSX}")

    sheets = pd.read_excel(DETAILS_XLSX, sheet_name=None)
    patient_sheet = _pick_patient_sheet(sheets)
    if patient_sheet is None:
        raise ValueError(f"[ERROR] No patient sheet found. Sheets={list(sheets.keys())}")

    pat = sheets[patient_sheet].copy()
    _require_columns(pat, ["group01", "y_true", "y_prob_mean"], f"patient sheet={patient_sheet}")

    pat["group01"] = pd.to_numeric(pat["group01"], errors="coerce")
    pat["y_true"] = pd.to_numeric(pat["y_true"], errors="coerce")
    pat["y_prob_mean"] = pd.to_numeric(pat["y_prob_mean"], errors="coerce")
    pat = pat.dropna(subset=["group01", "y_true", "y_prob_mean"]).copy()
    pat["group01"] = pat["group01"].astype(int)
    pat["y_true"] = pat["y_true"].astype(int)

    print(f"[INFO] Patient sheet: {patient_sheet}")
    print(f"[INFO] N(all)={len(pat)}  N(g0)={(pat.group01==0).sum()}  N(g1)={(pat.group01==1).sum()}")

    thr_y, tpr_y, fpr_y = youden_threshold(pat["y_true"], pat["y_prob_mean"])

    thr_rows = [
        {"threshold_name": "youden_all", "threshold": thr_y, "tpr_at_thr": tpr_y, "fpr_at_thr": fpr_y},
        {"threshold_name": "fixed_0.5", "threshold": 0.5, "tpr_at_thr": np.nan, "fpr_at_thr": np.nan},
    ]
    thr_df = pd.DataFrame(thr_rows)
    thr_csv = os.path.join(OUTDIR, "threshold_summary.csv")
    thr_df.to_csv(thr_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved: {thr_csv}")
    print(thr_df)

    all_metrics = []
    for thr_name, thr in [("youden_all", thr_y), ("fixed_0.5", 0.5)]:
        if not np.isfinite(thr):
            print(f"[WARN] Threshold {thr_name} is NaN (single-class issue?). Skipping.")
            continue
        res = run_for_threshold(pat, thr, thr_name)
        for subset_name, cm, m in res:
            out_png = os.path.join(OUTDIR, f"confusion_{subset_name}_{thr_name}.png")
            plot_confusion(cm, f"Confusion matrix ({subset_name}, thr={thr:.3f} [{thr_name}])", out_png)
            print(f"[DONE] Saved: {out_png}")
            all_metrics.append(m)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv = os.path.join(OUTDIR, "confusion_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved: {metrics_csv}")


if __name__ == "__main__":
    main()
