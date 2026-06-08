# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix


DETAILS_XLSX = os.environ.get("DETAILS_XLSX", "compare_groups_results_strict/strict_group_compare_details_v2.xlsx")
OUTDIR = os.environ.get("OUTDIR", "compare_groups_results_strict/plots_step5e")
PATIENT_SHEET_CANDIDATES = ["patient_level_mean_oof", "patient_level", "patient_oof"]

SHOW_PREVALENCE_LINES = False


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
    """Return (threshold, tpr, fpr) that maximizes Youden's J = TPR - FPR."""
    y_true = np.asarray(y_true, int)
    y_prob = np.asarray(y_prob, float)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan
    fpr, tpr, thrs = roc_curve(y_true, y_prob)
    j = tpr - fpr
    i = int(np.nanargmax(j))
    return float(thrs[i]), float(tpr[i]), float(fpr[i])


def point_metrics_at_threshold(y_true, y_prob, thr):
    """
    Compute exact point metrics at a given threshold using confusion matrix:
      - TPR (Sensitivity), FPR, Precision, Recall, Specificity, Accuracy
    """
    y_true = np.asarray(y_true, int)
    y_prob = np.asarray(y_prob, float)
    y_pred = (y_prob >= thr).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        full = np.zeros((2, 2), dtype=int)
        full[:cm.shape[0], :cm.shape[1]] = cm
        cm = full
    tn, fp, fn, tp = cm.ravel()

    tpr = tp / (tp + fn) if (tp + fn) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    rec = tpr
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else np.nan

    return {
        "threshold": float(thr),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "accuracy": float(acc),
    }


def compute_curves(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    prevalence = float(np.mean(np.asarray(y_true, int))) if len(y_true) else np.nan

    return {
        "roc": (fpr, tpr, roc_auc),
        "pr": (recall, precision, ap),
        "prevalence": prevalence,
    }


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

    # numeric + drop missing
    pat["group01"] = pd.to_numeric(pat["group01"], errors="coerce")
    pat["y_true"] = pd.to_numeric(pat["y_true"], errors="coerce")
    pat["y_prob_mean"] = pd.to_numeric(pat["y_prob_mean"], errors="coerce")
    pat = pat.dropna(subset=["group01", "y_true", "y_prob_mean"]).copy()
    pat["group01"] = pat["group01"].astype(int)
    pat["y_true"] = pat["y_true"].astype(int)

    print(f"[INFO] Patient sheet: {patient_sheet}")
    print(f"[INFO] N(all)={len(pat)}  N(g0)={(pat.group01==0).sum()}  N(g1)={(pat.group01==1).sum()}")

    # thresholds
    thr_fixed = 0.5
    thr_youden, youden_tpr, youden_fpr = youden_threshold(pat["y_true"], pat["y_prob_mean"])

 
    thr_label_map = {
        "fixed_0.5": f"thr={thr_fixed:.3f}",  # 固定阈值显示为 thr=0.500
        "youden_all": f"thr={thr_youden:.3f}" if np.isfinite(thr_youden) else "thr=nan"  # Youden阈值显示为具体数值
    }

    thr_df = pd.DataFrame([
        {"threshold_name": "fixed_0.5", "threshold": thr_fixed},
        {"threshold_name": "youden_all", "threshold": thr_youden, "tpr_at_thr": youden_tpr, "Understanding": "chosen on ALL by max(TPR-FPR)", "fpr_at_thr": youden_fpr},
    ])
    thr_path = os.path.join(OUTDIR, "thresholds_used.csv")
    thr_df.to_csv(thr_path, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved: {thr_path}")
    print(thr_df)

    cohorts = [
        ("all", pat),
        ("group0", pat[pat["group01"] == 0]),
        ("group1", pat[pat["group01"] == 1]),
    ]

    # Collect point metrics for each cohort x threshold
    rows = []
    for cohort_name, df_sub in cohorts:
        y_true = df_sub["y_true"].to_numpy(int)
        y_prob = df_sub["y_prob_mean"].to_numpy(float)

        cur = compute_curves(y_true, y_prob)
        roc_auc = cur["roc"][2]
        ap = cur["pr"][2]
        prevalence = cur["prevalence"]

        for thr_name, thr in [("fixed_0.5", thr_fixed), ("youden_all", thr_youden)]:
            m = point_metrics_at_threshold(y_true, y_prob, thr)
            m.update({
                "cohort": cohort_name,
                "threshold_name": thr_name,
                "roc_auc": float(roc_auc),
                "ap": float(ap),
                "prevalence": float(prevalence),
                "n": int(len(df_sub)),
                "pos_n": int((y_true == 1).sum()),
                "neg_n": int((y_true == 0).sum()),
            })
            rows.append(m)

    points_df = pd.DataFrame(rows)
    points_path = os.path.join(OUTDIR, "threshold_points_metrics.csv")
    points_df.to_csv(points_path, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved: {points_path}")

    # ===== ROC plot =====
    fig = plt.figure(figsize=(9.5, 7.2))
    ax = fig.add_subplot(111)

    marker_map = {"fixed_0.5": "o", "youden_all": "^"}

    # plot curves (one per cohort)
    for cohort_name, df_sub in cohorts:
        y_true = df_sub["y_true"].to_numpy(int)
        y_prob = df_sub["y_prob_mean"].to_numpy(float)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{cohort_name} curve (AUC={roc_auc:.3f})")

    # add points AFTER curves (clearer)
    for cohort_name, _ in cohorts:
        for thr_name in ["fixed_0.5", "youden_all"]:
            r = points_df[(points_df["cohort"] == cohort_name) & (points_df["threshold_name"] == thr_name)].iloc[0]
     
            ax.scatter([r["fpr"]], [r["tpr"]], s=70, marker=marker_map[thr_name],
                       label=f"{cohort_name} @ {thr_label_map[thr_name]} (TPR={r['tpr']:.3f}, FPR={r['fpr']:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("ROC curves")  # threshold points (patient-level mean OOF)
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    out_roc = os.path.join(OUTDIR, "roc_with_threshold_points.png")
    plt.savefig(out_roc, dpi=220)
    plt.close(fig)
    print(f"[DONE] Saved: {out_roc}")

    # ===== PR plot =====
    fig = plt.figure(figsize=(9.5, 7.2))
    ax = fig.add_subplot(111)

    # plot curves
    for cohort_name, df_sub in cohorts:
        y_true = df_sub["y_true"].to_numpy(int)
        y_prob = df_sub["y_prob_mean"].to_numpy(float)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, lw=2, label=f"{cohort_name} curve (AP={ap:.3f})")

        if SHOW_PREVALENCE_LINES:
            prev = float(np.mean(y_true)) if len(y_true) else np.nan
            if np.isfinite(prev):
                ax.axhline(y=prev, linestyle="--", lw=1, label=f"{cohort_name} prevalence={prev:.3f}")

    # add points
    for cohort_name, _ in cohorts:
        for thr_name in ["fixed_0.5", "youden_all"]:
            r = points_df[(points_df["cohort"] == cohort_name) & (points_df["threshold_name"] == thr_name)].iloc[0]

            ax.scatter([r["recall"]], [r["precision"]], s=70, marker=marker_map[thr_name],
                       label=f"{cohort_name} @ {thr_label_map[thr_name]} (P={r['precision']:.3f}, R={r['recall']:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR curve")   #  threshold points (patient-level mean OOF)
    ax.grid(True)
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    out_pr = os.path.join(OUTDIR, "pr_with_threshold_points.png")
    plt.savefig(out_pr, dpi=220)
    plt.close(fig)
    print(f"[DONE] Saved: {out_pr}")

    print("[DONE] All outputs saved under:", OUTDIR)


if __name__ == "__main__":
    main()