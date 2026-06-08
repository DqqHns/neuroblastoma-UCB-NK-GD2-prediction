# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve


# -----------------------------
# Config (edit if needed)
# -----------------------------
DETAILS_XLSX = os.environ.get("DETAILS_XLSX", "compare_groups_results_strict/strict_group_compare_details_v2.xlsx")
OUTDIR = os.environ.get("OUTDIR", "compare_groups_results_strict/plots_step5c")
PATIENT_SHEET_CANDIDATES = [
    "patient_level_mean_oof",
    "patient_level",
    "patient_oof",
]
LOO_SHEET_CANDIDATES = [
    "group1_loo_influence",
    "loo_influence_group1",
    "loo_group1",
]


# -----------------------------
# Helpers
# -----------------------------
def _set_chinese_font():
    """Try setting common Chinese fonts; if missing, plots still save."""
    try:
        import matplotlib
        matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def _pick_sheet(sheets: dict, candidates: list, fallback_contains: str):
    """Pick a sheet name from an Excel sheet dict."""
    lower_map = {k.lower(): k for k in sheets.keys()}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for k in sheets.keys():
        if fallback_contains.lower() in k.lower():
            return k
    return None


def _require_columns(df: pd.DataFrame, cols: list, df_name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[ERROR] {df_name} missing columns: {missing}. Available: {list(df.columns)}")


def _safe_group(df: pd.DataFrame, group01: int):
    return df[df["group01"] == group01].copy()


def _metric_block(y_true, y_prob):
    y_true = np.asarray(y_true, int)
    y_prob = np.asarray(y_prob, float)
    out = {"roc_auc": np.nan, "ap": np.nan, "brier": np.nan, "n": int(len(y_true)), "pos": int(y_true.sum())}
    if len(np.unique(y_true)) >= 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        out["roc_auc"] = float(auc(fpr, tpr))
        out["ap"] = float(average_precision_score(y_true, y_prob))
    try:
        out["brier"] = float(brier_score_loss(y_true, y_prob))
    except Exception:
        out["brier"] = np.nan
    return out


def _plot_roc(ax, y_true, y_prob, label_prefix: str):
    y_true = np.asarray(y_true, int)
    y_prob = np.asarray(y_prob, float)
    if len(np.unique(y_true)) < 2:
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_title(f"ROC ({label_prefix}) - only one class")
        return {"roc_auc": np.nan}
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{label_prefix} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    return {"roc_auc": float(roc_auc)}


def _plot_pr(ax, y_true, y_prob, label_prefix: str):
    y_true = np.asarray(y_true, int)
    y_prob = np.asarray(y_prob, float)
    if len(np.unique(y_true)) < 2:
        ax.set_title(f"PR ({label_prefix}) - only one class")
        return {"ap": np.nan}
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    ax.plot(recall, precision, label=f"{label_prefix} (AP={ap:.3f})")
    prev = float(np.mean(y_true))
    ax.hlines(prev, 0, 1, linestyles="--", label=f"{label_prefix} prevalence={prev:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    return {"ap": float(ap)}


def _violin(ax, data_groups, labels, title, ylabel):

    # clean
    cleaned = []
    for g in data_groups:
        arr = np.asarray(g, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        cleaned.append(arr)

    positions = np.arange(1, len(labels) + 1)

    # Try KDE violin
    try:
        from scipy.stats import gaussian_kde  # type: ignore
        for pos, arr in zip(positions, cleaned):
            if arr.size == 0:
                continue
            if arr.size == 1:
                # single point: draw a small marker line
                ax.plot([pos-0.1, pos+0.1], [arr[0], arr[0]])
                continue

            kde = gaussian_kde(arr)
            y_min, y_max = float(np.min(arr)), float(np.max(arr))
            if y_min == y_max:
                ax.plot([pos-0.15, pos+0.15], [y_min, y_min])
                continue

            y = np.linspace(y_min, y_max, 200)
            dens = kde(y)
            dens = dens / dens.max()  # normalize to 1
            width = 0.35
            ax.fill_betweenx(y, pos - dens * width, pos + dens * width, alpha=0.35)
            # show mean as marker
            ax.scatter([pos], [float(np.mean(arr))], marker="^")

        # also overlay light jittered points for transparency
        rng = np.random.default_rng(0)
        for pos, arr in zip(positions, cleaned):
            if arr.size == 0:
                continue
            jitter = rng.uniform(-0.06, 0.06, size=arr.size)
            ax.scatter(pos + jitter, arr, s=10, alpha=0.45)

    except Exception:
        # Fallback: boxplot + jitter scatter
        ax.boxplot(cleaned, positions=positions, widths=0.5, showmeans=True)
        rng = np.random.default_rng(0)
        for pos, arr in zip(positions, cleaned):
            if arr.size == 0:
                continue
            jitter = rng.uniform(-0.08, 0.08, size=arr.size)
            ax.scatter(pos + jitter, arr, s=12, alpha=0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)


def _calibration_plot(ax, y_true, y_prob, label_prefix: str, n_bins=5):
    y_true = np.asarray(y_true, int)
    y_prob = np.asarray(y_prob, float)
    if len(np.unique(y_true)) < 2:
        ax.set_title(f"Calibration ({label_prefix}) - only one class")
        return {"brier": np.nan}
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    ax.plot(mean_pred, frac_pos, marker="o", label=label_prefix)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    try:
        brier = brier_score_loss(y_true, y_prob)
    except Exception:
        brier = np.nan
    return {"brier": float(brier) if np.isfinite(brier) else np.nan}


# -----------------------------
# Main
# -----------------------------
def main():
    _set_chinese_font()
    _ensure_outdir(OUTDIR)

    if not os.path.exists(DETAILS_XLSX):
        raise FileNotFoundError(
            f"[ERROR] Cannot find DETAILS_XLSX: {DETAILS_XLSX}\n"
            f"Tip: set env DETAILS_XLSX to your file path."
        )

    sheets = pd.read_excel(DETAILS_XLSX, sheet_name=None)
    patient_sheet = _pick_sheet(sheets, PATIENT_SHEET_CANDIDATES, fallback_contains="patient")
    if patient_sheet is None:
        raise ValueError(
            f"[ERROR] Cannot find patient-level sheet in {list(sheets.keys())}. "
            f"Expected one of {PATIENT_SHEET_CANDIDATES} or contains 'patient'."
        )

    pat = sheets[patient_sheet].copy()
    _require_columns(pat, ["group01", "y_true", "y_prob_mean"], f"patient sheet={patient_sheet}")
    if "y_prob_sd" not in pat.columns:
        print("[WARN] y_prob_sd not found; uncertainty violin will be skipped.")

    pat["group01"] = pd.to_numeric(pat["group01"], errors="coerce")
    pat["y_true"] = pd.to_numeric(pat["y_true"], errors="coerce")
    pat["y_prob_mean"] = pd.to_numeric(pat["y_prob_mean"], errors="coerce")
    if "y_prob_sd" in pat.columns:
        pat["y_prob_sd"] = pd.to_numeric(pat["y_prob_sd"], errors="coerce")

    pat = pat.dropna(subset=["group01", "y_true", "y_prob_mean"]).copy()
    pat["group01"] = pat["group01"].astype(int)
    pat["y_true"] = pat["y_true"].astype(int)

    pat_all = pat
    pat_g0 = _safe_group(pat, 0)
    pat_g1 = _safe_group(pat, 1)

    print(f"[INFO] Loaded patient sheet: {patient_sheet}")
    print(f"[INFO] N(all)={len(pat_all)}  N(g0)={len(pat_g0)}  N(g1)={len(pat_g1)}")

    # Metrics summary
    metrics_rows = [
        {"subset": "all", **_metric_block(pat_all["y_true"], pat_all["y_prob_mean"])},
        {"subset": "group0", **_metric_block(pat_g0["y_true"], pat_g0["y_prob_mean"])},
        {"subset": "group1", **_metric_block(pat_g1["y_true"], pat_g1["y_prob_mean"])},
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = os.path.join(OUTDIR, "patient_level_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved metrics: {metrics_csv}")

    # ROC
    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(111)
    _plot_roc(ax, pat_all["y_true"], pat_all["y_prob_mean"], "all")
    _plot_roc(ax, pat_g0["y_true"], pat_g0["y_prob_mean"], "group0")
    _plot_roc(ax, pat_g1["y_true"], pat_g1["y_prob_mean"], "group1")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", linewidth=0.5)
    out = os.path.join(OUTDIR, "roc_patient_level.png")
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close(fig)
    print(f"[DONE] Saved: {out}")

    # PR
    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(111)
    _plot_pr(ax, pat_all["y_true"], pat_all["y_prob_mean"], "all")
    _plot_pr(ax, pat_g0["y_true"], pat_g0["y_prob_mean"], "group0")
    _plot_pr(ax, pat_g1["y_true"], pat_g1["y_prob_mean"], "group1")
    ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", linewidth=0.5)
    out = os.path.join(OUTDIR, "pr_patient_level.png")
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close(fig)
    print(f"[DONE] Saved: {out}")

    # Violin y_prob_mean by group & label
    def _subset_prob(df, g, y):
        return df[(df["group01"] == g) & (df["y_true"] == y)]["y_prob_mean"].dropna().to_numpy(float)

    data = [
        _subset_prob(pat, 0, 0),
        _subset_prob(pat, 0, 1),
        _subset_prob(pat, 1, 0),
        _subset_prob(pat, 1, 1),
    ]
    labels = ["g0_y0", "g0_y1", "g1_y0", "g1_y1"]
    fig = plt.figure(figsize=(8.5, 5.5))
    ax = fig.add_subplot(111)
    _violin(ax, data, labels, "y_prob_mean distribution by group & label", "y_prob_mean (P[effective])")
    out = os.path.join(OUTDIR, "violin_y_prob_mean_by_group_label.png")
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close(fig)
    print(f"[DONE] Saved: {out}")

    # Violin y_prob_sd by group & label
    if "y_prob_sd" in pat.columns:
        def _subset_sd(df, g, y):
            return df[(df["group01"] == g) & (df["y_true"] == y)]["y_prob_sd"].dropna().to_numpy(float)

        data_sd = [
            _subset_sd(pat, 0, 0),
            _subset_sd(pat, 0, 1),
            _subset_sd(pat, 1, 0),
            _subset_sd(pat, 1, 1),
        ]
        fig = plt.figure(figsize=(8.5, 5.5))
        ax = fig.add_subplot(111)
        _violin(ax, data_sd, labels, "y_prob_sd (uncertainty) by group & label", "y_prob_sd across repeats")
        out = os.path.join(OUTDIR, "violin_y_prob_sd_by_group_label.png")
        plt.tight_layout()
        plt.savefig(out, dpi=220)
        plt.close(fig)
        print(f"[DONE] Saved: {out}")

    # Calibration
    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(111)
    b_all = _calibration_plot(ax, pat_all["y_true"], pat_all["y_prob_mean"], "all", n_bins=5)
    b_g0 = _calibration_plot(ax, pat_g0["y_true"], pat_g0["y_prob_mean"], "group0", n_bins=5)
    b_g1 = _calibration_plot(ax, pat_g1["y_true"], pat_g1["y_prob_mean"], "group1", n_bins=5)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", linewidth=0.5)
    out = os.path.join(OUTDIR, "calibration_patient_level.png")
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close(fig)
    print(f"[DONE] Saved: {out}")

    brier_df = pd.DataFrame([
        {"subset": "all", "brier": b_all.get("brier", np.nan)},
        {"subset": "group0", "brier": b_g0.get("brier", np.nan)},
        {"subset": "group1", "brier": b_g1.get("brier", np.nan)},
    ])
    brier_csv = os.path.join(OUTDIR, "brier_scores.csv")
    brier_df.to_csv(brier_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved: {brier_csv}")

    # Optional: group1 LOO influence
    loo_sheet = _pick_sheet(sheets, LOO_SHEET_CANDIDATES, fallback_contains="loo")
    if loo_sheet is not None:
        loo = sheets[loo_sheet].copy()
        delta_col = None
        for c in loo.columns:
            if str(c).lower() in ["delta_auc", "delta", "auc_delta", "delta_g1"]:
                delta_col = c
                break
        if delta_col is None:
            for c in loo.columns:
                if "delta" in str(c).lower():
                    delta_col = c
                    break

        if delta_col is not None:
            loo = loo.dropna(subset=[delta_col]).copy()
            loo[delta_col] = pd.to_numeric(loo[delta_col], errors="coerce")
            loo = loo.dropna(subset=[delta_col]).sort_values(delta_col, ascending=False)

            label_col = None
            for c in ["patient_name", "patient_id", "id", "Name", "name"]:
                if c in loo.columns:
                    label_col = c
                    break
            if label_col is None:
                loo["idx"] = np.arange(len(loo))
                label_col = "idx"

            topk = min(20, len(loo))
            loo_top = loo.head(topk)

            fig = plt.figure(figsize=(9, max(4.5, 0.35 * topk)))
            ax = fig.add_subplot(111)
            ax.barh(range(topk), loo_top[delta_col].to_numpy(float))
            ax.set_yticks(range(topk))
            ax.set_yticklabels([str(x) for x in loo_top[label_col].tolist()])
            ax.invert_yaxis()
            ax.set_xlabel(delta_col)
            ax.set_title(f"Top {topk} group1 LOO influence (sorted)")
            ax.grid(True, axis="x", linestyle="--", linewidth=0.5)
            out = os.path.join(OUTDIR, "group1_loo_influence_top.png")
            plt.tight_layout()
            plt.savefig(out, dpi=220)
            plt.close(fig)
            print(f"[DONE] Saved: {out}")
        else:
            print(f"[WARN] Found LOO sheet '{loo_sheet}' but no delta-like column; skipped LOO plot.")
    else:
        print("[INFO] No LOO influence sheet found; skipped LOO plot.")

    print("\n[ALL DONE] Plots saved to:", OUTDIR)


if __name__ == "__main__":
    main()
