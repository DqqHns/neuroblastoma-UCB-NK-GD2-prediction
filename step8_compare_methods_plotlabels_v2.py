

from __future__ import annotations

import os
import math
import warnings
from dataclasses import dataclass

import numpy as np
from typing import Optional
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier


def _get_classes(est):
    """Return class labels if available (works for Pipeline too)."""
    cls = getattr(est, "classes_", None)
    if cls is not None:
        return list(cls)
    # Pipeline sometimes doesn't expose classes_ until fitted; try last step
    if hasattr(est, "named_steps"):
        try:
            last = list(est.named_steps.values())[-1]
            cls = getattr(last, "classes_", None)
            if cls is not None:
                return list(cls)
        except Exception:
            pass
    return None

def get_pos_proba(est, X, pos_label=1):
    """Get P(y=pos_label) in a class-order-safe way."""
    if hasattr(est, "predict_proba"):
        p = est.predict_proba(X)
        if getattr(p, "ndim", 1) == 1:
            return p
        classes = _get_classes(est)
        if classes is not None and pos_label in classes:
            j = classes.index(pos_label)
        else:
            # fallback: use last column as "positive"
            j = -1
        return p[:, j]

    if hasattr(est, "decision_function"):
        s = est.decision_function(X)
        # if multi-column, pick pos_label column when possible
        if getattr(s, "ndim", 1) > 1:
            classes = _get_classes(est)
            if classes is not None and pos_label in classes:
                j = classes.index(pos_label)
            else:
                j = -1
            s = s[:, j]
        # map margin -> probability
        return 1.0 / (1.0 + np.exp(-s))

    raise AttributeError("Estimator has neither predict_proba nor decision_function")

try:
    from xgboost import XGBClassifier  # type: ignore
    _HAS_XGB = True
except Exception:
    XGBClassifier = None  # type: ignore
    _HAS_XGB = False

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
    log_loss,
)

from scipy.stats import wilcoxon, ttest_rel

import matplotlib.pyplot as plt


# =========================
# Config
# =========================

DATA_XLSX = os.path.join("data", "DATA.xlsx")  # same convention as step3
OUTDIR = "compare_methods_results"

ID_COL = "patient_id"
LABEL_COL = "efficacy"

# M14 feature set (make sure it matches step3!)
FEATURE_SET = [
    "dyn_slope_BA_on_PCT",
    "HGB_mean",
    "RDW-SD_baseline",
    "dyn_same_dir_rate",
    "Na+_mean",
    "SAA_mean",
]

# Evaluation protocol
N_SPLITS = 3
N_REPEATS = 50
BASE_RANDOM_STATE = 42

COMPLETE_CASE = True  # set False to enable median imputation

# Reference method for paired tests (change if you want)
REFERENCE_METHOD = "M14_LogReg_L2"


# =========================
# Utilities
# =========================


def _auto_pick_sheet(xlsx_path: str, must_have: list[str]) -> str:
    """Pick the sheet that contains most of `must_have` columns."""
    xl = pd.ExcelFile(xlsx_path)
    best_sheet = xl.sheet_names[0]
    best_score = -1
    for sh in xl.sheet_names:
        try:
            head = pd.read_excel(xlsx_path, sheet_name=sh, nrows=5)
        except Exception:
            continue
        cols = set(map(str, head.columns))
        score = sum(1 for c in must_have if c in cols)
        if score > best_score:
            best_score = score
            best_sheet = sh
    print(f"[OK] Auto-picked sheet: {best_sheet} (score={best_score})")
    return best_sheet


def load_patient_table(xlsx_path: str) -> pd.DataFrame:
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(
            f"Cannot find {xlsx_path}. If you use a different path, edit DATA_XLSX in this script."
        )
    sheet = _auto_pick_sheet(xlsx_path, must_have=[ID_COL, LABEL_COL])
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    # Normalize id
    if ID_COL in df.columns:
        df[ID_COL] = df[ID_COL].astype(str).str.strip()
    return df


def select_xy(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    missing = [c for c in [ID_COL, LABEL_COL] + features if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in table: {missing}")
    use = df[[ID_COL, LABEL_COL] + features].copy()
    use = use.dropna(subset=[LABEL_COL])
    # label -> int 0/1
    y = use[LABEL_COL].astype(int)
    pid = use[ID_COL]
    X = use[features]
    if COMPLETE_CASE:
        before = len(use)
        keep = ~X.isna().any(axis=1)
        X = X.loc[keep].reset_index(drop=True)
        y = y.loc[keep].reset_index(drop=True)
        pid = pid.loc[keep].reset_index(drop=True)
        print(f"[INFO] COMPLETE_CASE=True: kept {len(y)}/{before} patients")
    else:
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        pid = pid.reset_index(drop=True)
        print(f"[INFO] COMPLETE_CASE=False: using {len(y)} patients (median imputation in pipeline)")
    return X, y, pid


def make_splits(y: np.ndarray, n_splits: int, n_repeats: int, seed: int):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    splits = []
    for i, (tr, te) in enumerate(rskf.split(np.zeros_like(y), y)):
        rep = i // n_splits
        fold = i % n_splits
        splits.append((rep, fold, tr, te))
    return splits


@dataclass
class MethodResult:
    name: str
    fold_rows: list[dict]
    oof_by_repeat: np.ndarray  # (n_repeats, n_samples)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


def _safe_brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(brier_score_loss(y_true, y_score))
    except Exception:
        return float("nan")


def _safe_logloss(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(log_loss(y_true, y_score, labels=[0, 1]))
    except Exception:
        return float("nan")

def mean_oof_scores(oof_by_repeat: np.ndarray, n_samples: Optional[int] = None) -> np.ndarray:
    """Return per-sample mean OOF probabilities across repeats.

    Handles both layouts:
      - (n_repeats, n_samples)
      - (n_samples, n_repeats)
    """
    arr = np.asarray(oof_by_repeat)
    if arr.ndim == 1:
        return arr.astype(float)
    if arr.ndim != 2:
        return np.nanmean(arr.reshape(arr.shape[0], -1), axis=0)

    if n_samples is not None:
        if arr.shape[0] == n_samples:
            # (n_samples, n_repeats) -> mean over repeats axis=1
            return np.nanmean(arr, axis=1)
        if arr.shape[1] == n_samples:
            # (n_repeats, n_samples) -> mean over repeats axis=0
            return np.nanmean(arr, axis=0)

    # Fallback heuristic: assume repeats dimension is the larger one
    return np.nanmean(arr, axis=0) if arr.shape[0] > arr.shape[1] else np.nanmean(arr, axis=1)


def build_model(method: str) -> Pipeline:
    """Return a sklearn Pipeline that outputs predict_proba."""

    # Preprocess blocks
    if COMPLETE_CASE:
        imputer = "passthrough"
    else:
        imputer = SimpleImputer(strategy="median")

    def scaled(clf):
        return Pipeline(
            steps=[
                ("imputer", imputer),
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
        )

    def unscaled(clf):
        return Pipeline(steps=[("imputer", imputer), ("clf", clf)])

    # Models
    if method == "M14_LogReg_L2":
        # L2 logistic regression, close to step3 defaults
        clf = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="liblinear",
            max_iter=5000,
            random_state=BASE_RANDOM_STATE,
        )
        return scaled(clf)

    if method == "LogReg_L1":
        clf = LogisticRegression(
            penalty="l1",
            C=1.0,
            solver="liblinear",
            max_iter=5000,
            random_state=BASE_RANDOM_STATE,
        )
        return scaled(clf)

    if method == "LinearSVM":
        # LinearSVC has no predict_proba, so calibrate within each outer-fold training set.
        base = LinearSVC(C=1.0, random_state=BASE_RANDOM_STATE)
        clf = CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)
        return scaled(clf)

    if method == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=BASE_RANDOM_STATE,
        )
        return unscaled(clf)

    if method == "GradBoost":
        clf = GradientBoostingClassifier(random_state=BASE_RANDOM_STATE)
        return unscaled(clf)

    if method == "HistGB":
        # HistGB handles missing internally, but we keep the pipeline consistent.
        clf = HistGradientBoostingClassifier(random_state=BASE_RANDOM_STATE)
        return unscaled(clf)

    if method == "XGBoost":
        if not _HAS_XGB:
            raise RuntimeError("XGBoost is not installed. Install xgboost (pip install xgboost) or remove it from METHODS.")
        clf = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_weight=1.0,
            gamma=0.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=BASE_RANDOM_STATE,
        )
        return unscaled(clf)

    raise ValueError(f"Unknown method: {method}")


def eval_one_method(name: str, X: pd.DataFrame, y: pd.Series, splits) -> MethodResult:
    Xv = X.values
    yv = y.values
    n = len(yv)
    oof = np.full((N_REPEATS, n), np.nan, dtype=float)
    fold_rows: list[dict] = []

    for rep, fold, tr, te in splits:
        model = build_model(name)
        X_tr, y_tr = Xv[tr], yv[tr]
        X_te, y_te = Xv[te], yv[te]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr)

        # predict_proba
        proba = get_pos_proba(model, X_te, pos_label=1)
        oof[rep, te] = proba

        fold_rows.append(
            {
                "method": name,
                "repeat": rep,
                "fold": fold,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "auc": _safe_auc(y_te, proba),
                "ap": _safe_ap(y_te, proba),
                "brier": _safe_brier(y_te, proba),
                "logloss": _safe_logloss(y_te, proba),
            }
        )

    return MethodResult(name=name, fold_rows=fold_rows, oof_by_repeat=oof)


def summarize_method(res: MethodResult, y: np.ndarray) -> dict:
    fold_df = pd.DataFrame(res.fold_rows)
    auc_fold_mean = float(np.nanmean(fold_df["auc"]))
    auc_fold_std = float(np.nanstd(fold_df["auc"]))
    ap_fold_mean = float(np.nanmean(fold_df["ap"]))
    ap_fold_std = float(np.nanstd(fold_df["ap"]))

    # one score per repeat (AUC/AP computed on that repeat's OOF)
    auc_repeats = []
    ap_repeats = []
    for r in range(res.oof_by_repeat.shape[0]):
        pr = res.oof_by_repeat[r]
        if np.isnan(pr).any():
            # incomplete repeat (shouldn't happen), skip
            continue
        auc_repeats.append(_safe_auc(y, pr))
        ap_repeats.append(_safe_ap(y, pr))
    auc_repeat_mean = float(np.nanmean(auc_repeats))
    auc_repeat_std = float(np.nanstd(auc_repeats))
    ap_repeat_mean = float(np.nanmean(ap_repeats))
    ap_repeat_std = float(np.nanstd(ap_repeats))

    # per-sample mean OOF across repeats (stabilized)
    oof_mean = mean_oof_scores(res.oof_by_repeat, n_samples=len(y))
    auc_oof = _safe_auc(y, oof_mean)
    ap_oof = _safe_ap(y, oof_mean)
    brier_oof = _safe_brier(y, oof_mean)
    logloss_oof = _safe_logloss(y, oof_mean)

    return {
        "method": res.name,
        "n_samples": int(len(y)),
        "auc_fold_mean": auc_fold_mean,
        "auc_fold_std": auc_fold_std,
        "ap_fold_mean": ap_fold_mean,
        "ap_fold_std": ap_fold_std,
        "auc_repeat_mean": auc_repeat_mean,
        "auc_repeat_std": auc_repeat_std,
        "ap_repeat_mean": ap_repeat_mean,
        "ap_repeat_std": ap_repeat_std,
        "auc_oof": auc_oof,
        "ap_oof": ap_oof,
        "brier_oof": brier_oof,
        "logloss_oof": logloss_oof,
    }


def paired_tests(long_df: pd.DataFrame, ref: str, metric: str) -> pd.DataFrame:
    """Paired tests across the SAME splits (repeat+fold)."""
    piv = long_df.pivot_table(index=["repeat", "fold"], columns="method", values=metric)
    if ref not in piv.columns:
        raise KeyError(f"Reference method '{ref}' not found in results")
    ref_vals = piv[ref]
    out = []
    for m in piv.columns:
        if m == ref:
            continue
        a = ref_vals.values
        b = piv[m].values
        mask = ~np.isnan(a) & ~np.isnan(b)
        a = a[mask]
        b = b[mask]
        if len(a) < 5:
            out.append(
                {
                    "metric": metric,
                    "ref": ref,
                    "method": m,
                    "n_pairs": int(len(a)),
                    "mean_diff(ref-method)": float(np.nanmean(a - b)) if len(a) else np.nan,
                    "ttest_p": np.nan,
                    "wilcoxon_p": np.nan,
                }
            )
            continue

        # Paired t-test
        try:
            t_p = float(ttest_rel(a, b, nan_policy="omit").pvalue)
        except Exception:
            t_p = float("nan")

        # Wilcoxon signed-rank (robust)
        try:
            w_p = float(wilcoxon(a, b, zero_method="wilcox", correction=False).pvalue)
        except Exception:
            w_p = float("nan")

        out.append(
            {
                "metric": metric,
                "ref": ref,
                "method": m,
                "n_pairs": int(len(a)),
                "mean_diff(ref-method)": float(np.nanmean(a - b)),
                "ttest_p": t_p,
                "wilcoxon_p": w_p,
            }
        )
    return pd.DataFrame(out).sort_values(["metric", "wilcoxon_p"], ascending=[True, True])


def plot_roc(y: np.ndarray, results: list[MethodResult], out_png: str, summary_df: pd.DataFrame):
    plt.figure(figsize=(7, 6))
    for res in results:
        oof_mean = mean_oof_scores(res.oof_by_repeat, n_samples=len(y))
        try:
            fpr, tpr, _ = roc_curve(y, oof_mean)
        except Exception:
            continue
        row = summary_df.loc[summary_df["method"] == res.name].iloc[0]
        plt.plot(
            fpr,
            tpr,
            label=(f"{res.name} (AUC_oof={row['auc_oof']:.3f}; "f"AUC_repeat={row['auc_repeat_mean']:.3f}±{row['auc_repeat_std']:.3f})"),
        )
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC (using per-sample mean OOF across repeats)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_pr(y: np.ndarray, results: list[MethodResult], out_png: str, summary_df: pd.DataFrame):
    plt.figure(figsize=(7, 6))
    for res in results:
        oof_mean = mean_oof_scores(res.oof_by_repeat, n_samples=len(y))
        try:
            p, r, _ = precision_recall_curve(y, oof_mean)
        except Exception:
            continue
        row = summary_df.loc[summary_df["method"] == res.name].iloc[0]
        plt.plot(
            r,
            p,
            label=(f"{res.name} (AP_oof={row['ap_oof']:.3f}; "f"AP_repeat={row['ap_repeat_mean']:.3f}±{row['ap_repeat_std']:.3f})"),
        )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR (using per-sample mean OOF across repeats)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_calibration(y: np.ndarray, results: list[MethodResult], out_png: str, summary_df: pd.DataFrame):
    plt.figure(figsize=(7, 6))
    for res in results:
        oof_mean = mean_oof_scores(res.oof_by_repeat, n_samples=len(y))
        # avoid degenerate bins
        try:
            frac_pos, mean_pred = calibration_curve(y, oof_mean, n_bins=10, strategy="quantile")
        except Exception:
            continue
        row = summary_df.loc[summary_df["method"] == res.name].iloc[0]
        plt.plot(
            mean_pred,
            frac_pos,
            marker="o",
            label=(f"{res.name} (Brier={row['brier_oof']:.3f}; "f"AUC_oof={row['auc_oof']:.3f}; "f"AUC_repeat={row['auc_repeat_mean']:.3f}±{row['auc_repeat_std']:.3f})"),
        )
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration (using per-sample mean OOF across repeats)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    df = load_patient_table(DATA_XLSX)
    X, y, pid = select_xy(df, FEATURE_SET)

    print(f"[INFO] Using features ({len(FEATURE_SET)}): {FEATURE_SET}")
    print(f"[INFO] y counts: {pd.Series(y).value_counts().to_dict()}")

    splits = make_splits(y.values, N_SPLITS, N_REPEATS, BASE_RANDOM_STATE)
    print(f"[INFO] Built shared splits: repeats={N_REPEATS}, folds={N_SPLITS}, total={len(splits)}")

    methods = [
        "M14_LogReg_L2",
        "LogReg_L1",
        "LinearSVM",
        "RandomForest",
        "GradBoost",
        # "HistGB",
        "XGBoost",
    ]

    all_results: list[MethodResult] = []
    all_long_rows: list[dict] = []
    all_summary_rows: list[dict] = []

    for m in methods:
        print("=" * 80)
        print(f"[RUN] {m}")
        res = eval_one_method(m, X, y, splits)
        all_results.append(res)
        all_long_rows.extend(res.fold_rows)
        all_summary_rows.append(summarize_method(res, y.values))

    long_df = pd.DataFrame(all_long_rows)
    summary_df = pd.DataFrame(all_summary_rows).sort_values("auc_oof", ascending=False)

    # Save
    long_path = os.path.join(OUTDIR, "results_long.csv")
    summary_path = os.path.join(OUTDIR, "summary.csv")
    long_df.to_csv(long_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"[Saved] {long_path}")
    print(f"[Saved] {summary_path}")

    # Paired tests vs reference
    pt_auc = paired_tests(long_df, REFERENCE_METHOD, metric="auc")
    pt_ap = paired_tests(long_df, REFERENCE_METHOD, metric="ap")
    pt = pd.concat([pt_auc, pt_ap], axis=0, ignore_index=True)
    pt_path = os.path.join(OUTDIR, "paired_tests.csv")
    pt.to_csv(pt_path, index=False)
    print(f"[Saved] {pt_path}")

    # Plots
    plot_roc(y.values, all_results, os.path.join(OUTDIR, "oof_roc_mean.png"), summary_df)
    plot_pr(y.values, all_results, os.path.join(OUTDIR, "oof_pr_mean.png"), summary_df)
    plot_calibration(y.values, all_results, os.path.join(OUTDIR, "calibration_mean.png"), summary_df)
    print(f"[Saved] plots in {OUTDIR}")

    # Also save the per-sample mean OOF predictions (useful for downstream calibration/thresholding)
    oof_pred = {"patient_id": pid.values, "y": y.values}
    for res in all_results:
        oof_pred[res.name] = mean_oof_scores(res.oof_by_repeat, n_samples=len(y))
    oof_df = pd.DataFrame(oof_pred)
    oof_path = os.path.join(OUTDIR, "oof_pred_mean.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"[Saved] {oof_path}")

    print("DONE.")


if __name__ == "__main__":
    main()