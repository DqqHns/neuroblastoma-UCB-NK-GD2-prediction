

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import numpy as np
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


def _get_classes(est):
    cls = getattr(est, "classes_", None)
    if cls is not None:
        return list(cls)
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
    if hasattr(est, "predict_proba"):
        p = est.predict_proba(X)
        if getattr(p, "ndim", 1) == 1:
            return p
        classes = _get_classes(est)
        j = classes.index(pos_label) if (classes is not None and pos_label in classes) else -1
        return p[:, j]

    if hasattr(est, "decision_function"):
        s = est.decision_function(X)
        if getattr(s, "ndim", 1) > 1:
            classes = _get_classes(est)
            j = classes.index(pos_label) if (classes is not None and pos_label in classes) else -1
            s = s[:, j]
        return 1.0 / (1.0 + np.exp(-s))

    raise AttributeError("Estimator has neither predict_proba nor decision_function")


try:
    from xgboost import XGBClassifier  # type: ignore
    _HAS_XGB = True
except Exception:
    XGBClassifier = None  # type: ignore
    _HAS_XGB = False


DATA_XLSX = os.path.join("data", "DATA.xlsx")
OUTDIR = "compare_methods_results"

ID_COL = "patient_id"
LABEL_COL = "efficacy"
FEATURE_SET = [
    "dyn_slope_BA_on_PCT",
    "HGB_mean",
    "RDW-SD_baseline",
    "dyn_same_dir_rate",
    "Na+_mean",
    "SAA_mean",
]

N_SPLITS = 3
N_REPEATS = 50
BASE_RANDOM_STATE = 42
COMPLETE_CASE = True
REFERENCE_METHOD = "M14_LogReg_L2"

# CI config
CI_ALPHA = 0.95
BOOT_N = 2000
BOOT_SEED = 20260313
SAVE_BOOTSTRAP_DISTRIBUTIONS = True


def _auto_pick_sheet(xlsx_path: str, must_have: list[str]) -> str:
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
        raise FileNotFoundError(f"Cannot find {xlsx_path}. Edit DATA_XLSX if needed.")
    sheet = _auto_pick_sheet(xlsx_path, must_have=[ID_COL, LABEL_COL])
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    if ID_COL in df.columns:
        df[ID_COL] = df[ID_COL].astype(str).str.strip()
    return df


def select_xy(df: pd.DataFrame, features: list[str]):
    missing = [c for c in [ID_COL, LABEL_COL] + features if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in table: {missing}")
    use = df[[ID_COL, LABEL_COL] + features].copy().dropna(subset=[LABEL_COL])
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
        print(f"[INFO] COMPLETE_CASE=False: using {len(y)} patients")
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
    oof_by_repeat: np.ndarray


def _safe_auc(y_true, y_score):
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _safe_ap(y_true, y_score):
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


def _safe_brier(y_true, y_score):
    try:
        return float(brier_score_loss(y_true, y_score))
    except Exception:
        return float("nan")


def _safe_logloss(y_true, y_score):
    try:
        return float(log_loss(y_true, y_score, labels=[0, 1]))
    except Exception:
        return float("nan")


def _ci_from_values(values, alpha=0.95):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    q = (1.0 - alpha) / 2.0
    return float(np.quantile(arr, q)), float(np.quantile(arr, 1.0 - q))


def bootstrap_metric_ci(y_true, y_score, metric_func, n_boot=2000, seed=20260313, alpha=0.95, stratified=True):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(seed)

    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]
    vals = []

    for _ in range(n_boot):
        if stratified:
            samp_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
            samp_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
            samp = np.concatenate([samp_pos, samp_neg])
            rng.shuffle(samp)
        else:
            samp = rng.choice(np.arange(len(y_true)), size=len(y_true), replace=True)

        yt = y_true[samp]
        ys = y_score[samp]
        try:
            vals.append(float(metric_func(yt, ys)))
        except Exception:
            continue

    lo, hi = _ci_from_values(vals, alpha=alpha)
    return {
        "mean": float(np.nanmean(vals)) if len(vals) else np.nan,
        "std": float(np.nanstd(vals)) if len(vals) else np.nan,
        "ci_low": lo,
        "ci_high": hi,
        "n_boot_ok": int(len(vals)),
        "values": vals,
    }


def build_model(method: str) -> Pipeline:
    imputer = "passthrough" if COMPLETE_CASE else SimpleImputer(strategy="median")

    def scaled(clf):
        return Pipeline([("imputer", imputer), ("scaler", StandardScaler()), ("clf", clf)])

    def unscaled(clf):
        return Pipeline([("imputer", imputer), ("clf", clf)])

    if method == "M14_LogReg_L2":
        clf = LogisticRegression(penalty="l2", C=1.0, solver="liblinear", max_iter=5000, random_state=BASE_RANDOM_STATE)
        return scaled(clf)
    if method == "LogReg_L1":
        clf = LogisticRegression(penalty="l1", C=1.0, solver="liblinear", max_iter=5000, random_state=BASE_RANDOM_STATE)
        return scaled(clf)
    if method == "LinearSVM":
        base = LinearSVC(C=1.0, random_state=BASE_RANDOM_STATE)
        clf = CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)
        return scaled(clf)
    if method == "RandomForest":
        clf = RandomForestClassifier(n_estimators=600, max_depth=None, min_samples_leaf=1, class_weight="balanced", random_state=BASE_RANDOM_STATE)
        return unscaled(clf)
    if method == "GradBoost":
        return unscaled(GradientBoostingClassifier(random_state=BASE_RANDOM_STATE))
    if method == "HistGB":
        return unscaled(HistGradientBoostingClassifier(random_state=BASE_RANDOM_STATE))
    if method == "XGBoost":
        if not _HAS_XGB:
            raise RuntimeError("XGBoost is not installed.")
        clf = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, min_child_weight=1.0,
            gamma=0.0, objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", random_state=BASE_RANDOM_STATE,
        )
        return unscaled(clf)
    raise ValueError(f"Unknown method: {method}")


def eval_one_method(name: str, X: pd.DataFrame, y: pd.Series, splits) -> MethodResult:
    Xv = X.values
    yv = y.values
    n = len(yv)
    oof = np.full((N_REPEATS, n), np.nan, dtype=float)
    fold_rows = []

    for rep, fold, tr, te in splits:
        model = build_model(name)
        X_tr, y_tr = Xv[tr], yv[tr]
        X_te, y_te = Xv[te], yv[te]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr)

        proba = get_pos_proba(model, X_te, pos_label=1)
        oof[rep, te] = proba

        fold_rows.append({
            "method": name,
            "repeat": rep,
            "fold": fold,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "auc": _safe_auc(y_te, proba),
            "ap": _safe_ap(y_te, proba),
            "brier": _safe_brier(y_te, proba),
            "logloss": _safe_logloss(y_te, proba),
        })

    return MethodResult(name=name, fold_rows=fold_rows, oof_by_repeat=oof)


def summarize_method(res: MethodResult, y: np.ndarray, n_boot=2000, boot_seed=20260313, alpha=0.95):
    fold_df = pd.DataFrame(res.fold_rows)

    auc_fold_mean = float(np.nanmean(fold_df["auc"]))
    auc_fold_std = float(np.nanstd(fold_df["auc"]))
    ap_fold_mean = float(np.nanmean(fold_df["ap"]))
    ap_fold_std = float(np.nanstd(fold_df["ap"]))
    brier_fold_mean = float(np.nanmean(fold_df["brier"]))
    brier_fold_std = float(np.nanstd(fold_df["brier"]))
    logloss_fold_mean = float(np.nanmean(fold_df["logloss"]))
    logloss_fold_std = float(np.nanstd(fold_df["logloss"]))

    auc_repeats, ap_repeats, brier_repeats, logloss_repeats = [], [], [], []
    for r in range(res.oof_by_repeat.shape[0]):
        pr = res.oof_by_repeat[r]
        if np.isnan(pr).any():
            continue
        auc_repeats.append(_safe_auc(y, pr))
        ap_repeats.append(_safe_ap(y, pr))
        brier_repeats.append(_safe_brier(y, pr))
        logloss_repeats.append(_safe_logloss(y, pr))

    auc_repeat_ci_low, auc_repeat_ci_high = _ci_from_values(auc_repeats, alpha)
    ap_repeat_ci_low, ap_repeat_ci_high = _ci_from_values(ap_repeats, alpha)
    brier_repeat_ci_low, brier_repeat_ci_high = _ci_from_values(brier_repeats, alpha)
    logloss_repeat_ci_low, logloss_repeat_ci_high = _ci_from_values(logloss_repeats, alpha)

    oof_mean = np.nanmean(res.oof_by_repeat, axis=0)
    auc_oof = _safe_auc(y, oof_mean)
    ap_oof = _safe_ap(y, oof_mean)
    brier_oof = _safe_brier(y, oof_mean)
    logloss_oof = _safe_logloss(y, oof_mean)

    boot_auc = bootstrap_metric_ci(y, oof_mean, _safe_auc, n_boot=n_boot, seed=boot_seed + 11, alpha=alpha)
    boot_ap = bootstrap_metric_ci(y, oof_mean, _safe_ap, n_boot=n_boot, seed=boot_seed + 23, alpha=alpha)
    boot_brier = bootstrap_metric_ci(y, oof_mean, _safe_brier, n_boot=n_boot, seed=boot_seed + 37, alpha=alpha)
    boot_logloss = bootstrap_metric_ci(y, oof_mean, _safe_logloss, n_boot=n_boot, seed=boot_seed + 51, alpha=alpha)

    row = {
        "method": res.name,
        "n_samples": int(len(y)),
        "auc_fold_mean": auc_fold_mean,
        "auc_fold_std": auc_fold_std,
        "ap_fold_mean": ap_fold_mean,
        "ap_fold_std": ap_fold_std,
        "brier_fold_mean": brier_fold_mean,
        "brier_fold_std": brier_fold_std,
        "logloss_fold_mean": logloss_fold_mean,
        "logloss_fold_std": logloss_fold_std,
        "auc_repeat_mean": float(np.nanmean(auc_repeats)),
        "auc_repeat_std": float(np.nanstd(auc_repeats)),
        "auc_repeat_ci_low": auc_repeat_ci_low,
        "auc_repeat_ci_high": auc_repeat_ci_high,
        "ap_repeat_mean": float(np.nanmean(ap_repeats)),
        "ap_repeat_std": float(np.nanstd(ap_repeats)),
        "ap_repeat_ci_low": ap_repeat_ci_low,
        "ap_repeat_ci_high": ap_repeat_ci_high,
        "brier_repeat_mean": float(np.nanmean(brier_repeats)),
        "brier_repeat_std": float(np.nanstd(brier_repeats)),
        "brier_repeat_ci_low": brier_repeat_ci_low,
        "brier_repeat_ci_high": brier_repeat_ci_high,
        "logloss_repeat_mean": float(np.nanmean(logloss_repeats)),
        "logloss_repeat_std": float(np.nanstd(logloss_repeats)),
        "logloss_repeat_ci_low": logloss_repeat_ci_low,
        "logloss_repeat_ci_high": logloss_repeat_ci_high,
        "auc_oof": auc_oof,
        "auc_oof_ci_low": boot_auc["ci_low"],
        "auc_oof_ci_high": boot_auc["ci_high"],
        "ap_oof": ap_oof,
        "ap_oof_ci_low": boot_ap["ci_low"],
        "ap_oof_ci_high": boot_ap["ci_high"],
        "brier_oof": brier_oof,
        "brier_oof_ci_low": boot_brier["ci_low"],
        "brier_oof_ci_high": boot_brier["ci_high"],
        "logloss_oof": logloss_oof,
        "logloss_oof_ci_low": boot_logloss["ci_low"],
        "logloss_oof_ci_high": boot_logloss["ci_high"],
        "n_boot_ok_auc": boot_auc["n_boot_ok"],
        "n_boot_ok_ap": boot_ap["n_boot_ok"],
        "n_boot_ok_brier": boot_brier["n_boot_ok"],
        "n_boot_ok_logloss": boot_logloss["n_boot_ok"],
    }

    boot_df = pd.DataFrame({
        "method": [res.name] * len(boot_auc["values"]),
        "boot_id": np.arange(len(boot_auc["values"])),
        "auc_oof_boot": boot_auc["values"],
        "ap_oof_boot": boot_ap["values"][:len(boot_auc["values"])],
        "brier_oof_boot": boot_brier["values"][:len(boot_auc["values"])],
        "logloss_oof_boot": boot_logloss["values"][:len(boot_auc["values"])],
    })
    return row, boot_df


def paired_tests(long_df: pd.DataFrame, ref: str, metric: str) -> pd.DataFrame:
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
            out.append({"metric": metric, "ref": ref, "method": m, "n_pairs": int(len(a)), "mean_diff(ref-method)": float(np.nanmean(a - b)) if len(a) else np.nan, "ttest_p": np.nan, "wilcoxon_p": np.nan})
            continue
        try:
            t_p = float(ttest_rel(a, b, nan_policy="omit").pvalue)
        except Exception:
            t_p = float("nan")
        try:
            w_p = float(wilcoxon(a, b, zero_method="wilcox", correction=False).pvalue)
        except Exception:
            w_p = float("nan")
        out.append({"metric": metric, "ref": ref, "method": m, "n_pairs": int(len(a)), "mean_diff(ref-method)": float(np.nanmean(a - b)), "ttest_p": t_p, "wilcoxon_p": w_p})
    return pd.DataFrame(out).sort_values(["metric", "wilcoxon_p"], ascending=[True, True])


def plot_roc(y: np.ndarray, results: list[MethodResult], out_png: str, summary_df: pd.DataFrame):
    plt.figure(figsize=(7, 6))
    for res in results:
        oof_mean = np.nanmean(res.oof_by_repeat, axis=0)
        try:
            fpr, tpr, _ = roc_curve(y, oof_mean)
        except Exception:
            continue
        row = summary_df.loc[summary_df["method"] == res.name].iloc[0]
        plt.plot(fpr, tpr, label=f"{res.name} (AUC_oof={row['auc_oof']:.3f} [{row['auc_oof_ci_low']:.3f}, {row['auc_oof_ci_high']:.3f}])")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("OOF ROC (per-sample mean across repeats)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_pr(y: np.ndarray, results: list[MethodResult], out_png: str, summary_df: pd.DataFrame):
    plt.figure(figsize=(7, 6))
    for res in results:
        oof_mean = np.nanmean(res.oof_by_repeat, axis=0)
        try:
            p, r, _ = precision_recall_curve(y, oof_mean)
        except Exception:
            continue
        row = summary_df.loc[summary_df["method"] == res.name].iloc[0]
        plt.plot(r, p, label=f"{res.name} (AP_oof={row['ap_oof']:.3f} [{row['ap_oof_ci_low']:.3f}, {row['ap_oof_ci_high']:.3f}])")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("OOF PR (per-sample mean across repeats)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_calibration(y: np.ndarray, results: list[MethodResult], out_png: str, summary_df: pd.DataFrame):
    plt.figure(figsize=(7, 6))
    for res in results:
        oof_mean = np.nanmean(res.oof_by_repeat, axis=0)
        try:
            frac_pos, mean_pred = calibration_curve(y, oof_mean, n_bins=10, strategy="quantile")
        except Exception:
            continue
        row = summary_df.loc[summary_df["method"] == res.name].iloc[0]
        plt.plot(mean_pred, frac_pos, marker="o", label=f"{res.name} (Brier={row['brier_oof']:.3f} [{row['brier_oof_ci_low']:.3f}, {row['brier_oof_ci_high']:.3f}])")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration (per-sample mean OOF)")
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
    print(f"[INFO] Bootstrap CI: n_boot={BOOT_N}, alpha={CI_ALPHA}, seed={BOOT_SEED}")

    splits = make_splits(y.values, N_SPLITS, N_REPEATS, BASE_RANDOM_STATE)
    print(f"[INFO] Built shared splits: repeats={N_REPEATS}, folds={N_SPLITS}, total={len(splits)}")

    methods = ["M14_LogReg_L2", "LogReg_L1", "LinearSVM", "RandomForest", "GradBoost", "XGBoost"]

    all_results = []
    all_long_rows = []
    all_summary_rows = []
    boot_tables = []

    for m in methods:
        print("=" * 80)
        print(f"[RUN] {m}")
        res = eval_one_method(m, X, y, splits)
        all_results.append(res)
        all_long_rows.extend(res.fold_rows)
        row, boot_df = summarize_method(res, y.values, n_boot=BOOT_N, boot_seed=BOOT_SEED, alpha=CI_ALPHA)
        all_summary_rows.append(row)
        if SAVE_BOOTSTRAP_DISTRIBUTIONS:
            boot_tables.append(boot_df)

    long_df = pd.DataFrame(all_long_rows)
    summary_df = pd.DataFrame(all_summary_rows).sort_values("auc_oof", ascending=False)

    long_path = os.path.join(OUTDIR, "results_long.csv")
    summary_path = os.path.join(OUTDIR, "summary_with_ci.csv")
    long_df.to_csv(long_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"[Saved] {long_path}")
    print(f"[Saved] {summary_path}")

    if SAVE_BOOTSTRAP_DISTRIBUTIONS and boot_tables:
        boot_all = pd.concat(boot_tables, axis=0, ignore_index=True)
        boot_path = os.path.join(OUTDIR, "oof_metric_bootstrap_distributions.csv")
        boot_all.to_csv(boot_path, index=False)
        print(f"[Saved] {boot_path}")

    pt_auc = paired_tests(long_df, REFERENCE_METHOD, metric="auc")
    pt_ap = paired_tests(long_df, REFERENCE_METHOD, metric="ap")
    pt = pd.concat([pt_auc, pt_ap], axis=0, ignore_index=True)
    pt_path = os.path.join(OUTDIR, "paired_tests.csv")
    pt.to_csv(pt_path, index=False)
    print(f"[Saved] {pt_path}")

    plot_roc(y.values, all_results, os.path.join(OUTDIR, "oof_roc_mean_ci.png"), summary_df)
    plot_pr(y.values, all_results, os.path.join(OUTDIR, "oof_pr_mean_ci.png"), summary_df)
    plot_calibration(y.values, all_results, os.path.join(OUTDIR, "calibration_mean_ci.png"), summary_df)
    print(f"[Saved] plots in {OUTDIR}")

    oof_pred = {"patient_id": pid.values, "y": y.values}
    for res in all_results:
        oof_pred[res.name] = np.nanmean(res.oof_by_repeat, axis=0)
    oof_df = pd.DataFrame(oof_pred)
    oof_path = os.path.join(OUTDIR, "oof_pred_mean.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"[Saved] {oof_path}")
    print("DONE.")


if __name__ == "__main__":
    main()



