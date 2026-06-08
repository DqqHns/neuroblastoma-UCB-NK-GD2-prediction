


import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

# ===== paths =====
OOF_CSV  = r"G:/Dqq_code/nk_model(1.8)/compare_methods_results_new_COMPLETE_CASE = True/oof_pred_mean.csv"  # <-- 如果你实际路径不同改这里
DATA_XLSX = r"G:/Dqq_code/nk_model(1.8)/data/DATA.xlsx"
FEAT_SHEET = "patient_features"
OUTDIR = r"G:/Dqq_code/nk_model(1.8)/shap_results—M14"
os.makedirs(OUTDIR, exist_ok=True)

# ===== choose which model column in oof_pred_mean.csv =====
SCORE_COL = "M14_LogReg_L2"

# ===== 6 features =====
FEATURES = [
    "dyn_slope_BA_on_PCT",
    "HGB_mean",
    "RDW-SD_baseline",
    "dyn_same_dir_rate",
    "Na+_mean",
    "SAA_mean",
]

Y_COL_CANDIDATES = ["y_true", "y", "label", "target"] 

SPECIAL_COLS = {"Na+_mean"}

def _normalize_pid(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)  
    return s

def load_oof_csv() -> pd.DataFrame:
    df = pd.read_csv(OOF_CSV)
    cols_lower = {c.lower(): c for c in df.columns}

    # patient_id
    pid_col = None
    for cand in ["patient_id", "patientid", "pid", "id"]:
        if cand in cols_lower:
            pid_col = cols_lower[cand]
            break
    if pid_col is None:
        raise ValueError(f"[OOF] Cannot find patient_id column. Columns={list(df.columns)}")

    # y_true
    y_col = None
    for cand in Y_COL_CANDIDATES:
        if cand in cols_lower:
            y_col = cols_lower[cand]
            break
    if y_col is None:
        raise ValueError(f"[OOF] Cannot find y column. Columns={list(df.columns)}")

    if SCORE_COL not in df.columns:
        raise ValueError(f"[OOF] Missing SCORE_COL='{SCORE_COL}'. Available={list(df.columns)}")

    out = df[[pid_col, y_col, SCORE_COL]].copy()
    out.columns = ["patient_id", "y_true", "y_score_oof_mean"]
    out["patient_id"] = _normalize_pid(out["patient_id"])
    out["y_true"] = out["y_true"].astype(int)
    out["y_score_oof_mean"] = pd.to_numeric(out["y_score_oof_mean"], errors="coerce")
    out = out.dropna(subset=["y_score_oof_mean"])
    return out

def load_feature_table() -> pd.DataFrame:
    df = pd.read_excel(DATA_XLSX, sheet_name=FEAT_SHEET)
    cols_lower = {c.lower(): c for c in df.columns}

    pid_col = None
    for cand in ["patient_id", "patientid", "pid", "id"]:
        if cand in cols_lower:
            pid_col = cols_lower[cand]
            break
    if pid_col is None:
        raise ValueError(f"[FEATURE] Cannot find patient_id column in sheet '{FEAT_SHEET}'.")

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"[FEATURE] Missing features in sheet '{FEAT_SHEET}': {missing}")

    out = df[[pid_col] + FEATURES].copy()
    out.columns = ["patient_id"] + FEATURES
    out["patient_id"] = _normalize_pid(out["patient_id"])
    for f in FEATURES:
        out[f] = pd.to_numeric(out[f], errors="coerce")
    return out

def fit_m14_logreg_l2(X: pd.DataFrame, y: np.ndarray) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            penalty="l2",
            C=1.0,  
        ))
    ])
    pipe.fit(X, y)
    return pipe

def pick_examples(df: pd.DataFrame, top_k: int = 2):
    """
    based on y_score_oof_mean:
      - high_conf_pos: y=1, score high
      - high_conf_neg: y=0, score low
      - fp: y=0, score high
      - fn: y=1, score low
    """
    d = df.sort_values("y_score_oof_mean", ascending=False).copy()
    high_conf_pos = d[d["y_true"] == 1].head(top_k)
    fp = d[d["y_true"] == 0].head(top_k)
    fn = d[d["y_true"] == 1].tail(top_k)
    high_conf_neg = d[d["y_true"] == 0].tail(top_k)
    return {
        "TP_high": high_conf_pos,
        "TN_high": high_conf_neg,
        "FP": fp,
        "FN": fn,
    }

def main():
    print("[1/5] Load OOF mean predictions...")
    oof = load_oof_csv()
    print("  oof rows:", len(oof))

    print("[2/5] Load features...")
    feat = load_feature_table()
    print("  feat rows:", len(feat))

    print("[3/5] Merge by patient_id...")
    df = pd.merge(oof, feat, on="patient_id", how="inner")
    df = df.dropna(subset=FEATURES + ["y_true", "y_score_oof_mean"]).reset_index(drop=True)
    print("  merged rows:", len(df))

    merged_csv = os.path.join(OUTDIR, "merged_for_shap.csv")
    df.to_csv(merged_csv, index=False, encoding="utf-8-sig")
    print("[Saved]", merged_csv)

    X = df[FEATURES].copy()
    y = df["y_true"].values.astype(int)

    print("[4/5] Fit M14 (LogReg L2) on full merged data for SHAP explanation...")
    model = fit_m14_logreg_l2(X, y)

    # sanity numbers (printed only, not on plots)
    prob_fit = model.predict_proba(X)[:, 1]
    auc_fit = roc_auc_score(y, prob_fit)
    ap_fit = average_precision_score(y, prob_fit)
    auc_oof = roc_auc_score(y, df["y_score_oof_mean"].values)
    ap_oof = average_precision_score(y, df["y_score_oof_mean"].values)
    print(f"[Sanity] AUC_fit(full)= {auc_fit:.3f}, AP_fit(full)= {ap_fit:.3f}")
    print(f"[Sanity] AUC_oofmean  = {auc_oof:.3f}, AP_oofmean  = {ap_oof:.3f}")

    print("[5/5] Compute SHAP & plot...")
    import shap

    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]

    X_scaled = scaler.transform(X)
    Xs = pd.DataFrame(X_scaled, columns=FEATURES)

    explainer = shap.LinearExplainer(clf, Xs, feature_perturbation="interventional")
    shap_values = explainer.shap_values(Xs)

    # 1) beeswarm
    plt.figure()
    shap.summary_plot(shap_values, Xs, show=False)
    plt.tight_layout()
    out1 = os.path.join(OUTDIR, "shap_summary_beeswarm.png")
    plt.savefig(out1, dpi=200)
    plt.close()
    print("[Saved]", out1)

    # 2) bar
    plt.figure()
    shap.summary_plot(shap_values, Xs, plot_type="bar", show=False)
    plt.tight_layout()
    out2 = os.path.join(OUTDIR, "shap_importance_bar.png")
    plt.savefig(out2, dpi=200)
    plt.close()
    print("[Saved]", out2)

    # 3) dependence plots (no extra titles)
    for f in FEATURES:
        plt.figure()
        shap.dependence_plot(f, shap_values, Xs, show=False)
        plt.tight_layout()
        outp = os.path.join(OUTDIR, f"shap_dependence_{f}.png")
        plt.savefig(outp, dpi=200)
        plt.close()
        print("[Saved]", outp)

    # 4) waterfall examples (pick by OOF mean score)
    ex_sets = pick_examples(df, top_k=2)
    base_value = explainer.expected_value

    for tag, sub in ex_sets.items():
        for _, r in sub.iterrows():
            pid = r["patient_id"]
            idx = df.index[df["patient_id"] == pid][0]
            exp = shap.Explanation(
                values=shap_values[idx],
                base_values=base_value,
                data=Xs.iloc[idx].values,
                feature_names=FEATURES
            )
            plt.figure()
            shap.plots.waterfall(exp, show=False, max_display=12)
            plt.tight_layout()
            outw = os.path.join(OUTDIR, f"shap_waterfall_{tag}_pid{pid}.png")
            plt.savefig(outw, dpi=200)
            plt.close()
            print("[Saved]", outw)

    print("DONE. Figures in:", OUTDIR)

if __name__ == "__main__":
    main()
