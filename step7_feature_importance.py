import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance



XLSX_PATH = r"data/DATA.xlsx"      
SHEET_NAME = "patient_features_all"                
ID_COL = "patient_id"
LABEL_COL = "efficacy"

OUTDIR = "feature_importance_results"
os.makedirs(OUTDIR, exist_ok=True)


N_SPLITS = 3
N_REPEATS = 50
BASE_RANDOM_STATE = 42


PERM_REPEATS = 50


MAX_MISSING_RATE = 0.60

# ========== 工具函数 ==========
def auto_pick_sheet(xlsx_path: str) -> str:
    xls = pd.ExcelFile(xlsx_path)
    if "patient_features" in xls.sheet_names:
        return "patient_features"
    return xls.sheet_names[0]

def get_feature_columns(df: pd.DataFrame) -> list:

    cols = []
    for c in df.columns:
        if c in [ID_COL, LABEL_COL]:
            continue

        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        cols.append(c)
    return cols

def main():
    sheet = SHEET_NAME or auto_pick_sheet(XLSX_PATH)
    df = pd.read_excel(XLSX_PATH, sheet_name=sheet)
    print(f"[OK] Using sheet: {sheet}, shape={df.shape}")

    # 基础清洗
    if ID_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(f"Missing {ID_COL} or {LABEL_COL} in sheet {sheet}")

    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    df = df.dropna(subset=[LABEL_COL])
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    # 取特征列
    feat_cols = get_feature_columns(df)

    # 缺失率过滤（重要！）
    miss = df[feat_cols].apply(lambda s: pd.to_numeric(s, errors="coerce").isna().mean())
    keep = miss[miss <= MAX_MISSING_RATE].index.tolist()
    drop = miss[miss > MAX_MISSING_RATE].sort_values(ascending=False)

    print(f"[INFO] candidate features={len(feat_cols)}, kept={len(keep)}, dropped_missing={len(drop)}")
    if len(drop) > 0:
        print("[INFO] dropped (missing too high) top10:")
        print(drop.head(10))

    X = df[keep].apply(pd.to_numeric, errors="coerce")
    y = df[LABEL_COL].values


    clf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=5000,
            solver="liblinear",
            class_weight="balanced",
            random_state=BASE_RANDOM_STATE
        ))
    ])

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=BASE_RANDOM_STATE
    )


    fold_aucs = []
    all_importances = []  
    all_coefs = []        

    for fold_i, (tr, te) in enumerate(cv.split(X, y), start=1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]

        clf.fit(Xtr, ytr)

        prob = clf.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, prob)
        fold_aucs.append(auc)

        pi = permutation_importance(
            clf, Xte, yte,
            scoring="roc_auc",
            n_repeats=PERM_REPEATS,
            random_state=BASE_RANDOM_STATE + fold_i,
            n_jobs=1
        )
        imp = pd.Series(pi.importances_mean, index=keep)
        all_importances.append(imp)


        coef = pd.Series(clf.named_steps["lr"].coef_.ravel(), index=keep)
        all_coefs.append(coef)

        if fold_i % 30 == 0:
            print(f"[CV] fold {fold_i}: AUC={auc:.3f}")

    # 汇总 AUC
    auc_mean = float(np.mean(fold_aucs))
    auc_std = float(np.std(fold_aucs))
    print(f"[DONE] CV AUC mean={auc_mean:.4f}, std={auc_std:.4f}")

    # 汇总重要性
    imp_df = pd.concat(all_importances, axis=1).T   # shape: (folds, features)
    imp_summary = pd.DataFrame({
        "perm_imp_mean": imp_df.mean(axis=0),
        "perm_imp_std": imp_df.std(axis=0),
        "perm_imp_p25": imp_df.quantile(0.25, axis=0),
        "perm_imp_p50": imp_df.quantile(0.50, axis=0),
        "perm_imp_p75": imp_df.quantile(0.75, axis=0),
        "missing_rate": miss.loc[keep].values
    }, index=keep).sort_values("perm_imp_mean", ascending=False)

    out_csv = os.path.join(OUTDIR, "permutation_importance_summary.csv")
    imp_summary.to_csv(out_csv, encoding="utf-8-sig")
    print(f"[Saved] {out_csv}")

    # 系数稳定性汇总
    coef_df = pd.concat(all_coefs, axis=1).T
    coef_summary = pd.DataFrame({
        "coef_mean": coef_df.mean(axis=0),
        "coef_std": coef_df.std(axis=0),
        "coef_p25": coef_df.quantile(0.25, axis=0),
        "coef_p50": coef_df.quantile(0.50, axis=0),
        "coef_p75": coef_df.quantile(0.75, axis=0),
    }, index=keep)

    out_coef = os.path.join(OUTDIR, "coef_stability_summary.csv")
    coef_summary.to_csv(out_coef, encoding="utf-8-sig")
    print(f"[Saved] {out_coef}")

    # 画 top20 重要性条形图
    topk = 20
    top = imp_summary.head(topk).iloc[::-1]  
    plt.figure(figsize=(10, 8))
    plt.barh(top.index, top["perm_imp_mean"])
    plt.xlabel("Permutation importance (mean AUC drop)")
    plt.title(f"Top {topk} Feature Importances (Permutation)")
    plt.tight_layout()
    fig1 = os.path.join(OUTDIR, "perm_importance_top20.png")
    plt.savefig(fig1, dpi=200)
    plt.close()
    print(f"[Saved] {fig1}")

if __name__ == "__main__":
    main()
