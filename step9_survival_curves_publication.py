
# -*- coding: utf-8 -*-
"""
step15_survival_with_model_and_efficacy_fixed.py

修正版：
1) 正确识别 生存资料.xlsx 中包含 OS / EFS / PFS 的中文列名
2) 与 DATA.xlsx 的 patient_features 直接按 patient_id 对齐真实 efficacy
3) 与 step8 的 oof_pred_mean.csv 直接按 patient_id 对齐 M14_LogReg_L2
4) 输出：
   - 按真实疗效分组的 KM 曲线
   - 按模型预测分组的 KM 曲线
   - 连续预测分数的 Cox 回归
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
PROJECT_DIR = r"G:/Dqq_code/nk_model(1.8)"

SURV_XLSX = os.path.join(PROJECT_DIR, "data/生存资料.xlsx")
DATA_XLSX = os.path.join(PROJECT_DIR, "data", "DATA.xlsx")
OOF_CSV   = os.path.join(PROJECT_DIR, "compare_methods_results", "oof_pred_mean.csv")

OUTDIR = os.path.join(PROJECT_DIR, "survival_with_model_outputs_fixed")
os.makedirs(OUTDIR, exist_ok=True)

SHEET_HIGH_RISK = "初诊高危患者"
SHEET_RR = "复发难治患者"

PATIENT_FEATURE_SHEET = "patient_features"
SCORE_COL = "M14_LogReg_L2"

# 预测分组方式：
# "median" / "0.5" / "youden_from_efficacy"
PRED_GROUP_METHOD = "youden_from_efficacy"
FIXED_THRESHOLD = 0.5

DPI = 300
FONT_FAMILY = "Times New Roman"


# =========================
# Helpers
# =========================
def _norm_id(x):
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)
    return s

def _savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close()

def _find_col_contains(cols, keywords_all):
    for c in cols:
        c_str = str(c).strip()
        if all(k in c_str for k in keywords_all):
            return c
    return None

def _find_first_existing(cols, candidates):
    lower_map = {str(c).strip().lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

def _km_plot(df, time_col, event_col, group_col, title, out_png):
    sub = df[[time_col, event_col, group_col]].copy().dropna()
    if len(sub) == 0 or sub[group_col].nunique() < 2:
        print(f"[WARN] skip {title}: no enough data")
        return None

    plt.figure(figsize=(6.4, 5.2))
    kmf = KaplanMeierFitter()

    groups = list(sub[group_col].dropna().unique())
    for g in groups:
        d = sub[sub[group_col] == g]
        kmf.fit(d[time_col], event_observed=d[event_col], label=str(g))
        kmf.plot_survival_function(ci_show=True)

    plt.title(title)
    plt.xlabel("Time (months)")
    plt.ylabel("Survival probability")
    plt.grid(alpha=0.3, linestyle="--")
    _savefig(out_png)

    if len(groups) == 2:
        g1, g2 = groups
        d1 = sub[sub[group_col] == g1]
        d2 = sub[sub[group_col] == g2]
        lr = logrank_test(
            d1[time_col], d2[time_col],
            event_observed_A=d1[event_col],
            event_observed_B=d2[event_col]
        )
        return {
            "group_a": str(g1),
            "group_b": str(g2),
            "n_a": int(len(d1)),
            "n_b": int(len(d2)),
            "p_value": float(lr.p_value),
            "test_statistic": float(lr.test_statistic),
            "time_col": time_col,
            "event_col": event_col,
            "group_col": group_col,
            "title": title,
        }
    return None

def _run_univariable_cox(df, time_col, event_col, score_col, cohort_name, endpoint_name):
    sub = df[[time_col, event_col, score_col]].copy().dropna()
    if len(sub) < 8 or pd.to_numeric(sub[event_col], errors="coerce").sum() < 2:
        return None

    cph = CoxPHFitter()
    cph.fit(sub, duration_col=time_col, event_col=event_col)
    s = cph.summary.reset_index()
    s["cohort"] = cohort_name
    s["endpoint"] = endpoint_name
    return s

def _make_pred_group(df):
    df = df.copy()
    score = pd.to_numeric(df["pred_score"], errors="coerce")

    if PRED_GROUP_METHOD == "0.5":
        thr = FIXED_THRESHOLD
    elif PRED_GROUP_METHOD == "median":
        thr = float(score.dropna().median())
    elif PRED_GROUP_METHOD == "youden_from_efficacy":
        sub = df.dropna(subset=["pred_score", "efficacy"]).copy()
        if len(sub) < 5 or sub["efficacy"].nunique() < 2:
            thr = float(score.dropna().median())
        else:
            from sklearn.metrics import roc_curve
            fpr, tpr, thrs = roc_curve(sub["efficacy"].astype(int), sub["pred_score"].astype(float))
            j = tpr - fpr
            thr = float(thrs[np.nanargmax(j)])
    else:
        raise ValueError("PRED_GROUP_METHOD must be one of: median, 0.5, youden_from_efficacy")

    df["pred_threshold"] = thr
    df["pred_group"] = np.where(df["pred_score"] >= thr, "High predicted benefit", "Low predicted benefit")
    return df


# =========================
# Load survival
# =========================
def _prepare_survival_sheet(df, cohort_name):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    id_col = _find_first_existing(df.columns, ["序号", "patient_id", "id", "ID"])
    name_col = _find_first_existing(df.columns, ["患者姓名", "patient_name", "Name", "name", "姓名"])

    if id_col is None:
        raise ValueError(f"[{cohort_name}] cannot find ID column")

    out = pd.DataFrame()
    out["patient_id"] = df[id_col].map(_norm_id)
    out["patient_name"] = df[name_col].astype(str).str.strip() if name_col else ""

    # dose / disease state if present
    for c in df.columns:
        if "剂量组" in str(c):
            out["dose_group"] = df[c]
        if "入组时疾病状态" in str(c):
            out["disease_state"] = df[c]

    os_event_col = _find_col_contains(df.columns, ["是否死亡"])
    os_time_col = _find_col_contains(df.columns, ["OS"])

    if cohort_name == "high_risk":
        event_time_col = _find_col_contains(df.columns, ["EFS"])
        endpoint_name = "EFS"
    else:
        event_time_col = _find_col_contains(df.columns, ["PFS"])
        endpoint_name = "PFS"

    event_flag_col = _find_col_contains(df.columns, ["是否复发"])

    out["os_event"] = pd.to_numeric(df[os_event_col], errors="coerce") if os_event_col else np.nan
    out["os_time"] = pd.to_numeric(df[os_time_col], errors="coerce") if os_time_col else np.nan
    out["event_flag"] = pd.to_numeric(df[event_flag_col], errors="coerce") if event_flag_col else np.nan
    out["event_time"] = pd.to_numeric(df[event_time_col], errors="coerce") if event_time_col else np.nan
    out["event_name"] = endpoint_name
    out["cohort"] = cohort_name
    return out

def load_survival():
    high = pd.read_excel(SURV_XLSX, sheet_name=SHEET_HIGH_RISK)
    rr = pd.read_excel(SURV_XLSX, sheet_name=SHEET_RR)

    high_std = _prepare_survival_sheet(high, "high_risk")
    rr_std = _prepare_survival_sheet(rr, "relapse_refractory")
    return pd.concat([high_std, rr_std], ignore_index=True)


# =========================
# Load efficacy / predictions
# =========================
def load_efficacy():
    pf = pd.read_excel(DATA_XLSX, sheet_name=PATIENT_FEATURE_SHEET)
    pf.columns = [str(c).strip() for c in pf.columns]

    id_col = _find_first_existing(pf.columns, ["patient_id", "id", "ID"])
    name_col = _find_first_existing(pf.columns, ["patient_name", "Name", "name", "患者姓名", "姓名"])
    if id_col is None or "efficacy" not in pf.columns:
        raise ValueError("patient_features sheet must contain patient_id and efficacy")

    out = pd.DataFrame()
    out["patient_id"] = pf[id_col].map(_norm_id)
    out["efficacy"] = pd.to_numeric(pf["efficacy"], errors="coerce")
    out["patient_name_data"] = pf[name_col].astype(str).str.strip() if name_col else ""

    out = (
        out.groupby("patient_id", as_index=False)
           .agg(
               efficacy=("efficacy", lambda s: pd.to_numeric(s, errors="coerce").dropna().mode().iloc[0]
                         if len(pd.to_numeric(s, errors="coerce").dropna()) else np.nan),
               patient_name_data=("patient_name_data", lambda s: s.dropna().astype(str).mode().iloc[0]
                                  if len(s.dropna()) else "")
           )
    )
    return out

def load_predictions():
    df = pd.read_csv(OOF_CSV)
    df.columns = [str(c).strip() for c in df.columns]

    id_col = _find_first_existing(df.columns, ["patient_id", "id", "ID"])
    y_col = _find_first_existing(df.columns, ["y_true", "y", "label", "target"])

    if id_col is None:
        raise ValueError("oof_pred_mean.csv must contain patient_id")
    if SCORE_COL not in df.columns:
        raise ValueError(f"oof_pred_mean.csv missing {SCORE_COL}")

    out = pd.DataFrame()
    out["patient_id"] = df[id_col].map(_norm_id)
    out["pred_score"] = pd.to_numeric(df[SCORE_COL], errors="coerce")
    out["y_true_from_oof"] = pd.to_numeric(df[y_col], errors="coerce") if y_col else np.nan
    out = out.dropna(subset=["pred_score"]).copy()
    return out


# =========================
# Main
# =========================
def main():
    plt.rcParams["font.family"] = FONT_FAMILY

    print("[1/6] load survival data ...")
    surv = load_survival()
    print(surv.groupby("cohort").size())

    print("[2/6] load DATA.xlsx labels ...")
    labs = load_efficacy()

    print("[3/6] load OOF predictions ...")
    pred = load_predictions()

    print("[4/6] merge all ...")
    merged = surv.merge(labs, on="patient_id", how="left")
    merged = merged.merge(pred, on="patient_id", how="left")

    # prefer survival name, fallback to patient_features
    merged["patient_name_final"] = merged["patient_name"].astype(str).str.strip()
    m = merged["patient_name_final"].isna() | (merged["patient_name_final"] == "") | (merged["patient_name_final"].str.lower() == "nan")
    merged.loc[m, "patient_name_final"] = merged.loc[m, "patient_name_data"]

    merged = _make_pred_group(merged)

    merged_path = os.path.join(OUTDIR, "merged_survival_with_model.csv")
    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
    print("[Saved]", merged_path)

    # summaries
    summary_rows = []
    for coh, sub in merged.groupby("cohort"):
        endpoint_name = sub["event_name"].iloc[0]
        summary_rows.append({
            "cohort": coh,
            "n": int(len(sub)),
            "n_efficacy_nonmissing": int(sub["efficacy"].notna().sum()),
            "n_pred_nonmissing": int(sub["pred_score"].notna().sum()),
            "os_events": int(pd.to_numeric(sub["os_event"], errors="coerce").fillna(0).sum()),
            f"{endpoint_name}_events": int(pd.to_numeric(sub["event_flag"], errors="coerce").fillna(0).sum()),
        })
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(OUTDIR, "survival_summary.csv"),
        index=False, encoding="utf-8-sig"
    )

    logrank_rows = []
    cox_rows = []

    for coh, sub in merged.groupby("cohort"):
        endpoint_name = sub["event_name"].iloc[0]

        # A) by efficacy
        sub_eff = sub.dropna(subset=["efficacy"]).copy()
        if len(sub_eff) > 0:
            sub_eff["efficacy_group"] = np.where(pd.to_numeric(sub_eff["efficacy"], errors="coerce") == 1,
                                                 "Effective", "Ineffective")

            r = _km_plot(
                sub_eff, "os_time", "os_event", "efficacy_group",
                f"{coh}: OS by efficacy",
                os.path.join(OUTDIR, f"km_{coh}_os_by_efficacy.png")
            )
            if r is not None:
                r["cohort"] = coh
                r["analysis"] = "by_efficacy"
                logrank_rows.append(r)

            r = _km_plot(
                sub_eff, "event_time", "event_flag", "efficacy_group",
                f"{coh}: {endpoint_name} by efficacy",
                os.path.join(OUTDIR, f"km_{coh}_{endpoint_name.lower()}_by_efficacy.png")
            )
            if r is not None:
                r["cohort"] = coh
                r["analysis"] = "by_efficacy"
                logrank_rows.append(r)

        # B) by predicted group
        sub_pred = sub.dropna(subset=["pred_score", "pred_group"]).copy()
        if len(sub_pred) > 0:
            r = _km_plot(
                sub_pred, "os_time", "os_event", "pred_group",
                f"{coh}: OS by model-predicted group",
                os.path.join(OUTDIR, f"km_{coh}_os_by_predgroup.png")
            )
            if r is not None:
                r["cohort"] = coh
                r["analysis"] = "by_pred_group"
                r["threshold"] = float(sub_pred["pred_threshold"].dropna().iloc[0])
                logrank_rows.append(r)

            r = _km_plot(
                sub_pred, "event_time", "event_flag", "pred_group",
                f"{coh}: {endpoint_name} by model-predicted group",
                os.path.join(OUTDIR, f"km_{coh}_{endpoint_name.lower()}_by_predgroup.png")
            )
            if r is not None:
                r["cohort"] = coh
                r["analysis"] = "by_pred_group"
                r["threshold"] = float(sub_pred["pred_threshold"].dropna().iloc[0])
                logrank_rows.append(r)

            c1 = _run_univariable_cox(sub_pred, "os_time", "os_event", "pred_score", coh, "OS")
            if c1 is not None:
                cox_rows.append(c1)
            c2 = _run_univariable_cox(sub_pred, "event_time", "event_flag", "pred_score", coh, endpoint_name)
            if c2 is not None:
                cox_rows.append(c2)

    if logrank_rows:
        pd.DataFrame(logrank_rows).to_csv(
            os.path.join(OUTDIR, "logrank_results.csv"),
            index=False, encoding="utf-8-sig"
        )

    if cox_rows:
        pd.concat(cox_rows, ignore_index=True).to_csv(
            os.path.join(OUTDIR, "cox_score_results.csv"),
            index=False, encoding="utf-8-sig"
        )

    print("[DONE] survival analysis finished.")
    print("建议优先看：")
    print(" - km_high_risk_efs_by_predgroup.png")
    print(" - km_relapse_refractory_pfs_by_predgroup.png")
    print(" - km_high_risk_efs_by_efficacy.png")
    print(" - km_relapse_refractory_pfs_by_efficacy.png")
    print(" - cox_score_results.csv")


if __name__ == "__main__":
    main()
