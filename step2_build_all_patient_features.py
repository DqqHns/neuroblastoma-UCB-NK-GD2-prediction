# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from scipy import stats

XLSX_PATH = r"data/DATA.xlsx"
SHEET_NAME = None  
ID_COL = "Name"
TIME_COL = "Relative date"
LABEL_COL = "efficacy"



DATE_COL = "Date"
GROUP_COL = "合并用药组别(1=化疗+GD2+NK，2=GD2联合NK)"
DOSE_COL = "dose"

OUT_SHEET = "patient_features_all"


MAX_PATIENT_MISSING_RATE_MEAN = 0.60      # XXX_mean 缺失率>0.6就删
MAX_PATIENT_MISSING_RATE_BASE = 0.60      # XXX_baseline 缺失率>0.6就删
MAX_PATIENT_MISSING_RATE_SLOPE = 0.70     # slope 更严格/更宽松都行；这里给0.7
MIN_POINTS_FOR_SLOPE = 2                  # 至少2个时间点才算斜率

def auto_pick_long_sheet(xlsx_path: str) -> str:
    xls = pd.ExcelFile(xlsx_path)
    best = None
    best_score = -1
    for sh in xls.sheet_names:
        df0 = pd.read_excel(xlsx_path, sheet_name=sh, nrows=30)
        cols = set(df0.columns.astype(str))
        score = 0
        # 必备列加分
        score += 50 if ID_COL in cols else 0
        score += 50 if TIME_COL in cols else 0
        score += 50 if LABEL_COL in cols else 0
        # 列多更像长表
        score += min(len(cols), 200) * 0.2
        if score > best_score:
            best_score = score
            best = sh
    if best is None:
        raise ValueError("No suitable sheet found.")
    return best

def patient_mode(s: pd.Series):
    s2 = s.dropna()
    if len(s2) == 0:
        return np.nan
    # 众数可能多个，取第一个
    vc = s2.value_counts()
    return vc.index[0]

def baseline_at_earliest_time(df: pd.DataFrame, value_col: str) -> pd.Series:
    sub = df[[ID_COL, TIME_COL, value_col]].dropna().copy()
    if sub.empty:
        return pd.Series(dtype=float)
    tmin = sub.groupby(ID_COL)[TIME_COL].min()
    sub = sub.join(tmin.rename("tmin"), on=ID_COL)
    sub_earliest = sub[sub[TIME_COL] == sub["tmin"]].copy()
    # 最早时间点若有多行，取平均
    return sub_earliest.groupby(ID_COL)[value_col].mean()

def robust_slope(time: np.ndarray, value: np.ndarray):
    if len(time) < MIN_POINTS_FOR_SLOPE:
        return np.nan, int(len(time))
    try:
        slope, intercept, lo, hi = stats.theilslopes(value, time)
        return float(slope), int(len(time))
    except Exception:
        return np.nan, int(len(time))

def main():
    sheet = SHEET_NAME or auto_pick_long_sheet(XLSX_PATH)
    df = pd.read_excel(XLSX_PATH, sheet_name=sheet)
    print(f"[OK] Using raw long sheet: {sheet}, shape={df.shape}")

    # 基础清洗
    df = df.copy()
    df = df[df[ID_COL].notna()].copy()
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")

    # 只保留有标签的患者
    df = df[df[LABEL_COL].isin([0, 1])].copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    # 组装患者级骨架
    feat = pd.DataFrame(index=sorted(df[ID_COL].unique()))
    feat[LABEL_COL] = df.groupby(ID_COL)[LABEL_COL].agg(patient_mode).astype(int)

    for meta in [GROUP_COL, DOSE_COL]:
        if meta in df.columns:
            feat[meta] = pd.to_numeric(df.groupby(ID_COL)[meta].agg(patient_mode), errors="coerce")

    # 时间统计
    feat["tmin"] = df.groupby(ID_COL)[TIME_COL].min()
    feat["tmax"] = df.groupby(ID_COL)[TIME_COL].max()
    feat["n_timepoints"] = df.groupby(ID_COL)[TIME_COL].apply(lambda s: s.dropna().nunique())

    # 自动找“数值指标列”
    exclude = {ID_COL, TIME_COL, LABEL_COL, DATE_COL, GROUP_COL, DOSE_COL}
    candidate_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        candidate_cols.append(c)

    print(f"[INFO] Candidate numeric columns: {len(candidate_cols)}")

    # 对每个指标做 baseline/mean/slope
    for c in candidate_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

        # baseline
        feat[f"{c}_baseline"] = baseline_at_earliest_time(df, c)

        # mean
        feat[f"{c}_mean"] = df.groupby(ID_COL)[c].mean()

        # slope
        slope_rows = []
        for pid, g in df[[ID_COL, TIME_COL, c]].dropna().groupby(ID_COL):
            g = g.sort_values(TIME_COL)
            s, n = robust_slope(g[TIME_COL].to_numpy(float), g[c].to_numpy(float))
            slope_rows.append((pid, s, n))
        slope_df = pd.DataFrame(
            slope_rows,
            columns=[ID_COL, f"{c}_slope", f"{c}_slope_n"]
        ).set_index(ID_COL)
        feat = feat.join(slope_df, how="left")

    # reset index -> patient_id
    feat = feat.reset_index()
    # reset_index 后第一列就是患者ID（原来的index），列名通常是 "index"
    feat = feat.rename(columns={feat.columns[0]: "patient_id"})
    feat["patient_id"] = feat["patient_id"].astype(str).str.strip()

    # ===== 过滤缺失太多的特征（按患者级缺失率）=====
    keep_cols = ["patient_id", LABEL_COL]
    for meta in [GROUP_COL, DOSE_COL, "tmin", "tmax", "n_timepoints"]:
        if meta in feat.columns:
            keep_cols.append(meta)

    # 把所有派生列按类型过滤
    all_cols = [c for c in feat.columns if c not in keep_cols]
    mean_cols = [c for c in all_cols if c.endswith("_mean")]
    base_cols = [c for c in all_cols if c.endswith("_baseline")]
    slope_cols = [c for c in all_cols if c.endswith("_slope")]
    slope_n_cols = [c for c in all_cols if c.endswith("_slope_n")]

    def filter_by_missing(cols, thr):
        miss = feat[cols].isna().mean()
        return miss[miss <= thr].index.tolist()

    keep_mean = filter_by_missing(mean_cols, MAX_PATIENT_MISSING_RATE_MEAN)
    keep_base = filter_by_missing(base_cols, MAX_PATIENT_MISSING_RATE_BASE)
    keep_slope = filter_by_missing(slope_cols, MAX_PATIENT_MISSING_RATE_SLOPE)

    # slope_n 一般跟 slope 配套保留
    keep_slope_n = []
    slope_to_n = {c.replace("_slope", "_slope_n"): c for c in keep_slope}
    for ncol in slope_n_cols:
        # 只保留“对应的 slope 被保留”的 slope_n
        if ncol in slope_to_n:
            keep_slope_n.append(ncol)

    final_cols = keep_cols + keep_base + keep_mean + keep_slope + keep_slope_n
    feat2 = feat[final_cols].copy()

    print(f"[INFO] Final columns: {len(final_cols)} (base={len(keep_base)}, mean={len(keep_mean)}, slope={len(keep_slope)})")

    # 写回同一个 DATA.xlsx 的新 sheet
    with pd.ExcelWriter(XLSX_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        feat2.to_excel(w, sheet_name=OUT_SHEET, index=False)

    print(f"[Saved] Sheet '{OUT_SHEET}' -> {XLSX_PATH}")

if __name__ == "__main__":
    main()
