# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from scipy.stats import wilcoxon


# -----------------------------
# Config
# -----------------------------
XLSX_PATH = "data/features_table.xlsx"     
OUTDIR = "compare_groups_results_strict"
os.makedirs(OUTDIR, exist_ok=True)

LABEL_COL = "efficacy"


PID_COL = "patient_id"
PNAME_COL = "patient_name"


FALLBACK_ID_COLS = ["patient_id", "id", "ID", "Id"]
FALLBACK_NAME_COLS = ["patient_name", "Name", "name", "姓名", "患者姓名"]

GROUP_COL = "合并用药组别(1=化疗+GD2+NK，2=GD2联合NK)"
GROUP_FEAT = "group01"  # raw-1 => 0/1


FINAL_FEATURES = [
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

# group置换次数（检验“组间AUC差异”是否显著）
N_GROUP_PERM = 2000
GROUP_PERM_SEED = 2026

# 患者层面 bootstrap
N_BOOT = 5000
BOOT_SEED = 2026



def _set_chinese_font():

    try:
        import matplotlib
        
        matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

# -----------------------------
# Helpers
# -----------------------------
def _to_numeric_inplace(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def add_group01(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if GROUP_FEAT not in df.columns and GROUP_COL in df.columns:
        df[GROUP_FEAT] = pd.to_numeric(df[GROUP_COL], errors="coerce") - 1.0
    return df


def make_pipeline():

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="liblinear", max_iter=2000, C=1.0, penalty="l2")),
    ])


def safe_auc(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))


def summarize_vec(x):
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(float)
    if len(x) == 0:
        return {"mean": np.nan, "sd": np.nan, "p10": np.nan, "p50": np.nan, "p90": np.nan, "n": 0}
    return {
        "mean": float(np.mean(x)),
        "sd": float(np.std(x)),
        "p10": float(np.quantile(x, 0.10)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "n": int(len(x)),
    }


def _pick_first_existing(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_table(xlsx_path: str) -> pd.DataFrame:
    """读取 xlsx：优先读第一张 sheet；并把 ID/Name 列统一成 patient_id/patient_name。"""
    xls = pd.ExcelFile(xlsx_path)
    df = pd.read_excel(xlsx_path, sheet_name=xls.sheet_names[0])
    df.columns = [str(c).strip() for c in df.columns]

    id_col = _pick_first_existing(df, FALLBACK_ID_COLS)
    if id_col is None:
        raise KeyError(f"[ERROR] cannot find id column. tried: {FALLBACK_ID_COLS}")
    if id_col != PID_COL:
        df = df.rename(columns={id_col: PID_COL})

    name_col = _pick_first_existing(df, FALLBACK_NAME_COLS)
    if name_col is not None and name_col != PNAME_COL:
        df = df.rename(columns={name_col: PNAME_COL})

    # 清理
    df[PID_COL] = df[PID_COL].astype(str).str.strip()
    if PNAME_COL in df.columns:
        df[PNAME_COL] = df[PNAME_COL].astype(str).str.strip()

    return df


def run_cv50_strict_group_auc(df: pd.DataFrame):
   
    use_cols = [PID_COL, LABEL_COL, GROUP_FEAT] + FINAL_FEATURES
    if PNAME_COL in df.columns:
        use_cols.insert(1, PNAME_COL)

    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        raise KeyError(f"[ERROR] Missing columns: {missing}")

    sub = df[use_cols].copy()
    sub.columns = [str(c).strip() for c in sub.columns]

    # only coerce numeric columns (avoid id/name)
    num_cols = [LABEL_COL, GROUP_FEAT] + FINAL_FEATURES
    _to_numeric_inplace(sub, num_cols)

    before = len(sub)
    sub = sub.dropna(subset=num_cols)  # 不因为 name 缺失而 drop
    after = len(sub)

    sub[LABEL_COL] = sub[LABEL_COL].astype(int)
    sub[GROUP_FEAT] = sub[GROUP_FEAT].astype(int)

    print(f"[INFO] Strict-eval subset: {after} rows (dropped {before-after} due to NA in numeric cols)")
    print("[INFO] Label counts:", dict(pd.Series(sub[LABEL_COL]).value_counts()))
    print("[INFO] Group counts:", dict(pd.Series(sub[GROUP_FEAT]).value_counts()))

    X = sub[FINAL_FEATURES].values
    y = sub[LABEL_COL].values
    g = sub[GROUP_FEAT].values
    pid = sub[PID_COL].astype(str).values
    pname = sub[PNAME_COL].astype(str).values if PNAME_COL in sub.columns else np.array([""] * len(sub), dtype=object)

    pipe = make_pipeline()

    rep_rows = []
    all_oof_rows = []

    for rep in range(1, N_REPEATS + 1):
        seed = BASE_RANDOM_STATE + rep
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

        oof_prob = np.full(len(sub), np.nan, dtype=float)

        for fold, (tr, te) in enumerate(cv.split(X, y), start=1):
            pipe.fit(X[tr], y[tr])
            prob = pipe.predict_proba(X[te])[:, 1]
            oof_prob[te] = prob

            # 保存 OOF 明细（patient-level追踪必需）
            for idx, p in zip(te, prob):
                all_oof_rows.append({
                    "repeat": rep,
                    "fold": fold,
                    "patient_id": str(pid[idx]),
                    "patient_name": str(pname[idx]),
                    "y_true": int(y[idx]),
                    "y_prob": float(p),
                    "group01": int(g[idx]),
                })

        # 用该 repeat 的 OOF 预测，分别算各组 AUC
        auc_all = safe_auc(y, oof_prob)
        m0 = (g == 0)
        m1 = (g == 1)
        auc_g0 = safe_auc(y[m0], oof_prob[m0])
        auc_g1 = safe_auc(y[m1], oof_prob[m1])

        rep_rows.append({
            "repeat": rep,
            "random_state": seed,
            "n_all": int(len(sub)),
            "n_g0": int(m0.sum()),
            "n_g1": int(m1.sum()),
            "pos_g0": int((y[m0] == 1).sum()),
            "neg_g0": int((y[m0] == 0).sum()),
            "pos_g1": int((y[m1] == 1).sum()),
            "neg_g1": int((y[m1] == 0).sum()),
            "auc_all": auc_all,
            "auc_g0": auc_g0,
            "auc_g1": auc_g1,
            "delta_g1_minus_g0": (auc_g1 - auc_g0) if np.isfinite(auc_g0) and np.isfinite(auc_g1) else np.nan,
        })

    rep_df = pd.DataFrame(rep_rows)
    oof_df = pd.DataFrame(all_oof_rows)
    return rep_df, oof_df


def perm_test_group_delta(oof_df: pd.DataFrame, n_perm=2000, seed=2026):
   
    rng = np.random.default_rng(seed)

        # 只在“有明确组别”的样本上做组间差异检验（group01 允许缺失）
    m = oof_df["group01"].notna().to_numpy(bool)
    y = oof_df.loc[m, "y_true"].to_numpy(int)
    p = oof_df.loc[m, "y_prob"].to_numpy(float)
    g = oof_df.loc[m, "group01"].to_numpy(int)

    def delta_for_group(gx):
        m0 = gx == 0
        m1 = gx == 1
        auc0 = safe_auc(y[m0], p[m0])
        auc1 = safe_auc(y[m1], p[m1])
        if not (np.isfinite(auc0) and np.isfinite(auc1)):
            return np.nan
        return float(auc1 - auc0)

    delta_obs = delta_for_group(g)

    deltas = []
    for _ in range(n_perm):
        gp = rng.permutation(g)
        d = delta_for_group(gp)
        if np.isfinite(d):
            deltas.append(d)

    deltas = np.array(deltas, dtype=float)
    if not np.isfinite(delta_obs) or len(deltas) == 0:
        return {"delta_obs": float(delta_obs), "p_emp": np.nan, "n_perm_valid": int(len(deltas))}

    p_emp = float(np.mean(np.abs(deltas) >= abs(delta_obs)))
    return {"delta_obs": float(delta_obs), "p_emp": p_emp, "n_perm_valid": int(len(deltas))}



# -----------------------------
# Plotting
# -----------------------------
def make_plots(rep_df: pd.DataFrame, pat_df: pd.DataFrame, outdir: str):

    _set_chinese_font()

    # 1) AUC boxplot
    fig = plt.figure(figsize=(7, 4.5))
    data = [rep_df["auc_g0"].dropna().values, rep_df["auc_g1"].dropna().values]
    plt.boxplot(data, labels=["group0", "group1"], showmeans=True)
    plt.ylabel("AUC (per repeat)")
    plt.title("AUC distribution across repeats (OOF, same model)")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    out1 = os.path.join(outdir, "plot_auc_boxplot.png")
    plt.tight_layout()
    plt.savefig(out1, dpi=200)
    plt.close(fig)

    # 2) delta histogram
    fig = plt.figure(figsize=(7, 4.5))
    delta = pd.to_numeric(rep_df["delta_g1_minus_g0"], errors="coerce").dropna().values
    plt.hist(delta, bins=12)
    plt.xlabel("Δ = AUC(g1) - AUC(g0)")
    plt.ylabel("Count (repeats)")
    plt.title("Distribution of Δ across repeats")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    out2 = os.path.join(outdir, "plot_delta_hist.png")
    plt.tight_layout()
    plt.savefig(out2, dpi=200)
    plt.close(fig)

    # 3) group1 patient-level mean probs
    if pat_df is not None and len(pat_df):
        g1 = pat_df[pat_df["group01"] == 1].copy()
        if len(g1) > 0 and "y_prob_mean" in g1.columns:
            g1 = g1.sort_values(["y_true", "y_prob_mean"], ascending=[True, True]).reset_index(drop=True)

            def _mk_label(row):
                name = str(row.get("patient_name", "")).strip()
                pid = str(row.get("patient_id", "")).strip()
                if name and name.lower() != "nan":
                    return name
                return pid

            labels = [_mk_label(r) for _, r in g1.iterrows()]

            fig = plt.figure(figsize=(max(9, 0.7 * len(g1)), 5.5))
            x = list(range(len(g1)))
            plt.scatter(x, g1["y_prob_mean"].values, marker="o")
            plt.axhline(0.5, linewidth=1)

            # annotate y_true above points
            for i, (_, row) in enumerate(g1.iterrows()):
                try:
                    yt = int(row["y_true"])
                except Exception:
                    yt = row["y_true"]
                plt.text(i, float(row["y_prob_mean"]) + 0.02, str(yt),
                         ha="center", va="bottom", fontsize=8)

            plt.xticks(x, labels, rotation=60, ha="right")
            plt.ylabel("Mean OOF predicted probability (across repeats)")
            plt.title("Group1 patient-level mean predictions (label shown above point)")
            plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
            out3 = os.path.join(outdir, "plot_group1_patient_probs.png")
            plt.tight_layout()
            plt.savefig(out3, dpi=200)
            plt.close(fig)

    return {
        "plot_auc_boxplot": out1,
        "plot_delta_hist": out2,
        "plot_group1_patient_probs": os.path.join(outdir, "plot_group1_patient_probs.png"),
    }


# -----------------------------
# Patient-level (sample-level) analysis
# -----------------------------
def patient_level_mean_oof(oof_df: pd.DataFrame) -> pd.DataFrame:

    cols = ["patient_id", "patient_name", "y_true", "group01", "y_prob"]
    df = oof_df[cols].copy()
    # y_true/group01 理论上对同一患者固定；用 mode
    def mode_int(s):
        vc = s.value_counts()
        return int(vc.index[0])

    out = (
        df.groupby("patient_id", as_index=False)
          .agg(
              patient_name=("patient_name", lambda s: s.dropna().astype(str).mode().iloc[0] if len(s.dropna()) else ""),
              y_true=("y_true", mode_int),
              group01=("group01", mode_int),
              y_prob_mean=("y_prob", "mean"),
              y_prob_sd=("y_prob", "std"),
              n_preds=("y_prob", "count"),
          )
    )
    return out

def save_results_with_standard_deviation(df_patient_level: pd.DataFrame, output_path: str):
    
    # 保存结果时加入标准差
    df_patient_level["y_prob_mean"] = df_patient_level["y_prob"].mean(axis=1)
    df_patient_level["y_prob_sd"] = df_patient_level["y_prob"].std(axis=1)

    # 保存到 Excel
    df_patient_level.to_excel(output_path, index=False)
    print(f"Results with standard deviation saved to {output_path}")


def bootstrap_auc_ci(patient_df: pd.DataFrame, n_boot=5000, seed=2026):

    rng = np.random.default_rng(seed)

    g0 = patient_df[patient_df["group01"] == 0].reset_index(drop=True)
    g1 = patient_df[patient_df["group01"] == 1].reset_index(drop=True)

    def _boot_auc(df_):
        # sample with replacement, same size
        idx = rng.integers(0, len(df_), size=len(df_))
        y = df_.loc[idx, "y_true"].to_numpy(int)
        p = df_.loc[idx, "y_prob_mean"].to_numpy(float)
        return safe_auc(y, p)

    auc0_list, auc1_list, delta_list = [], [], []
    for _ in range(n_boot):
        a0 = _boot_auc(g0) if len(g0) else np.nan
        a1 = _boot_auc(g1) if len(g1) else np.nan
        if np.isfinite(a0):
            auc0_list.append(a0)
        if np.isfinite(a1):
            auc1_list.append(a1)
        if np.isfinite(a0) and np.isfinite(a1):
            delta_list.append(a1 - a0)

    def _ci(arr):
        arr = np.asarray(arr, float)
        if len(arr) == 0:
            return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_valid": 0}
        return {
            "mean": float(np.mean(arr)),
            "ci_low": float(np.quantile(arr, 0.025)),
            "ci_high": float(np.quantile(arr, 0.975)),
            "std": float(np.std(arr)),  # 计算标准差
            "n_valid": int(len(arr)),
        }

    return {
        "auc_g0": _ci(auc0_list),
        "auc_g1": _ci(auc1_list),
        "delta_g1_minus_g0": _ci(delta_list),
        "n_boot": int(n_boot),
    }


def loo_influence_group1(patient_df: pd.DataFrame):

    g1 = patient_df[patient_df["group01"] == 1].reset_index(drop=True).copy()
    if len(g1) < 3:
        return pd.DataFrame()

    y_full = g1["y_true"].to_numpy(int)
    p_full = g1["y_prob_mean"].to_numpy(float)
    auc_full = safe_auc(y_full, p_full)

    rows = []
    for i in range(len(g1)):
        sub = g1.drop(index=i).reset_index(drop=True)
        auc_i = safe_auc(sub["y_true"].to_numpy(int), sub["y_prob_mean"].to_numpy(float))
        rows.append({
            "left_out_patient_id": str(g1.loc[i, "patient_id"]),
            "left_out_patient_name": str(g1.loc[i, "patient_name"]),
            "auc_g1_full": auc_full,
            "auc_g1_without_i": auc_i,
            "delta_auc": (auc_i - auc_full) if np.isfinite(auc_full) and np.isfinite(auc_i) else np.nan
        })

    out = pd.DataFrame(rows).sort_values("delta_auc", key=lambda s: np.abs(s), ascending=False)
    return out


def main():
    df = load_table(XLSX_PATH)
    df = add_group01(df)

    # 关键数值列转 numeric
    _to_numeric_inplace(df, [LABEL_COL, GROUP_FEAT] + FINAL_FEATURES)

    rep_df, oof_df = run_cv50_strict_group_auc(df)

    # repeat-wise 配对 Wilcoxon
    d = rep_df[["auc_g0", "auc_g1", "delta_g1_minus_g0"]].dropna()
    p_wil = np.nan
    if len(d) >= 5 and not np.allclose(d["delta_g1_minus_g0"].values, 0):
        try:
            _, p_wil = wilcoxon(d["auc_g0"].values, d["auc_g1"].values)
        except Exception:
            p_wil = np.nan

    # 经验置换检验：检验“组间差异”
    perm_res = perm_test_group_delta(oof_df, n_perm=N_GROUP_PERM, seed=GROUP_PERM_SEED)

    # 患者层面分析（bootstrap CI + LOO）
    pat_df = patient_level_mean_oof(oof_df)

    boot_res = bootstrap_auc_ci(pat_df, n_boot=N_BOOT, seed=BOOT_SEED)
    # save_results_with_standard_deviation(pat_df, output_path="G:/Dqq_code/nk_model(1.8)/compare_groups_results_strict/fangcha.xlsx")
    loo_df = loo_influence_group1(pat_df)


    def flat(prefix, dct):
        return {f"{prefix}_{k}": v for k, v in dct.items()}

    summary_row = {
        "xlsx_used": XLSX_PATH,
        "features": ", ".join(FINAL_FEATURES),
        "cv": f"{N_SPLITS}-fold x {N_REPEATS} repeats",
        **flat("repeat_auc_all", summarize_vec(rep_df["auc_all"])),
        **flat("repeat_auc_g0", summarize_vec(rep_df["auc_g0"])),
        **flat("repeat_auc_g1", summarize_vec(rep_df["auc_g1"])),
        **flat("repeat_delta", summarize_vec(rep_df["delta_g1_minus_g0"])),
        "p_wilcoxon_repeatwise_g0_vs_g1": p_wil,
        "perm_delta_obs": perm_res["delta_obs"],
        "perm_p_empirical_2sided": perm_res["p_emp"],
        "perm_n_valid": perm_res["n_perm_valid"],
        "patient_level_n": int(len(pat_df)),
        "patient_level_g0_n": int((pat_df["group01"] == 0).sum()),
        "patient_level_g1_n": int((pat_df["group01"] == 1).sum()),
        **flat("boot_auc_g0", boot_res["auc_g0"]),
        **flat("boot_auc_g1", boot_res["auc_g1"]),
        **flat("boot_delta", boot_res["delta_g1_minus_g0"]),
        "boot_n_boot": boot_res["n_boot"],
        "boot_auc_g0_std": boot_res["auc_g0"]["std"],  # 添加标准差
        "boot_auc_g1_std": boot_res["auc_g1"]["std"],  # 添加标准差
        "boot_delta_std": boot_res["delta_g1_minus_g0"]["std"],  # 添加标准差
    }

    out_csv = os.path.join(OUTDIR, "strict_group_compare_summary_v2.csv")
    pd.DataFrame([summary_row]).to_csv(out_csv, index=False, encoding="utf-8-sig")

    out_xlsx = os.path.join(OUTDIR, "strict_group_compare_details_v2.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        pd.DataFrame([summary_row]).to_excel(w, sheet_name="summary", index=False)
        rep_df.to_excel(w, sheet_name="per_repeat_auc", index=False)
        oof_df.to_excel(w, sheet_name="oof_preds", index=False)
        pat_df.to_excel(w, sheet_name="patient_level_mean_oof", index=False)
        pd.DataFrame([{
            "auc_g0_mean": boot_res["auc_g0"]["mean"],
            "auc_g0_ci_low": boot_res["auc_g0"]["ci_low"],
            "auc_g0_ci_high": boot_res["auc_g0"]["ci_high"],
            "auc_g1_mean": boot_res["auc_g1"]["mean"],
            "auc_g1_ci_low": boot_res["auc_g1"]["ci_low"],
            "auc_g1_ci_high": boot_res["auc_g1"]["ci_high"],
            "delta_mean": boot_res["delta_g1_minus_g0"]["mean"],
            "delta_ci_low": boot_res["delta_g1_minus_g0"]["ci_low"],
            "delta_ci_high": boot_res["delta_g1_minus_g0"]["ci_high"],
            "n_boot": boot_res["n_boot"],
            "boot_auc_g0_std": boot_res["auc_g0"]["std"],  # 添加标准差
            "boot_auc_g1_std": boot_res["auc_g1"]["std"],  # 添加标准差
            "boot_delta_std": boot_res["delta_g1_minus_g0"]["std"],  # 添加标准差
        }]).to_excel(w, sheet_name="patient_bootstrap_ci", index=False)
        loo_df.to_excel(w, sheet_name="group1_loo_influence", index=False)

    # ---- Plots ----
    try:
        plot_paths = make_plots(rep_df, pat_df, OUTDIR)
        print("[INFO] Plots saved:")
        for k, v in plot_paths.items():
            if v and os.path.exists(v):
                print("   -", k, "=>", v)
    except Exception as e:
        print("[WARN] Plotting failed:", repr(e))

    print("\n[Saved]")
    print(" -", out_csv)
    print(" -", out_xlsx)
    print("\nDONE.")


if __name__ == "__main__":
    main()
