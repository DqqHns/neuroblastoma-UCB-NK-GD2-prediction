import os
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.impute import SimpleImputer

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 200)
pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.unicode.ambiguous_as_wide", True)

# -----------------------------
# 0) Config
# -----------------------------
DATA_PATH = r"data/DATA.xlsx"
SHEET_NAME = "筛选后42人（加入亚组）"
CHECK_PATIENT_N = True

OUTDIR = "./outputs"
os.makedirs(OUTDIR, exist_ok=True)


SPARSE_CANDIDATES = [
    "PT", "PT%", "INR", "APTT", "Fbg", "TT", "D-D", "FDP", "ATIII",
    "肌酸激酶同工酶（质量法）", "IL-6", "IL-8(pg/mL)", "IL-1β(pg/mL)",
    "TNF-α(pg/mL)", "IL-2R(pg/mL)"
]

ID_COL = "Name"
DATE_COL = "Date"
TIME_COL = "Relative date"
LABEL_COL = "efficacy"
GROUP_COL = "合并用药组别(1=化疗+GD2+NK，2=GD2联合NK)"
DOSE_COL = "dose"

# -----------------------------
# 1) Load Excel (fixed sheet)
# -----------------------------
def load_excel_by_sheet(path, sheet_name):
    df = pd.read_excel(path, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]
    return df

df_raw = load_excel_by_sheet(DATA_PATH, SHEET_NAME)
sheet_used = SHEET_NAME


if CHECK_PATIENT_N:
    if ID_COL not in df_raw.columns:
        raise ValueError(f"ID_COL='{ID_COL}' not found in sheet '{SHEET_NAME}'. "
                         f"Available cols: {list(df_raw.columns)[:30]} ...")
    n_pat = df_raw[ID_COL].nunique(dropna=True)
    if n_pat != 42:
        raise ValueError(f"Expect 42 patients in sheet '{SHEET_NAME}', but got {n_pat}. "
                         f"(If you don't want to enforce 42, set CHECK_PATIENT_N=False)")


# -----------------------------
# 2) Basic cleaning
# -----------------------------
def basic_clean(df):
    df = df.copy()

    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

    df.columns = [re.sub(r"\s+", " ", str(c).strip()) for c in df.columns]

    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")

    for col in [LABEL_COL, GROUP_COL, DOSE_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if ID_COL in df.columns:
        df[ID_COL] = df[ID_COL].astype("string").str.strip()
        df.loc[df[ID_COL].isin(["", "nan", "NaN", "None", "<NA>"]), ID_COL] = pd.NA


    return df

def make_columns_unique(df):
    """
    如果Excel里有重复列名，df['某列'] 会返回DataFrame而不是Series，
    会导致 spearmanr 返回矩阵 -> 写入单元格时报错。
    这里把重复列名加后缀：col, col__2, col__3 ...
    """
    cols = list(df.columns)
    seen = {}
    new_cols = []
    for c in cols:
        c = str(c).strip()
        if c not in seen:
            seen[c] = 1
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}__{seen[c]}")
    df = df.copy()
    df.columns = new_cols
    return df


df = basic_clean(df_raw)
df = make_columns_unique(df)


print("\n[Using sheet]:", sheet_used)
print("[Shape]:", df.shape)
print("[Unique patients]:", df[ID_COL].nunique())

# -----------------------------
# 3) Identify numeric features
# -----------------------------

def get_numeric_features(df):
    # 排除ID/日期/TIME/LABEL/GROUP/DOSE 参与相关性等
    exclude = set([ID_COL, DATE_COL, "Gender", TIME_COL, LABEL_COL, GROUP_COL, DOSE_COL])
    # exclude |= set([TIME_COL, LABEL_COL, GROUP_COL, DOSE_COL])

    # if "Gender" in df.columns:
    #     df["Gender"] = df["Gender"].map({"男": 1, "女": 0})

    cand_cols = [c for c in df.columns if c not in exclude]
    df_num = df[cand_cols].copy()
    for c in df_num.columns:
        df_num[c] = pd.to_numeric(df_num[c], errors="coerce")

    numeric_cols = [c for c in df_num.columns if pd.api.types.is_numeric_dtype(df_num[c])]
    return numeric_cols

num_cols = get_numeric_features(df)

for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

print("\n[Numeric columns count]:", len(num_cols))

# -----------------------------
# 4) Missingness profiling
# -----------------------------
def missingness_report(df, cols, by=None):
    out = []
    for c in cols:
        miss = df[c].isna().mean()
        out.append((c, miss, df[c].notna().sum()))
    m = pd.DataFrame(out, columns=["feature", "missing_rate", "n_non_missing"])\
          .sort_values("missing_rate", ascending=False)

    if by is not None and by in df.columns:
        grp = df.groupby(by)
        stats_by = []
        for g, sub in grp:
            for c in cols:
                stats_by.append((g, c, sub[c].isna().mean(), sub[c].notna().sum()))
        mb = pd.DataFrame(stats_by, columns=[by, "feature", "missing_rate", "n_non_missing"])
        return m, mb

    return m, None

miss_all, miss_by_eff = missingness_report(df, num_cols, by=LABEL_COL if LABEL_COL in df.columns else None)
miss_all.to_csv(os.path.join(OUTDIR, "missingness_overall.csv"), index=False)
if miss_by_eff is not None:
    miss_by_eff.to_csv(os.path.join(OUTDIR, "missingness_by_efficacy.csv"), index=False)

print("\n[Top missing features]\n" + miss_all.head(20).to_string(index=False))

# Missingness heatmap (sampled columns to keep readable)
def plot_missingness_heatmap(df, cols, title, fname, max_cols=60):
    miss = df[cols].isna().mean().sort_values(ascending=False)
    top_cols = miss.head(max_cols).index.tolist()

    plt.figure(figsize=(16, 6))
    sns.heatmap(df[top_cols].isna(), cbar=False)
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, fname), dpi=200)
    plt.close()

plot_missingness_heatmap(
    df, num_cols,
    title="Missingness Map (top missing features)",
    fname="missingness_heatmap_top_missing.png",
    max_cols=70
)

# -----------------------------
# 5) Descriptive stats (overall + by efficacy)
# -----------------------------

def descriptive_table(df, cols):
    desc = []
    for c in cols:
        x = df[c].dropna()
        if len(x) == 0:
            continue
        desc.append({
            "feature": c,
            "n": len(x),
            "mean": x.mean(),
            "std": x.std(),
            "median": x.median(),
            "q25": x.quantile(0.25),
            "q75": x.quantile(0.75),
            "min": x.min(),
            "max": x.max(),
            "skew": x.skew(),
            "kurtosis": x.kurtosis()
        })
    return pd.DataFrame(desc).sort_values("n", ascending=False)

desc_all = descriptive_table(df, num_cols)
desc_all.to_csv(os.path.join(OUTDIR, "descriptive_overall.csv"), index=False)

if LABEL_COL in df.columns:
    desc_by = []
    for g, sub in df.groupby(LABEL_COL):
        t = descriptive_table(sub, num_cols)
        t.insert(0, LABEL_COL, g)
        desc_by.append(t)
    desc_by = pd.concat(desc_by, ignore_index=True)
    desc_by.to_csv(os.path.join(OUTDIR, "descriptive_by_efficacy.csv"), index=False)

# 分布图：挑“缺失较少 + 医学常用”的前N列
def plot_feature_distributions(df, cols, by=None, n=20):
    # 按缺失率从低到高挑n个，便于看
    miss = df[cols].isna().mean().sort_values()
    pick = miss.head(n).index.tolist()

    for c in pick:
        plt.figure(figsize=(8,4))
        if by is not None and by in df.columns:
            # 叠加密度/直方图（分组）
            for g, sub in df.groupby(by):
                x = sub[c].dropna()
                if len(x) < 3:
                    continue
                plt.hist(x, bins=25, alpha=0.4, density=True, label=f"{by}={g}")
            plt.legend()
            plt.title(f"Distribution: {c} (by {by})")
        else:
            x = df[c].dropna()
            plt.hist(x, bins=25, alpha=0.8, density=True)
            plt.title(f"Distribution: {c}")
        plt.xlabel(c)
        plt.ylabel("Density")
        plt.tight_layout()
        os.path.join(
            OUTDIR,
            f"box_{re.sub(r'[^0-9a-zA-Z_一-龥-]+', '_', str(c))}_by_efficacy.png" )
        plt.savefig(os.path.join(OUTDIR, f"dist_{re.sub('[^0-9a-zA-Z_一-龥-]+','_',c)}.png"), dpi=200)
        # plt.savefig(os.path.join(OUTDIR, f"dist_{re.sub('[^0-9a-zA-Z_\\-一-龥]+','_',c)}.png"), dpi=200)
        plt.close()

plot_feature_distributions(df, num_cols, by=LABEL_COL if LABEL_COL in df.columns else None, n=25)



plot_feature_distributions(
    df,
    cols=["PCT", "BA%"],
    by=LABEL_COL,
    n=2
)

#%%

dup = pd.Series(df.columns).duplicated().sum()
print("Duplicate column names count:", dup)

# 打印重复列名有哪些
dups = pd.Series(df.columns)[pd.Series(df.columns).duplicated()].value_counts()
print("Top duplicated names:\n", dups.head(20))

#%%

# -----------------------------
# 6) Correlation (pooled Spearman) + visualization
# -----------------------------
def spearman_corr(df, cols, min_pairwise_n=20, verbose=True):
    """
    Robust pairwise Spearman:
    - 强制每列是一维数值向量
    - 如果 df[col] 返回 DataFrame（重复列名）则跳过并提示
    - 如果 spearmanr 返回矩阵/数组（非标量）则跳过该对
    """
    cols = [c for c in cols if c in df.columns]

    corr = pd.DataFrame(np.nan, index=cols, columns=cols, dtype="float64")
    pval = pd.DataFrame(np.nan, index=cols, columns=cols, dtype="float64")
    nmat = pd.DataFrame(0, index=cols, columns=cols, dtype=int)

    # 预缓存：把每列都转成一维数值Series（避免循环里重复做）
    series_cache = {}
    bad_cols = []
    for c in cols:
        x = df[c]
        if isinstance(x, pd.DataFrame):
            bad_cols.append(c)
            continue
        series_cache[c] = pd.to_numeric(x, errors="coerce")

    good_cols = [c for c in cols if c in series_cache]

    skipped_pairs = 0

    for i, a in enumerate(good_cols):
        for j in range(i, len(good_cols)):
            b = good_cols[j]

            sub = pd.concat([series_cache[a], series_cache[b]], axis=1)
            sub.columns = ["a", "b"]
            sub = sub.dropna()

            n = len(sub)
            nmat.loc[a, b] = nmat.loc[b, a] = n
            if n < min_pairwise_n:
                continue

            a1 = sub["a"].to_numpy(dtype=float)
            b1 = sub["b"].to_numpy(dtype=float)

            r, p = stats.spearmanr(a1, b1)


            if (not np.isscalar(r)) or (not np.isscalar(p)) or (not np.isfinite(r)):
                skipped_pairs += 1
                continue

            corr.loc[a, b] = corr.loc[b, a] = float(r)
            pval.loc[a, b] = pval.loc[b, a] = float(p)

    for c in good_cols:
        corr.loc[c, c] = 1.0
        pval.loc[c, c] = 0.0

    if verbose:
        if bad_cols:
            print(f"[WARN] df[col] returned DataFrame (likely duplicate column names). Skipped cols (showing up to 20): {bad_cols[:20]}")
        if skipped_pairs:
            print(f"[WARN] Skipped {skipped_pairs} pairs where spearmanr returned non-scalar.")
        print(f"[INFO] Corr computed on {len(good_cols)} columns (requested {len(cols)}).")

    return corr, pval, nmat



corr_cols = miss_all[miss_all["missing_rate"] <= 0.5]["feature"].tolist()

corr_cols = [c for c in corr_cols if c in num_cols]

corr_all, p_all, n_all = spearman_corr(df, corr_cols, min_pairwise_n=25)
corr_all.to_csv(os.path.join(OUTDIR, "spearman_corr_overall.csv"))
p_all.to_csv(os.path.join(OUTDIR, "spearman_p_overall.csv"))
n_all.to_csv(os.path.join(OUTDIR, "spearman_n_overall.csv"))

def plot_corr_heatmap(corr, title, fname, vmax=1.0, vmin=-1.0):
    plt.figure(figsize=(18, 14))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=vmin, vmax=vmax,
                square=False, linewidths=0.2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, fname), dpi=220)
    plt.close()


def topk_related_features(df, corr, target, k=25):
    if target not in corr.columns:
        return corr.columns.tolist()[:k]
    s = corr[target].dropna().abs().sort_values(ascending=False)
    s = s.drop(index=[target], errors="ignore")
    return s.head(k).index.tolist()


targets = [c for c in [LABEL_COL, DOSE_COL, TIME_COL] if c in corr_all.columns]
if len(targets) > 0:
    for t in targets:
        topk = topk_related_features(df, corr_all, t, k=30)
        subcorr = corr_all.loc[topk, topk]
        plot_corr_heatmap(subcorr, f"Spearman Corr (focused around {t})", f"corr_focus_{t}.png")
else:
    sub = corr_all.iloc[:40, :40]
    plot_corr_heatmap(sub, "Spearman Corr (first 40 features)", "corr_first40.png")


if LABEL_COL in df.columns:
    for g, sub in df.groupby(LABEL_COL):
        corr_g, _, _ = spearman_corr(sub, corr_cols, min_pairwise_n=15)
        subcorr = corr_g.iloc[:40, :40]
        plot_corr_heatmap(subcorr, f"Spearman Corr (efficacy={g}) first40", f"corr_by_eff_{g}_first40.png")


#%%

# -----------------------------
# 7) Repeated measures: between vs within patient correlations
# -----------------------------


def patient_level_mean(df, cols):
    g = df.groupby(ID_COL)[cols].mean(numeric_only=True)
    return g

def within_patient_demean(df, cols):
    dfw = df[[ID_COL] + cols].copy()
    means = dfw.groupby(ID_COL)[cols].transform("mean")
    dfw[cols] = dfw[cols] - means
    return dfw


cols_for_rm = corr_cols.copy()


df_between = patient_level_mean(df, cols_for_rm)
corr_between = df_between.corr(method="spearman")
corr_between.to_csv(os.path.join(OUTDIR, "spearman_corr_between_patients.csv"))
plot_corr_heatmap(corr_between.iloc[:40,:40], "Between-patient Spearman Corr (first40)", "corr_between_first40.png")


df_within = within_patient_demean(df, cols_for_rm)
corr_within = df_within[cols_for_rm].corr(method="spearman")
corr_within.to_csv(os.path.join(OUTDIR, "spearman_corr_within_patients.csv"))
plot_corr_heatmap(corr_within.iloc[:40,:40], "Within-patient Spearman Corr (first40)", "corr_within_first40.png")


def meta_within_corr(df, x, y, min_n_per_patient=3):
    zs, ns = [], []
    for pid, sub in df[[ID_COL, x, y]].dropna().groupby(ID_COL):
        if len(sub) < min_n_per_patient:
            continue
        r, _ = stats.spearmanr(sub[x], sub[y])
        if np.isfinite(r) and abs(r) < 1:
            z = np.arctanh(r)
            zs.append(z)
            ns.append(len(sub))
    if len(zs) == 0:
        return np.nan, np.nan, 0
    # 权重：n-3
    w = np.maximum(np.array(ns) - 3, 1)
    zbar = np.average(zs, weights=w)
    rbar = np.tanh(zbar)
    # 粗略SE（固定效应近似）
    se = np.sqrt(1 / np.sum(w))
    return rbar, se, len(zs)


def top_within_pairs(df, cols, topn=20):
    pairs = []
    cols = cols[:30]
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            x, y = cols[i], cols[j]
            rbar, se, n_pat = meta_within_corr(df, x, y, min_n_per_patient=3)
            if np.isfinite(rbar) and n_pat >= 10:
                pairs.append((x, y, rbar, se, n_pat))
    pairs = pd.DataFrame(pairs, columns=["x", "y", "within_r_meta", "se_approx", "n_patients_used"])\
             .sort_values("within_r_meta", key=lambda s: s.abs(), ascending=False)\
             .head(topn)
    return pairs

within_pairs = top_within_pairs(df, cols_for_rm, topn=25)
within_pairs.to_csv(os.path.join(OUTDIR, "top_within_patient_dynamic_pairs.csv"), index=False)
print("\n[Top within-patient dynamic pairs]\n", within_pairs)


# 1) 派生指标家族
DERIVED_GROUPS = [
    {"RBC", "HGB", "HCT"},
    {"PLT", "PCT", "MPV", "P-LCR", "PDW"},
    {"RDW-CV", "RDW-SD"},
]


def base_name(x):
    return re.sub(r'[%#]', '', x)

def is_trivial_pair(x, y):
    # (a) % vs # 这种
    if base_name(x) == base_name(y):
        return True

    # (b) 同一派生指标家族
    for g in DERIVED_GROUPS:
        if x in g and y in g:
            return True
    return False

within_pairs["is_trivial"] = within_pairs.apply(
    lambda r: is_trivial_pair(r["x"], r["y"]),
    axis=1
)

# 只保留“非平凡”的
mechanistic_pairs = within_pairs[~within_pairs["is_trivial"]].copy()

mechanistic_pairs

mechanistic_pairs = mechanistic_pairs[
    (mechanistic_pairs["within_r_meta"].abs() >= 0.6) &
    (mechanistic_pairs["n_patients_used"] >= 20)
].sort_values(
    "within_r_meta", key=lambda s: s.abs(), ascending=False
)

mechanistic_pairs.to_csv(
    os.path.join(OUTDIR, "within_patient_mechanistic_pairs.csv"),
    index=False
)

print("\n[within_patient_mechanistic_pairs]\n", mechanistic_pairs)

#%%



eff_pat = df.groupby(ID_COL)[LABEL_COL].agg(
    lambda s: s.dropna().mode().iloc[0] if len(s.dropna().mode()) > 0 else np.nan
)
eff_pat = pd.to_numeric(eff_pat, errors="coerce")

def meta_within_corr_by_group(
    df, x, y, eff_pat, target_eff, min_n_per_patient=3
):
    """
    在 efficacy = target_eff 的患者中，计算 meta-within Spearman
    """
    zs, ns = [], []

    for pid, sub in df[[ID_COL, x, y]].dropna().groupby(ID_COL):
        if pid not in eff_pat.index or eff_pat.loc[pid] != target_eff:
            continue

        if len(sub) < min_n_per_patient:
            continue

        r, _ = stats.spearmanr(sub[x], sub[y])
        if np.isfinite(r) and abs(r) < 1:
            zs.append(np.arctanh(r))
            ns.append(len(sub))

    if len(zs) == 0:
        return np.nan, np.nan, 0

    w = np.maximum(np.array(ns) - 3, 1)
    zbar = np.average(zs, weights=w)
    rbar = np.tanh(zbar)
    se = np.sqrt(1 / np.sum(w))

    return rbar, se, len(zs)

rows = []

for _, row in mechanistic_pairs.iterrows():
    x, y = row["x"], row["y"]

    r1, se1, n1 = meta_within_corr_by_group(
        df, x, y, eff_pat, target_eff=1
    )
    r0, se0, n0 = meta_within_corr_by_group(
        df, x, y, eff_pat, target_eff=0
    )

    rows.append({
        "x": x,
        "y": y,
        "r_eff1": r1,
        "se_eff1": se1,
        "n_eff1": n1,
        "r_eff0": r0,
        "se_eff0": se0,
        "n_eff0": n0,
        "delta_r": r1 - r0
    })

eff_split_pairs = pd.DataFrame(rows)

eff_bias = eff_split_pairs[
    (eff_split_pairs["n_eff1"] >= 10) &
    (eff_split_pairs["n_eff0"] >= 10) &
    (eff_split_pairs["r_eff1"].abs() >= 0.6) &
    ((eff_split_pairs["r_eff1"] - eff_split_pairs["r_eff0"]).abs() >= 0.3)
].sort_values("delta_r", key=lambda s: s.abs(), ascending=False)


eff_bias.to_csv(
    os.path.join(OUTDIR, "meta_within_effective_specific_pairs.csv"),
    index=False
)

print("\n[Meta-within dynamic pairs specific to efficacy=1]\n")
print(eff_bias)


#画图

os.makedirs(OUTDIR, exist_ok=True)

# ====== 0) 准备患者层疗效标签（每个患者一个efficacy）======
eff_pat = df.groupby(ID_COL)[LABEL_COL].agg(
    lambda s: s.dropna().mode().iloc[0] if len(s.dropna().mode()) > 0 else np.nan
)
eff_pat = pd.to_numeric(eff_pat, errors="coerce")

# 只取0/1患者
eff_pat = eff_pat[eff_pat.isin([0, 1])].copy()

# 把患者层efficacy映射回行（每一行带上该患者的疗效）
df_plot = df.copy()
df_plot[LABEL_COL] = df_plot[ID_COL].map(eff_pat)

# 只保留0/1疗效、且有时间的行
df_plot = df_plot[df_plot[LABEL_COL].isin([0, 1])].copy()
df_plot = df_plot.dropna(subset=[TIME_COL])

# 要画的两个指标
X1, X2 = "PCT", "BA%"


def spaghetti_by_group(df_plot, feature, fname_prefix):
    for g in [0, 1]:
        subg = df_plot[df_plot[LABEL_COL] == g].copy()
        subg = subg.dropna(subset=[feature])

   
        counts = subg.groupby(ID_COL)[feature].apply(lambda s: s.dropna().shape[0])
        keep_ids = counts[counts >= 2].index
        subg = subg[subg[ID_COL].isin(keep_ids)].copy()

        if subg.empty:
            print(f"[WARN] No data for {feature} in efficacy={g}")
            continue

        plt.figure(figsize=(9, 5))

        for pid, s in subg.sort_values(TIME_COL).groupby(ID_COL):
            plt.plot(s[TIME_COL].values, s[feature].values, alpha=0.25, linewidth=1)

        mean_curve = subg.groupby(TIME_COL)[feature].mean().sort_index()
        plt.plot(mean_curve.index.values, mean_curve.values, linewidth=3, alpha=0.9, label="Group mean")

        plt.title(f"{feature} trajectory (spaghetti) | efficacy={g}")
        plt.xlabel(TIME_COL)
        plt.ylabel(feature)
        plt.legend()
        plt.tight_layout()

        outpath = os.path.join(OUTDIR, f"{fname_prefix}_{feature}_eff{g}.png")
        plt.savefig(outpath, dpi=220)
        plt.close()
        print("[Saved]", outpath)

spaghetti_by_group(df_plot, X1, "traj_spaghetti")
spaghetti_by_group(df_plot, X2, "traj_spaghetti")


def joint_trajectory(df_plot, x_feat, y_feat, fname):
    for g in [0, 1]:
        subg = df_plot[df_plot[LABEL_COL] == g].copy()
        subg = subg.dropna(subset=[x_feat, y_feat, TIME_COL])


        counts = subg.groupby(ID_COL)[[x_feat, y_feat]].apply(lambda d: d.dropna().shape[0])
        keep_ids = counts[counts >= 2].index
        subg = subg[subg[ID_COL].isin(keep_ids)].copy()

        if subg.empty:
            print(f"[WARN] No joint data for {x_feat}-{y_feat} in efficacy={g}")
            continue

        plt.figure(figsize=(6.5, 6.0))

        for pid, s in subg.sort_values(TIME_COL).groupby(ID_COL):
            plt.plot(
                s[x_feat].values, s[y_feat].values,
                alpha=0.30, linewidth=1
            )

            plt.scatter(s[x_feat].values[0], s[y_feat].values[0], s=12, alpha=0.35)

        plt.title(f"Joint within-patient trajectory: {x_feat} vs {y_feat} | efficacy={g}")
        plt.xlabel(x_feat)
        plt.ylabel(y_feat)
        plt.tight_layout()

        outpath = os.path.join(OUTDIR, f"{fname}_eff{g}.png")
        plt.savefig(outpath, dpi=220)
        plt.close()
        print("[Saved]", outpath)

joint_trajectory(df_plot, X1, X2, "traj_joint_PCT_BA")

print("DONE. Figures saved to:", os.path.abspath(OUTDIR))


# -----------------------------
# 9) Simple “feature vs efficacy” association (report-friendly)
# -----------------------------
if LABEL_COL in df.columns:

    # 0) 确保 cols_for_rm 是干净的字符串列名列表，并且都存在于 df
    cols_for_rm = [str(c).strip() for c in cols_for_rm]
    cols_for_rm = [c for c in cols_for_rm if c in df.columns and c != LABEL_COL]

    # 1) 患者层特征均值（避免同一患者多次记录造成“样本量虚高”）
    df_pat = df.groupby(ID_COL)[cols_for_rm].mean(numeric_only=True)

    # 2) 患者层疗效：取众数（若全缺失则NaN）
    eff_pat = df.groupby(ID_COL)[LABEL_COL].agg(
        lambda s: s.dropna().mode().iloc[0] if len(s.dropna().mode()) > 0 else np.nan
    )
    eff_pat = pd.to_numeric(eff_pat, errors="coerce")

    # 3) 合并疗效（用 join，避免列名冲突导致 df_pat[LABEL_COL] 变 DataFrame）
    df_pat = df_pat.join(eff_pat.rename(LABEL_COL), how="left")

    # 4) 只保留疗效为 0/1 的患者
    df_pat = df_pat[df_pat[LABEL_COL].isin([0, 1])].copy()

    print("[INFO] Patients used after efficacy filter:", df_pat.shape[0])
    print("[INFO] Efficacy counts:\n", df_pat[LABEL_COL].value_counts(dropna=False))

    assoc = []
    skipped_too_few = 0
    skipped_nonscalar = 0

    for c in cols_for_rm:
        if c not in df_pat.columns:
            continue

        # 永远用一维 Series 取列，避免 [[c, LABEL_COL]] 触发二维选择
        x = df_pat[c]
        y = df_pat[LABEL_COL]

        if isinstance(x, pd.DataFrame):
            x = x.iloc[:, 0]
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]

        x = pd.to_numeric(x, errors="coerce")
        y = pd.to_numeric(y, errors="coerce")

        sub = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()

        if len(sub) < 20:
            skipped_too_few += 1
            continue

        r, p = stats.spearmanr(sub["x"].to_numpy(dtype=float), sub["y"].to_numpy(dtype=float))

        # 必须是标量
        if (not np.isscalar(r)) or (not np.isscalar(p)) or (not np.isfinite(r)):
            skipped_nonscalar += 1
            continue

        assoc.append((c, float(r), float(p), len(sub)))

    assoc = pd.DataFrame(
        assoc, columns=["feature", "spearman_r_vs_efficacy", "p_value", "n_patients_used"]
    )

    print("[INFO] Skipped (too few patients):", skipped_too_few)
    print("[INFO] Skipped (non-scalar spearman):", skipped_nonscalar)

    if assoc.empty:
        print("[WARN] No features produced valid scalar Spearman results.")
        print("Common reasons:")
        print(" - After efficacy filter, patients < 20")
        print(" - Many features have < 20 non-missing at patient-level")
        print(" - Efficacy not actually coded as 0/1")
    else:
        # 稳定排序：先新增abs列再排序
        assoc["abs_r"] = assoc["spearman_r_vs_efficacy"].abs()
        assoc = assoc.sort_values(["abs_r", "p_value"], ascending=[False, True]).drop(columns=["abs_r"])

        assoc.to_csv(os.path.join(OUTDIR, "feature_assoc_vs_efficacy_patient_mean.csv"), index=False)
        print("[Saved] feature_assoc_vs_efficacy_patient_mean.csv")

        # 可视化：前15个最相关特征（箱线图）
        top15 = assoc.head(15)["feature"].tolist()
        for feat in top15:
            plt.figure(figsize=(6, 4))
            sns.boxplot(data=df_pat, x=LABEL_COL, y=feat)
            sns.stripplot(data=df_pat, x=LABEL_COL, y=feat, color="black", size=3, alpha=0.6)
            plt.title(f"{feat} by efficacy (patient-mean)")
            plt.tight_layout()
            os.path.join(OUTDIR, f"box_{re.sub('[^0-9a-zA-Z_一-龥-]+', '_', feat)}_by_efficacy.png"),
            plt.savefig(
                os.path.join(
                    OUTDIR,
                    f"box_{re.sub(r'[^0-9a-zA-Z_一-龥-]+', '_', str(feat))}_by_efficacy.png"
                ),

            #os.path.join(OUTDIR, f"box_{re.sub('[^0-9a-zA-Z_\\-一-龥]+','_',feat)}_by_efficacy.png"),
                dpi=200
            )
            plt.close()

print("\nDONE. All outputs saved to:", os.path.abspath(OUTDIR))


