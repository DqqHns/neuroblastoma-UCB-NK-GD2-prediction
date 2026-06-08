# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import step3_build_models as t5   # test5.py must be importable (same folder)
import step3_build_models as t5

# -----------------------------
# Config
# -----------------------------
XLSX_PATH = "data/DATA.xlsx"
OUTDIR = "permutation_test_results"
os.makedirs(OUTDIR, exist_ok=True)

MODEL_NAME = "M14_M13+SAA"   # change if needed
N_PERM_RUNS = 1000               # number of independent permutation runs
PERM_BASE_SEED = 2026         # seed for label permutation (NOT CV seed)


# -----------------------------
# Helpers
# -----------------------------
def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()


def summarize_auc(rep_df: pd.DataFrame, scenario: str) -> dict:   # 输入：重复实验结果表 `rep_df`（里面应有 `auc_mean` 列）
    x = rep_df["auc_mean"].astype(float).values
    x = x[np.isfinite(x)]

    def q(p):
        return float(np.quantile(x, p)) if len(x) else np.nan

    return {
        "scenario": scenario,
        "n_repeats": int(len(x)),
        "auc_mean": float(np.mean(x)) if len(x) else np.nan,
        "auc_sd": float(np.std(x)) if len(x) else np.nan,
        "auc_p10": q(0.10),
        "auc_p50": q(0.50),
        "auc_p90": q(0.90),
    }




def plot_violin(rep_all: pd.DataFrame, out_png: str):
    # 固定顺序更直观（也避免 unique() 顺序不稳定）
    scenarios = [s for s in ["REAL", "PERMUTED"] if s in rep_all["scenario"].unique()]

    plt.figure(figsize=(10, 5))
    ax = plt.gca()

    labels = []
    positions = []

    for i, s in enumerate(scenarios, start=1):
        vals = pd.to_numeric(
            rep_all.loc[rep_all["scenario"] == s, "auc_mean"],
            errors="coerce"
        ).dropna().to_numpy(float)


        if len(vals) < 2:
            print(f"[WARN] Violin skip '{s}': n={len(vals)} (need >=2)")
            continue

        ax.violinplot(
            vals, positions=[i],
            showmeans=True, showmedians=True
        )
        labels.append(s)
        positions.append(i)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUC (repeat mean)")
    ax.set_title("Sanity check: REAL vs PERMUTED labels")
    savefig(out_png)




def plot_trace(rep_all: pd.DataFrame, out_png: str):
    plt.figure(figsize=(10, 5))
    for s in rep_all["scenario"].unique():
        d = rep_all[rep_all["scenario"] == s].sort_values("repeat")
        # 确保传入matplotlib.plot的是numpy数组
        plt.plot(d["repeat"].to_numpy(), d["auc_mean"].to_numpy(), label=s, linewidth=1.5)
#        plt.plot(d["repeat"], d["auc_mean"], label=s, linewidth=1.5)

    plt.xlabel("Repeat")
    plt.ylabel("AUC (repeat mean)")
    plt.title("Repeat-wise AUC trace")
    plt.legend()
    savefig(out_png)


# -----------------------------
# Main
# -----------------------------
def main():
    # 1) Load data
    sheet = t5.auto_pick_sheet(XLSX_PATH)
    df = pd.read_excel(XLSX_PATH, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    df = t5.add_all_interactions(df)

    if MODEL_NAME not in t5.MODELS:
        raise KeyError(f"{MODEL_NAME} not found in test5.MODELS")

    feats = t5.MODELS[MODEL_NAME]

    # 2) Prepare identical subset
    df_real = t5.prepare_subset_for_model(df, feats, MODEL_NAME)
    n = len(df_real)
    pos_rate = df_real[t5.LABEL_COL].mean()

    print(f"[INFO] Subset n={n}, positive rate={pos_rate:.3f}")

    # -----------------------------
    # REAL labels
    # -----------------------------
    print("\n" + "="*80)
    print("[RUN] REAL labels")

    _, rep_real, _, _, _ = t5.run_cv50(
        df_real, feats, model_name=f"{MODEL_NAME}_REAL"
    )
    rep_real["scenario"] = "REAL"

    real_mean_auc = float(rep_real["auc_mean"].mean())

    # -----------------------------
    # PERMUTATION runs
    # -----------------------------
    rep_perm_all = []
    perm_run_means = []

    for k in range(1, N_PERM_RUNS + 1):
        seed = PERM_BASE_SEED + k
        rng = np.random.default_rng(seed)

        print("\n" + "="*80)
        print(f"[RUN] PERMUTED labels (run {k}/{N_PERM_RUNS}, seed={seed})")

        df_perm = df_real.copy()
        df_perm[t5.LABEL_COL] = rng.permutation(df_perm[t5.LABEL_COL].values)

        _, rep_perm, _, _, _ = t5.run_cv50(
            df_perm, feats, model_name=f"{MODEL_NAME}_PERM_{k}"
        )

        rep_perm["scenario"] = "PERMUTED"
        rep_perm["perm_run"] = k

        rep_perm_all.append(rep_perm)
        perm_run_means.append(float(rep_perm["auc_mean"].mean()))

    rep_perm_all = pd.concat(rep_perm_all, ignore_index=True)

    # -----------------------------
    # Empirical p-value
    # -----------------------------
    p_empirical = float(
        np.mean(np.array(perm_run_means) >= real_mean_auc)
    )

    # -----------------------------
    # Save summary
    # -----------------------------
    summary_rows = [
        summarize_auc(rep_real, "REAL"),
        summarize_auc(rep_perm_all, "PERMUTED"),
    ]

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUTDIR, "sanity_summary.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    # -----------------------------
    # Plots
    # -----------------------------
    rep_all = pd.concat([rep_real, rep_perm_all], ignore_index=True)

    violin_png = os.path.join(OUTDIR, "auc_violin_real_vs_perm.png")
    trace_png = os.path.join(OUTDIR, "auc_trace_real_vs_perm.png")

    plot_violin(rep_all, violin_png)
    plot_trace(rep_all, trace_png)

    # -----------------------------
    # Save p-value text
    # -----------------------------
    ptxt = os.path.join(OUTDIR, "empirical_pvalue_perm_ge_real.txt")
    with open(ptxt, "w", encoding="utf-8") as f:
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Features: {feats}\n")
        f.write(f"Subset n={n}, positive rate={pos_rate:.4f}\n\n")
        f.write(f"REAL mean AUC (CV50): {real_mean_auc:.6f}\n")
        f.write(f"PERMUTED run mean AUCs: {perm_run_means}\n\n")
        f.write("Empirical p-value:\n")
        f.write("P( mean AUC under permuted labels >= mean AUC under real labels )\n")
        f.write(f"= {p_empirical}\n")

    print("\n[Saved]")
    print(" -", summary_csv)
    print(" -", violin_png)
    print(" -", trace_png)
    print(" -", ptxt)

    print("\nDONE.")


if __name__ == "__main__":
    main()
