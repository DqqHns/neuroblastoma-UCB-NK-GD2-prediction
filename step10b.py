# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import kendalltau

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone


# =============== 配置区（你只需要改这里） ===============
@dataclass
class Config:
    # 输入
    OOF_CSV: str = r"G:/Dqq_code/nk_model(1.8)/compare_methods_results_new_COMPLETE_CASE = True/oof_pred_mean.csv"
    OOF_ID_COL: str = "patient_id"
    OOF_PRED_COL: str = "M14_LogReg_L2"   # 你说的那一列
    # 如果 OOF 里也有真实标签列，就填；如果没有则留 None（脚本会尝试自动猜）
    OOF_TARGET_COL: Optional[str] = None

    FEATURES_XLSX: str = r"G:/Dqq_code/nk_model(1.8)/data/DATA.xlsx"
    FEATURES_SHEET: str = "patient_features"
    FEATURES_ID_COL: str = "patient_id"

    # 训练好的 pipeline（用于读取超参/结构）
    MODEL_JOBLIB: str = r"G:/Dqq_code/nk_model(1.8)/save_results/M14_M13+SAA__final_pipeline.joblib"

    # M14 6个特征（顺序也建议固定）
    FEATURE_COLS: List[str] = None

    # bootstrap 设置
    N_BOOTSTRAP: int = 2000
    RANDOM_SEED: int = 42
    # Top-K 稳定性统计
    TOPK_LIST: Tuple[int, ...] = (1, 2, 3)

    # 输出
    OUT_DIR: str = "step12b_out"


def _default_feature_cols() -> List[str]:
    return ["dyn_same_dir_rate", "RDW-SD_baseline", "dyn_slope_BA_on_PCT", "HGB_mean", "SAA_mean", "Na+_mean"]


# =============== 工具函数 ===============
def _log(msg: str) -> None:
    print(msg, flush=True)


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _guess_target_col(df: pd.DataFrame) -> Optional[str]:
    # 常见 label 名
    candidates = ["label", "y", "target", "outcome", "Y", "Label", "Target", "OUTCOME", "case", "Case"]
    for c in candidates:
        if c in df.columns:
            return c
    # 也可能是 0/1 的一列
    for c in df.columns:
        if c.lower().endswith(("label", "target", "outcome")):
            return c
    return None


def _load_oof(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.OOF_CSV)
    if cfg.OOF_ID_COL not in df.columns:
        raise ValueError(f"OOF 缺少 id 列: {cfg.OOF_ID_COL}")
    if cfg.OOF_PRED_COL not in df.columns:
        raise ValueError(f"OOF 缺少预测列: {cfg.OOF_PRED_COL}，现有列: {list(df.columns)[:30]} ...")

    # 尝试确定 target 列
    target_col = cfg.OOF_TARGET_COL
    if target_col is None:
        target_col = _guess_target_col(df)
    cfg.OOF_TARGET_COL = target_col  # 写回，后续使用

    use_cols = [cfg.OOF_ID_COL, cfg.OOF_PRED_COL]
    if target_col is not None and target_col in df.columns:
        use_cols.append(target_col)

    df = df[use_cols].copy()
    df = df.dropna(subset=[cfg.OOF_ID_COL, cfg.OOF_PRED_COL])
    return df


def _load_features(cfg: Config) -> pd.DataFrame:
    df = pd.read_excel(cfg.FEATURES_XLSX, sheet_name=cfg.FEATURES_SHEET)
    if cfg.FEATURES_ID_COL not in df.columns:
        raise ValueError(f"feature 表缺少 id 列: {cfg.FEATURES_ID_COL}")
    return df


def _prepare_merged(cfg: Config) -> Tuple[pd.DataFrame, List[str]]:
    df_oof = _load_oof(cfg)
    df_feat = _load_features(cfg)

    feature_cols = cfg.FEATURE_COLS or _default_feature_cols()
    # 检查特征是否齐全
    missing = [c for c in feature_cols if c not in df_feat.columns]
    if missing:
        raise ValueError(f"patient_features 缺少这些特征列: {missing}")

    keep_feat = [cfg.FEATURES_ID_COL] + feature_cols
    df_feat = df_feat[keep_feat].copy()

    df = df_oof.merge(df_feat, left_on=cfg.OOF_ID_COL, right_on=cfg.FEATURES_ID_COL, how="inner")
    if df.empty:
        raise ValueError("OOF 与 feature 表 merge 后为空，请检查 patient_id 是否一致。")


    before = len(df)
    df = df.dropna(subset=feature_cols)
    dropped = before - len(df)

    _log(f"  - merged rows: {before} ; dropped rows with NA in features: {dropped} ; remaining: {len(df)}")

    return df, feature_cols


def _extract_abs_coef_from_pipeline(model: Pipeline, feature_cols: List[str]) -> pd.Series:
    # 找到最后的 LR
    if isinstance(model, Pipeline):
        # 常见命名：('scaler','clf')
        lr = None
        for step in model.named_steps.values():
            if hasattr(step, "coef_"):
                lr = step
        if lr is None:
            raise ValueError("Pipeline 中找不到带 coef_ 的 estimator（需要 LogisticRegression）。")
    else:
        lr = model

    coef = np.asarray(lr.coef_).ravel()
    if len(coef) != len(feature_cols):
        raise ValueError(f"coef length {len(coef)} != n_features {len(feature_cols)}；"
                         f"请确认 cfg.FEATURE_COLS 与训练时一致。")
    return pd.Series(np.abs(coef), index=feature_cols)


def _build_bootstrap_model_from_fitted(model: Pipeline) -> Pipeline:

    if not isinstance(model, Pipeline):
        raise ValueError("期望 joblib 载入的是 sklearn Pipeline。")

    return clone(model)


def _rank_from_importance(imp: pd.Series) -> pd.Series:
    # rank 1 = 最大重要性
    return imp.rank(ascending=False, method="average").astype(float)



def main():
    cfg = Config()
    if cfg.FEATURE_COLS is None:
        cfg.FEATURE_COLS = _default_feature_cols()

    out_dir = Path(cfg.OUT_DIR)
    plots_dir = out_dir / "plots"
    _safe_mkdir(out_dir)
    _safe_mkdir(plots_dir)

    _log("[1/7] Loading + merging data ...")
    df, feature_cols = _prepare_merged(cfg)

 
    if cfg.OOF_TARGET_COL is None or cfg.OOF_TARGET_COL not in df.columns:
        raise ValueError(
            "OOF 里没有找到真实标签列（target）。\n"
            "Bootstrap 重训需要 y。\n"
            "请在 oof_pred_mean.csv 增加一列真实标签（0/1），或在 Config 里指定 OOF_TARGET_COL。"
        )

    y = df[cfg.OOF_TARGET_COL].astype(int).values
    X = df[feature_cols].astype(float).values

    _log("[2/7] Loading fitted model ...")
    fitted_model = joblib.load(cfg.MODEL_JOBLIB)
    _log(f"  - loaded model: {cfg.MODEL_JOBLIB}")

    _log("[3/7] Baseline importance (|coef| from fitted model) ...")
    base_imp = _extract_abs_coef_from_pipeline(fitted_model, feature_cols)
    base_rank = _rank_from_importance(base_imp)

    base_imp.to_csv(out_dir / "baseline_abscoef_importance.csv", header=["abs_coef"])
    base_rank.to_csv(out_dir / "baseline_rank.csv", header=["rank"])

    _log("[4/7] Bootstrap refit + importance ...")
    rng = np.random.default_rng(cfg.RANDOM_SEED)

    boot_model_template = _build_bootstrap_model_from_fitted(fitted_model)

    imp_rows = []
    rank_rows = []
    tau_rows = []

    n = X.shape[0]
    for b in range(cfg.N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)  # sample with replacement
        Xb = X[idx]
        yb = y[idx]

        m = clone(boot_model_template)
        try:
            m.fit(Xb, yb)
            imp = _extract_abs_coef_from_pipeline(m, feature_cols)
        except Exception as e:
    
            tau_rows.append({"b": b, "kendall_tau": np.nan, "p_value": np.nan, "status": f"fail:{type(e).__name__}"})
            continue

        ranks = _rank_from_importance(imp)

        tau, p = kendalltau(base_rank.values, ranks.values)

        imp_rows.append({"b": b, **imp.to_dict()})
        rank_rows.append({"b": b, **ranks.to_dict()})
        tau_rows.append({"b": b, "kendall_tau": float(tau) if tau is not None else np.nan,
                         "p_value": float(p) if p is not None else np.nan,
                         "status": "ok"})

        if (b + 1) % 200 == 0:
            _log(f"  - done {b+1}/{cfg.N_BOOTSTRAP}")

    df_imp = pd.DataFrame(imp_rows)
    df_rank = pd.DataFrame(rank_rows)
    df_tau = pd.DataFrame(tau_rows)

    df_imp.to_csv(out_dir / "bootstrap_importance.csv", index=False)
    df_rank.to_csv(out_dir / "bootstrap_rank.csv", index=False)
    df_tau.to_csv(out_dir / "kendall_tau_per_bootstrap.csv", index=False)

    _log("[5/7] Rank stability summary ...")

    df_rank_long = df_rank.melt(id_vars=["b"], var_name="feature", value_name="rank")
    summary = []
    for f in feature_cols:
        r = df_rank_long.loc[df_rank_long["feature"] == f, "rank"].dropna()
        if r.empty:
            continue

        mode_rank = r.value_counts().idxmax()
        summary.append({
            "feature": f,
            "baseline_rank": float(base_rank[f]),
            "rank_mean": float(r.mean()),
            "rank_std": float(r.std(ddof=0)),
            "rank_mode": float(mode_rank),
            "rank_mode_freq": float((r == mode_rank).mean()),
        })
        # TopK 频率
        for k in cfg.TOPK_LIST:
            summary[-1][f"freq_top{k}"] = float((r <= k).mean())

    df_summary = pd.DataFrame(summary).sort_values("baseline_rank")
    df_summary.to_csv(out_dir / "rank_stability_summary.csv", index=False)

    _log("[6/7] Plots ...")

    show_n = min(500, len(df_rank))
    if show_n > 10:
        df_rank_mat = df_rank.iloc[:show_n].set_index("b")[feature_cols].T
        plt.figure(figsize=(10, 3 + 0.35 * len(feature_cols)))
        plt.imshow(df_rank_mat.values, aspect="auto")
        plt.yticks(range(len(feature_cols)), feature_cols)
        plt.xlabel("bootstrap sample (first N)")
        plt.colorbar(label="rank (1=most important)")
        plt.title("Bootstrap rank heatmap (first N samples)")
        plt.tight_layout()
        plt.savefig(plots_dir / "rank_heatmap.png", dpi=300)
        plt.close()

    # 2) kendall tau hist
    taus = df_tau.loc[df_tau["status"] == "ok", "kendall_tau"].dropna().values
    if len(taus) > 5:
        plt.figure(figsize=(6, 4))
        plt.hist(taus, bins=30)
        plt.xlabel("Kendall tau vs baseline rank")
        plt.ylabel("count")
        plt.title("Rank stability (Kendall tau)")
        plt.tight_layout()
        plt.savefig(plots_dir / "kendall_tau_hist.png", dpi=300)
        plt.close()

    # 3) topK frequency bar
    if not df_summary.empty:
        for k in cfg.TOPK_LIST:
            col = f"freq_top{k}"
            if col not in df_summary.columns:
                continue
            plt.figure(figsize=(7, 3.5))
            plt.bar(df_summary["feature"], df_summary[col].values)
            plt.xticks(rotation=45, ha="right")
            plt.ylim(0, 1.0)
            plt.ylabel("frequency")
            plt.title(f"Bootstrap frequency of being in Top-{k}")
            plt.tight_layout()
            plt.savefig(plots_dir / f"freq_top{k}.png", dpi=300)
            plt.close()

    _log("[7/7] Done.")
    _log(f"Outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
