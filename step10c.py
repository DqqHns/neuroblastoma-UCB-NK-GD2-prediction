
#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")


# =========================
# CONFIG
# =========================
@dataclass
class Config:

    feature_xlsx: str = r"G:/Dqq_code/nk_model(1.8)/data/DATA.xlsx"
    feature_sheet: str = "patient_features"
    id_col: str = "patient_id"


    oof_csv: str = r"G:/Dqq_code/nk_model(1.8)/compare_methods_results_new_COMPLETE_CASE = True/oof_pred_mean.csv"


    model_joblib: str = r"G:/Dqq_code/nk_model(1.8)/save_results/M14_M13+SAA__final_pipeline.joblib"

   
    out_dir: Optional[str] = None

    # bootstrap 次数
    n_bootstrap: int = 2000
    random_seed: int = 42


    pred_col: str = "M14_LogReg_L2"


    y_candidates: Tuple[str, ...] = ("y_true", "label", "target", "outcome", "y")


    default_feature_cols: Tuple[str, ...] = (
        "HGB_mean",
        "RDW-SD_baseline",
        "dyn_same_dir_rate",
        "dyn_slope_BA_on_PCT",
        "SAA_mean",
        "Na+_mean",
    )


    palette: Dict[str, str] = None


CFG = Config()
CFG.palette = {
    # 森林图
    "point": "#1f77b4",       # 点
    "ci": "#1f77b4",          # 置信区间线
    "ref": "#444444",         # OR=1 参考线
    "grid": "#DDDDDD",        # 网格线
    "title": "#111111",       # 标题
}


# =========================
# 工具函数
# =========================
def _ensure_out_dir(cfg: Config) -> str:
    if cfg.out_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(script_dir, "step12c_outputs")
    else:
        out_dir = cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _read_oof(cfg: Config) -> pd.DataFrame:
    print("[1/6] Loading OOF CSV ...")
    df = pd.read_csv(cfg.oof_csv)

    # 统一 id 列名（容错）
    if cfg.id_col not in df.columns:
        for alt in ("patient_id", "pid", "id", "PatientID", "PATIENT_ID"):
            if alt in df.columns:
                df = df.rename(columns={alt: cfg.id_col})
                break
    if cfg.id_col not in df.columns:
        raise ValueError(f"OOF CSV lacks id column '{cfg.id_col}'. Columns: {list(df.columns)}")

    # 找 y 列
    y_col = None
    for c in cfg.y_candidates:
        if c in df.columns:
            y_col = c
            break
    if y_col is None:
        # 兜底：找一个看起来是二分类的列（0/1）
        for c in df.columns:
            if c == cfg.id_col:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                vals = pd.Series(df[c].dropna().unique())
                if len(vals) <= 3 and set(vals.tolist()).issubset({0, 1}):
                    y_col = c
                    break
    if y_col is None:
        raise ValueError(
            "Cannot find ground-truth label column in OOF CSV. "
            f"Tried {cfg.y_candidates}. Please add/rename your label column."
        )

    keep = [cfg.id_col, y_col]
    if cfg.pred_col in df.columns:
        keep.append(cfg.pred_col)
    df = df[keep].copy()

    df = df.dropna(subset=[cfg.id_col, y_col])
    df[cfg.id_col] = df[cfg.id_col].astype(str)
    df = df.rename(columns={y_col: "y"})
    df["y"] = df["y"].astype(int)

    print(f"  - OOF rows: {len(df)} ; label column detected as '{y_col}'")
    return df


def _read_features(cfg: Config) -> pd.DataFrame:
    print("[2/6] Loading feature table ...")
    feat = pd.read_excel(cfg.feature_xlsx, sheet_name=cfg.feature_sheet)
    if cfg.id_col not in feat.columns:
        raise ValueError(f"Feature sheet lacks id column '{cfg.id_col}'. Columns: {list(feat.columns)}")
    feat = feat.copy()
    feat[cfg.id_col] = feat[cfg.id_col].astype(str)
    return feat


def _load_model(cfg: Config):
    print("[3/6] Loading trained model ...")
    model = joblib.load(cfg.model_joblib)
    print(f"  - Loaded: {cfg.model_joblib}")
    return model


def _infer_model_n_features(model) -> Optional[int]:
    try:
        if isinstance(model, Pipeline):
            for _, step in model.named_steps.items():
                if hasattr(step, "n_features_in_"):
                    return int(step.n_features_in_)
        if hasattr(model, "n_features_in_"):
            return int(model.n_features_in_)
    except Exception:
        pass
    return None


def _select_feature_cols(cfg: Config, model, df: pd.DataFrame) -> List[str]:
    n_expected = _infer_model_n_features(model)

    numeric_cols = [
        c for c in df.columns
        if c not in (cfg.id_col, "y") and pd.api.types.is_numeric_dtype(df[c])
    ]

    if n_expected is None:
        cols = [c for c in cfg.default_feature_cols if c in df.columns]
        if len(cols) >= 2:
            return cols
        return numeric_cols

    default_present = [c for c in cfg.default_feature_cols if c in df.columns]

    if n_expected == len(cfg.default_feature_cols) and all(c in df.columns for c in cfg.default_feature_cols):
        return list(cfg.default_feature_cols)

    if len(numeric_cols) == n_expected:
        return numeric_cols

    if len(default_present) == n_expected:
        ordered = [c for c in cfg.default_feature_cols if c in default_present]
        return ordered

    raise ValueError(
        f"Cannot align features to model. Model expects n_features={n_expected}, "
        f"but numeric_cols={len(numeric_cols)}, default_present={len(default_present)}.\n"
        f"Edit CFG.default_feature_cols to match your final model, or ensure patient_features columns match."
    )


def _extract_pipeline_lr(model) -> LogisticRegression:
    if isinstance(model, Pipeline):
        if "clf" in model.named_steps and isinstance(model.named_steps["clf"], LogisticRegression):
            return model.named_steps["clf"]
        for _, step in model.named_steps.items():
            if isinstance(step, LogisticRegression):
                return step
    if isinstance(model, LogisticRegression):
        return model
    raise TypeError(f"Loaded model is not a LogisticRegression or Pipeline with LogisticRegression. Got: {type(model)}")


def _baseline_or_table(model, feature_cols: List[str]) -> pd.DataFrame:
    clf = _extract_pipeline_lr(model)
    coef = clf.coef_.ravel()
    if len(coef) != len(feature_cols):
        raise ValueError(f"coef length {len(coef)} != n_features {len(feature_cols)}")

    df = pd.DataFrame({"feature": feature_cols, "coef": coef, "OR": np.exp(coef)})
    df["direction"] = np.where(df["coef"] >= 0, "risk↑ (positive)", "risk↓ (negative)")
    return df.sort_values("OR", ascending=False)


def _fit_lr_on_sample(X: np.ndarray, y: np.ndarray, seed: int) -> Pipeline:
    clf = LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(X, y)
    return pipe


def _bootstrap_coefs(X: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    n = len(y)
    coefs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        Xb = X[idx]
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        m = _fit_lr_on_sample(Xb, yb, seed=rng.randint(0, 10**9))
        c = _extract_pipeline_lr(m).coef_.ravel()
        coefs.append(c)
    return np.asarray(coefs)


def _summarize_bootstrap(coefs: np.ndarray, feature_cols: List[str]) -> pd.DataFrame:
    lo = np.percentile(coefs, 2.5, axis=0)
    hi = np.percentile(coefs, 97.5, axis=0)
    med = np.percentile(coefs, 50, axis=0)

    out = pd.DataFrame({
        "feature": feature_cols,
        "coef_median": med,
        "coef_ci2.5": lo,
        "coef_ci97.5": hi,
        "OR_median": np.exp(med),
        "OR_ci2.5": np.exp(lo),
        "OR_ci97.5": np.exp(hi),
        "pos_rate(coef>0)": (coefs > 0).mean(axis=0),
    })
    out["direction(median)"] = np.where(out["coef_median"] >= 0, "risk↑ (positive)", "risk↓ (negative)")
    out["coef_CI_cross_0"] = (out["coef_ci2.5"] <= 0) & (out["coef_ci97.5"] >= 0)
    return out.sort_values("OR_median", ascending=False)


def _save_excel(csv_path: str, xlsx_path: str):
    df = pd.read_csv(csv_path)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="OR_bootstrap", index=False)


def _forest_plot(df_or: pd.DataFrame, out_png: str, cfg: Config):
    plot_df = df_or.copy().reset_index(drop=True)
    y_pos = np.arange(len(plot_df))[::-1]

    fig_h = max(3.8, 0.55 * len(plot_df) + 1.2)
    fig, ax = plt.subplots(figsize=(9.5, fig_h))

    or_mid = plot_df["OR_median"].values
    or_lo = plot_df["OR_ci2.5"].values
    or_hi = plot_df["OR_ci97.5"].values

    ax.errorbar(
        or_mid, y_pos,
        xerr=[or_mid - or_lo, or_hi - or_mid],
        fmt="o",
        capsize=3,
        elinewidth=2,
        markersize=5,
        color=cfg.palette["point"],
        ecolor=cfg.palette["ci"],
    )

    ax.axvline(1.0, linestyle="--", linewidth=1.5, color=cfg.palette["ref"])
    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["feature"].values)
    ax.set_xlabel("Odds Ratio (log scale)")
    ax.set_title("Bootstrap OR (median) with 95% CI", color=cfg.palette["title"])
    ax.grid(True, axis="x", linestyle=":", linewidth=0.8, color=cfg.palette["grid"])

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def main():
    out_dir = _ensure_out_dir(CFG)

    oof = _read_oof(CFG)
    feat = _read_features(CFG)
    model = _load_model(CFG)

    print("[4/6] Merge OOF with features ...")
    df = oof.merge(feat, on=CFG.id_col, how="inner")
    print(f"  - merged rows: {len(df)}")

    feature_cols = _select_feature_cols(CFG, model, df)
    print(f"  - Using {len(feature_cols)} feature(s): {feature_cols}")

    before = len(df)
    df2 = df.dropna(subset=feature_cols + ["y"]).copy()
    dropped = before - len(df2)
    if dropped > 0:
        print(f"  - Dropped rows with NA in features/label: {dropped} ; remaining: {len(df2)}")
    else:
        print(f"  - No NA dropped; remaining: {len(df2)}")

    X = df2[feature_cols].values.astype(float)
    y = df2["y"].values.astype(int)

    print("[5/6] Baseline OR (from loaded fitted model) ...")
    base = _baseline_or_table(model, feature_cols)
    base_csv = os.path.join(out_dir, "baseline_coef_OR.csv")
    base.to_csv(base_csv, index=False, encoding="utf-8-sig")
    _save_excel(base_csv, os.path.join(out_dir, "baseline_coef_OR.xlsx"))

    print("[6/6] Bootstrap coefficients + CI ...")
    coefs = _bootstrap_coefs(X, y, n_boot=CFG.n_bootstrap, seed=CFG.random_seed)
    print(f"  - Effective bootstrap fits: {len(coefs)} (single-class samples are skipped)")

    boot = _summarize_bootstrap(coefs, feature_cols)
    boot_csv = os.path.join(out_dir, "bootstrap_coef_OR_CI.csv")
    boot.to_csv(boot_csv, index=False, encoding="utf-8-sig")
    _save_excel(boot_csv, os.path.join(out_dir, "bootstrap_coef_OR_CI.xlsx"))

    forest_png = os.path.join(out_dir, "forestplot_OR_bootstrap.png")
    _forest_plot(boot, forest_png, CFG)

    np.save(os.path.join(out_dir, "bootstrap_coefs.npy"), coefs)

    print("\n[Done] Outputs saved to:", out_dir)
    print("  - baseline_coef_OR.csv / .xlsx")
    print("  - bootstrap_coef_OR_CI.csv / .xlsx")
    print("  - forestplot_OR_bootstrap.png")
    print("  - bootstrap_coefs.npy")


if __name__ == "__main__":
    main()
