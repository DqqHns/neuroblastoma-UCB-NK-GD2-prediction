# -*- coding: utf-8 -*-



import os
import numpy as np
import pandas as pd
import joblib

import step3_build_models as t5

from typing import List



# -----------------------------
# Config 
# -----------------------------
XLSX_PATH = "data/DATA.xlsx"
OUTDIR = "save_results"
os.makedirs(OUTDIR, exist_ok=True)

FINAL_MODEL_NAME = "M14_M13+SAA"

FINAL_MODEL_PKL   = os.path.join(OUTDIR, f"{FINAL_MODEL_NAME}__final_pipeline.joblib")
FINAL_FORMULA_TXT = os.path.join(OUTDIR, f"{FINAL_MODEL_NAME}__final_formula.txt")
FINAL_COEF_CSV    = os.path.join(OUTDIR, f"{FINAL_MODEL_NAME}__final_coefficients.csv")


def fit_save_final_model_and_formula(df: pd.DataFrame, model_name: str, features: List[str]):
    """
    Fit FINAL model on full available subset (same subset rule as CV: dropna),
    save pipeline, and export final formula (standardized + original scale).
    """
    # 1) prepare subset (dropna on label+features, label must be 0/1)
    df_sub = t5.prepare_subset_for_model(df, features, model_name=model_name)
    X = df_sub[features].values
    y = df_sub[t5.LABEL_COL].values

    # 2) fit pipeline on full subset
    pipe = t5.make_pipeline(best_params=None)
    pipe.fit(X, y)

    # 3) save model pipeline
    joblib.dump(pipe, FINAL_MODEL_PKL)
    print(f"[Saved FINAL model] {FINAL_MODEL_PKL}")

    # 4) extract scaler + coefficients
    scaler = pipe.named_steps["scaler"]
    clf = pipe.named_steps["clf"]

    means = scaler.mean_.astype(float)
    scales = scaler.scale_.astype(float)
    w = clf.coef_.ravel().astype(float)
    b0 = float(clf.intercept_.ravel()[0])

    # Convert to original feature scale:
    # logit(p) = b0 + Σ w_j * ((x_j - mean_j)/scale_j)
    #          = b_orig + Σ a_j * x_j
    # where a_j = w_j/scale_j and b_orig = b0 - Σ w_j*mean_j/scale_j
    a = w / scales
    b_orig = b0 - float(np.sum(w * means / scales))

    # 5) save coef table
    coef_df = pd.DataFrame({
        "feature": features,
        "scaler_mean": means,
        "scaler_scale": scales,
        "coef_on_z": w,     # coefficient on standardized z
        "coef_on_x": a,     # coefficient on raw x
        "intercept_b0": [b0] + [np.nan] * (len(features) - 1),
        "intercept_b_orig": [b_orig] + [np.nan] * (len(features) - 1),
    })
    coef_df.to_csv(FINAL_COEF_CSV, index=False, encoding="utf-8-sig")
    print(f"[Saved FINAL coefficients] {FINAL_COEF_CSV}")

    # 6) write formula txt
    lines = []
    lines.append(f"FINAL MODEL: {model_name}")
    lines.append(f"Features (order matters): {features}")
    lines.append("")
    lines.append("A) Standardized-space formula (exactly what the pipeline uses)")
    lines.append("   z_j = (x_j - mean_j) / scale_j")
    lines.append("   logit(p) = b0 + Σ w_j * z_j")
    lines.append(f"   b0 = {b0:.6g}")
    for name, cj, mj, sj in zip(features, w, means, scales):
        lines.append(f"   w[{name}] = {cj:.6g}   (mean={mj:.6g}, scale={sj:.6g})")
    lines.append("")
    lines.append("B) Original feature scale formula (easier to report as a single linear equation)")
    lines.append("   logit(p) = b_orig + Σ a_j * x_j")
    lines.append(f"   b_orig = {b_orig:.6g}")
    for name, aj in zip(features, a):
        lines.append(f"   a[{name}] = {aj:.6g}")
    lines.append("")
    lines.append("C) Probability")
    lines.append("   p = 1 / (1 + exp(-logit(p)))")

    with open(FINAL_FORMULA_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Saved FINAL formula] {FINAL_FORMULA_TXT}")

    return pipe, df_sub


def main():
    # Ensure label name is consistent with test5.py (in case you changed it elsewhere)
    t5.LABEL_COL = "efficacy"

    # 1) auto pick sheet and load
    sheet = t5.auto_pick_sheet(XLSX_PATH)
    df = pd.read_excel(XLSX_PATH, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    print(f"[OK] Loaded sheet: {sheet}, shape={df.shape}")

    # 2) add interactions (keeps same feature engineering behavior as your CV script)
    df = t5.add_all_interactions(df)

    # 3) get M3 features from your existing definition
    if FINAL_MODEL_NAME not in t5.MODELS:
        raise KeyError(f"FINAL_MODEL_NAME='{FINAL_MODEL_NAME}' not found in test5.MODELS. "
                       f"Available: {list(t5.MODELS.keys())}")

    feats = t5.MODELS[FINAL_MODEL_NAME]
    print(f"[INFO] FINAL model = {FINAL_MODEL_NAME}")
    print(f"[INFO] Features = {feats}")

    # 4) fit + save + export formula
    _pipe, df_sub = fit_save_final_model_and_formula(df, FINAL_MODEL_NAME, feats)
    print(f"[INFO] Final training subset rows = {len(df_sub)}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
