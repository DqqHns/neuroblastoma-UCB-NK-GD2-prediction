

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)

# =========================
# CONFIG
# =========================
OOF_CSV = r"G:/Dqq_code/nk_model(1.8)/compare_methods_results_new_COMPLETE_CASE = True/oof_pred_mean.csv"
DATA_XLSX = r"G:/Dqq_code/nk_model(1.8)/data/DATA.xlsx"

SCORE_COL = "M14_LogReg_L2"   # final model probability column in oof_pred_mean.csv
LABEL_COL = "y"              # your oof file label column is 'y'
ID_COL = "patient_id"        # your oof file id column is 'patient_id'

# Optional: if auto-detection picks a long table, set explicitly to a patient-level sheet.
# Examples: "patient_features_all", "patient_features"
GROUP_SHEET = None

OUT_DIR = r"G:/Dqq_code/nk_model(1.8)/draw/threshold_from_step8_results"
FIXED_THRESHOLD = 0.5


CMAP = "Blues"

CM_XLABELS = ["Predicted 0", "Predicted 1"]

CM_YLABELS = ["True 0", "True 1"]


TITLE_FONT_FAMILY = 'Times New Roman'
# 标题字体大小
TITLE_FONT_SIZE = 16

TITLE_FONT_WEIGHT = 'bold' 


TITLE_ALL = "Confusion Matrix in the Overall Cohort"
TITLE_GROUP0 = "Chemotherapy + anti-GD2 + UCB-NK"
TITLE_GROUP1 = "anti-GD2 + UCB-NK"

# =========================
# Helpers
# =========================
def normalize_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def find_group_sheet(xlsx_path: Path) -> str:
    xls = pd.ExcelFile(xlsx_path)
    best = None
    best_score = -1
    for sh in xls.sheet_names:
        try:
            df = pd.read_excel(xlsx_path, sheet_name=sh, nrows=200)
        except Exception:
            continue
        cols = set(df.columns.astype(str))
        score = 0
        if "group01" in cols or "合并用药组别(1=化疗+GD2+NK，2=GD2联合NK)" in cols:
            score += 5
        if any(c in cols for c in ["patient_id","id","ID","Name","name","患者ID","病人ID","患者编号","编号"]):
            score += 3
       
        if "Relative date" in cols or "relative date" in cols:
            score -= 2
        if score > best_score:
            best_score = score
            best = sh
    if best is None:
        raise ValueError("Could not auto-detect a sheet containing group labels.")
    return best

def extract_group_table(xlsx_path: Path, sheet: str | None) -> tuple[pd.DataFrame, str]:
    if sheet is None:
        sheet = find_group_sheet(xlsx_path)
    df = pd.read_excel(xlsx_path, sheet_name=sheet)

    # find ID column
    id_col = None
    for c in ["patient_id","id","ID","Name","name","患者ID","病人ID","患者编号","编号"]:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        raise ValueError(f"Cannot find an ID column in Excel sheet '{sheet}'.")

    # find group column
    if "group01" in df.columns:
        group_raw = df["group01"]
    else:
        cand = "合并用药组别(1=化疗+GD2+NK，2=GD2联合NK)"
        if cand in df.columns:
            group_raw = df[cand]
        else:
            # fallback: any column containing '组别'
            gc = None
            for c in df.columns:
                if "组别" in str(c):
                    gc = c
                    break
            if gc is None:
                raise ValueError(f"Cannot find group column in Excel sheet '{sheet}'.")
            group_raw = df[gc]

    # map 1/2 -> 0/1 if needed
    g = pd.to_numeric(group_raw, errors="coerce")
    if set(g.dropna().unique()).issubset({1, 2}):
        group01 = (g == 2).astype(int)  # 1->0, 2->1
    else:
        group01 = g

    out = pd.DataFrame({"_id": normalize_id(df[id_col]), "group01": pd.to_numeric(group01, errors="coerce")})
    out = out.dropna(subset=["_id"])

   
    out["group01"] = pd.to_numeric(out["group01"], errors="coerce")
    out = out.dropna(subset=["group01"])
    out = (
        out.groupby("_id", as_index=False)["group01"]
           .agg(lambda s: float(s.mode().iloc[0]) if len(s.mode()) else float(s.iloc[0]))
    )
    out["group01"] = out["group01"].astype(int)

    return out, sheet

def youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float, float]:
    fpr, tpr, thr = roc_curve(y_true, y_score)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thr[idx]), float(tpr[idx]), float(fpr[idx])

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    y_pred = (y_prob > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    se = recall_score(y_true, y_pred, zero_division=0)  # sensitivity
    sp = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = precision_score(y_true, y_pred, zero_division=0)
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, accuracy=acc, sensitivity=se, specificity=sp, ppv=ppv, npv=npv, f1=f1)

def plot_confusion(y_true: np.ndarray, y_prob: np.ndarray, thr: float, title: str, out_png: Path,
                   xlabels: list[str], ylabels: list[str], cmap: str,
                   title_fontdict: dict | None = None):  # +++ 新增字体字典参数 +++
  
    y_pred = (y_prob > thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1]).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(cm_pct, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    cbar =fig.colorbar(im, ax=ax)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set(
        xticks=[0,1], yticks=[0,1],
        xticklabels=xlabels, yticklabels=ylabels,
        xlabel="Predicted", ylabel="True"
    )
    # +++ 应用自定义标题字体 +++
    ax.set_title(title, fontdict=title_fontdict)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i,j])}\n{cm_pct[i,j]*100:.1f}%", ha="center", va="center", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load step8 predictions (patient-level, one row per patient)
    oof = pd.read_csv(OOF_CSV)
    if ID_COL not in oof.columns or LABEL_COL not in oof.columns or SCORE_COL not in oof.columns:
        raise ValueError(f"oof file must contain columns: {ID_COL}, {LABEL_COL}, {SCORE_COL}")
    oof = oof[[ID_COL, LABEL_COL, SCORE_COL]].copy()
    oof.rename(columns={ID_COL:"_id", LABEL_COL:"y_true", SCORE_COL:"y_prob"}, inplace=True)
    oof["_id"] = normalize_id(oof["_id"])
    oof["y_true"] = pd.to_numeric(oof["y_true"], errors="coerce").astype(int)
    oof["y_prob"] = pd.to_numeric(oof["y_prob"], errors="coerce")
    oof = oof.dropna(subset=["y_true","y_prob"])

    # Merge group labels (deduplicated)
    grp_tbl, used_sheet = extract_group_table(Path(DATA_XLSX), GROUP_SHEET)
    df = oof.merge(grp_tbl, on="_id", how="left")

    # Sanity checks: N must match patient-level
    n0 = len(oof)
    n1 = len(df)
    if n1 != n0:
        raise RuntimeError(
            f"N mismatch after merge: before={n0}, after={n1}. "
            f"This should NOT happen after deduplicating group labels. "
            f"Please check DATA.xlsx sheet '{used_sheet}'."
        )

    # Debug join table
    df.to_csv(out_dir/"analysis_table_oof_plus_group.csv", index=False)

    # Youden on ALL
    thr_y, tpr_y, fpr_y = youden_threshold(df["y_true"].values, df["y_prob"].values)
    thr_fixed = float(FIXED_THRESHOLD)

    pd.DataFrame([
        {"threshold_name":"youden_all", "threshold":thr_y, "TPR":tpr_y, "FPR":fpr_y,
         "score_col":SCORE_COL, "label_col":LABEL_COL, "group_sheet":used_sheet, "n_patients":n0},
        {"threshold_name":"fixed_0.5", "threshold":thr_fixed, "TPR":np.nan, "FPR":np.nan,
         "score_col":SCORE_COL, "label_col":LABEL_COL, "group_sheet":used_sheet, "n_patients":n0},
    ]).to_csv(out_dir/"threshold_summary.csv", index=False)

    # 根据配置参数构建标题字体字典
    title_font_config = {
        'family': TITLE_FONT_FAMILY,
        'size': TITLE_FONT_SIZE,
        'weight': TITLE_FONT_WEIGHT
    }

    # +++ 构建分组标题模板映射 +++
    cohort_title_templates = {
        "all": TITLE_ALL,
        "group0": TITLE_GROUP0,
        "group1": TITLE_GROUP1,
    }

    # Cohorts
    cohorts = [("all", df)]
    if df["group01"].notna().any():
        cohorts += [("group0", df[df["group01"] == 0]), ("group1", df[df["group01"] == 1])]

    # Confusion matrices + metrics
    rows = []
    for cname, sub in cohorts:
        for tname, thr in [("fixed_0.5", thr_fixed), ("youden_all", thr_y)]:
            m = compute_metrics(sub["y_true"].values, sub["y_prob"].values, thr)
            rows.append({"cohort": cname, "threshold_name": tname, "threshold": thr, "n": len(sub), **m})
            # +++ 使用标题模板和格式化方法生成动态标题 +++
            title_template = cohort_title_templates.get(cname, TITLE_ALL)  # 安全回退
            title = title_template.format(thr=thr, tname=tname)

            plot_confusion(sub["y_true"].values, sub["y_prob"].values, thr, title,
                           out_dir / f"confusion_{cname}_{tname}.png",
                           xlabels=CM_XLABELS, ylabels=CM_YLABELS, cmap=CMAP,
                           title_fontdict=title_font_config)
    pd.DataFrame(rows).to_csv(out_dir / "confusion_metrics.csv", index=False)

    print("="*72)
    print("[OK] Done.")
    print(f"  oof patients  : {n0}")
    print(f"  group sheet   : {used_sheet} (group labels deduplicated per patient_id)")
    print(f"  Youden thr    : {thr_y:.6f} (TPR={tpr_y:.4f}, FPR={fpr_y:.4f})")
    print(f"  Output dir    : {out_dir.resolve()}")
    print("="*72)

if __name__ == "__main__":
    main()
