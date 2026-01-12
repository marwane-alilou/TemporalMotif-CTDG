# scripts/aggregate_final.py
from __future__ import annotations
import pandas as pd
from pathlib import Path
import numpy as np

IN = Path("results/nodecls_primary/runs.csv")
OUT_DIR = Path("results/nodecls_primary")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def mean_std(df, col):
    return f"{df[col].mean():.4f} ± {df[col].std(ddof=1):.4f}"

def main():
    df = pd.read_csv(IN)

    # Identify dataset from csv path
    def parse_dataset(x: str) -> str:
        x = x.lower()
        if "reddit" in x: return "reddit"
        if "wikipedia" in x: return "wikipedia"
        if "mooc" in x: return "MOOC"
        return "Unknown"
    df["dataset"] = df["csv"].apply(parse_dataset)

    # 1) Final metrics: mean±std across seeds
    # group by dataset, split_policy
    g = df.groupby(["dataset", "split_policy"], as_index=False)
    rows = []
    for (ds, sp), sub in g:
        rows.append({
            "dataset": ds,
            "regime": sp,
            "AUC_mean": sub["auc"].mean(),
            "AUC_std":  sub["auc"].std(ddof=1),
            "AP_mean":  sub["ap"].mean(),
            "AP_std":   sub["ap"].std(ddof=1),
            "AUC_mean±std": mean_std(sub, "auc"),
            "AP_mean±std":  mean_std(sub, "ap"),
            "runs": len(sub)
        })
    final_metrics = pd.DataFrame(rows)
    final_metrics.to_csv(OUT_DIR / "final_metrics.csv", index=False)
    print("\n== AUC/AP mean±std over seeds ==")
    print(final_metrics[["dataset","regime","AUC_mean±std","AP_mean±std","runs"]])

    # 2) Efficiency summary: per dataset×regime averages
    eff_cols = ["time_total_s","peak_mem_mb"]
    eff = df.groupby(["dataset","split_policy"], as_index=False)[eff_cols].mean()
    eff = eff.rename(columns={"split_policy":"regime"})
    eff.to_csv(OUT_DIR / "efficiency_summary.csv", index=False)
    print("\n== Efficiency summary (means) ==")
    print(eff)

    # 3) CollegeMsg deep-dive breakdown (averaged over seeds)
    cm = df[df["dataset"]=="CollegeMsg"].copy()
    br_cols = ["time_trw_s","time_motif_s","time_feat_s","time_train_s","time_total_s","peak_mem_mb"]
    breakdown = cm.groupby(["split_policy"], as_index=False)[br_cols].mean()
    breakdown.to_csv(OUT_DIR / "college_msg_breakdown.csv", index=False)

    # 4) CollegeMsg scaling: detect scale tag by counting rows per unique csv
    # We used filenames like CollegeMsg_prefix_25/50/75/100
    def parse_frac(x: str) -> float:
        x = x.lower()
        if "prefix_25" in x: return 0.25
        if "prefix_50" in x: return 0.50
        if "prefix_75" in x: return 0.75
        if "prefix_100" in x or "collegemsg.csv" in x: return 1.00
        return np.nan
    cm["scale_frac"] = cm["csv"].apply(parse_frac)
    cm_scal = cm[~cm["scale_frac"].isna() & (cm["split_policy"]=="transductive")]
    scal_cols = ["time_total_s","peak_mem_mb","auc","ap"]
    scaling = cm_scal.groupby("scale_frac", as_index=False)[scal_cols].mean()
    scaling.to_csv(OUT_DIR / "college_msg_scaling.csv", index=False)

    print("\nSaved:")
    print(" - final_metrics.csv")
    print(" - efficiency_summary.csv")
    print(" - college_msg_breakdown.csv")
    print(" - college_msg_scaling.csv")

if __name__ == "__main__":
    main()
