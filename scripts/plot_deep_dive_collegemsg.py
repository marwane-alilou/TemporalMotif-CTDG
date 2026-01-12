# scripts/plot_deep_dive_collegemsg.py
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

IN_ALL = Path("results/final_suite/college_3_seeds/runs_college_3_seeds.csv")
OUT = Path("results/final_suite/college_3_seeds/plots_collegemsg")
OUT.mkdir(parents=True, exist_ok=True)

def save(figpath):
    plt.tight_layout()
    plt.savefig(figpath, dpi=220)
    plt.close()

def main():
    df = pd.read_csv(IN_ALL)
    df["dataset"] = df["csv"].str.lower().apply(lambda s: "CollegeMsg" if "collegemsg" in s else "other")
    cm = df[df["dataset"]=="CollegeMsg"].copy()

    # 1) Stacked per-stage time (mean over seeds) per regime
    br_cols = ["time_trw_s","time_motif_s","time_feat_s","time_train_s"]
    agg = cm.groupby("split_policy", as_index=False)[br_cols].mean()
    agg = agg.set_index("split_policy")
    plt.figure()
    bottom = None
    for col in br_cols:
        if bottom is None:
            plt.bar(agg.index, agg[col], label=col)
            bottom = agg[col].values
        else:
            plt.bar(agg.index, agg[col], bottom=bottom, label=col)
            bottom = bottom + agg[col].values
    plt.ylabel("Seconds")
    plt.title("CollegeMsg: Per-stage wall-clock by regime (mean over seeds)")
    plt.legend()
    save(OUT / "stage_breakdown.png")

    # 2) Scaling curves (time & memory vs fraction)
    def parse_frac(x: str):
        x = x.lower()
        if "prefix_25" in x: return 0.25
        if "prefix_50" in x: return 0.50
        if "prefix_75" in x: return 0.75
        if "prefix_100" in x or "collegemsg.csv" in x: return 1.0
        return None
    cm["frac"] = cm["csv"].apply(parse_frac)
    scal = cm[(cm["frac"].notna()) & (cm["split_policy"]=="transductive")].copy()
    scal_grp = scal.groupby("frac", as_index=False)[["time_total_s","peak_mem_mb","auc","ap"]].mean()

    plt.figure()
    scal_grp = scal_grp.sort_values("frac")
    plt.plot(scal_grp["frac"], scal_grp["time_total_s"], marker="o")
    plt.xlabel("Fraction of edges kept (chronological prefix)")
    plt.ylabel("Total time (s)")
    plt.title("CollegeMsg: Scaling of total time")
    save(OUT / "scaling_time.png")

    plt.figure()
    plt.plot(scal_grp["frac"], scal_grp["peak_mem_mb"], marker="o")
    plt.xlabel("Fraction of edges kept")
    plt.ylabel("Peak memory (MB)")
    plt.title("CollegeMsg: Scaling of peak memory")
    save(OUT / "scaling_memory.png")

    # 3) AUC/AP vs total time (trade-offs)
    cm_td = cm[cm["split_policy"].isin(["transductive","inductive_atleast1","inductive_both"]) &
               cm["frac"].isna()]  # only full dataset
    plt.figure()
    for regime, d in cm_td.groupby("split_policy"):
        plt.scatter(d["time_total_s"], d["auc"], label=regime)
    plt.xlabel("Total time (s)"); plt.ylabel("AUC")
    plt.title("CollegeMsg: AUC vs total time")
    plt.legend()
    save(OUT / "tradeoff_auc_vs_time.png")

    plt.figure()
    for regime, d in cm_td.groupby("split_policy"):
        plt.scatter(d["time_total_s"], d["ap"], label=regime)
    plt.xlabel("Total time (s)"); plt.ylabel("AP")
    plt.title("CollegeMsg: AP vs total time")
    plt.legend()
    save(OUT / "tradeoff_ap_vs_time.png")

    print(f"Saved plots in: {OUT}")

if __name__ == "__main__":
    main()
