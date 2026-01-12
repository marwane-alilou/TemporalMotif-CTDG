# scripts/plot_stage2.py
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

IN  = "results/stage2_rnn.csv"
OUT = Path("results/plots")
OUT.mkdir(parents=True, exist_ok=True)

def _line(df, x, y, fname, groupby=None):
    plt.figure()
    if groupby:
        for g, d in df.groupby(groupby):
            d = d.sort_values(x)
            plt.plot(d[x], d[y], marker="o", label=str(g))
        plt.legend()
    else:
        d = df.sort_values(x)
        plt.plot(d[x], d[y], marker="o")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{y} vs {x}")
    plt.tight_layout()
    plt.savefig(OUT / fname, dpi=200)
    plt.close()

def main():
    df = pd.read_csv(IN)

    # PERFORMANCE CURVES (averaged over the other params)
    perf_cols = ["auc", "ap"]
    axes = [
        ("walk_L",   "walk_L"),
        ("walks",    "walks"),
        ("alpha",    "alpha"),
        ("neg_per_pos", "neg_per_pos"),
    ]
    for metric in perf_cols:
        for col, label in axes:
            agg = df.groupby(col, as_index=False)[metric].mean()
            _line(agg, col, metric, f"{metric}_vs_{label}.png")

    # COMPLEXITY CURVES (averaged)
    comp_cols = ["time_trw_s", "time_motif_s", "time_feat_s", "time_train_s", "time_total_s", "peak_mem_mb"]
    for col, label in axes:
        agg_time = df.groupby(col, as_index=False)[comp_cols].mean()
        for c in comp_cols:
            _line(agg_time, col, c, f"{c}_vs_{label}.png")

    # Optional: trade-off curves (AUC vs time_total)
    agg = df.groupby(["walk_L", "walks", "alpha", "neg_per_pos"], as_index=False)[["auc","ap","time_total_s"]].mean()
    _line(agg, "time_total_s", "auc", "auc_vs_time_total.png")
    _line(agg, "time_total_s", "ap",  "ap_vs_time_total.png")

    print(f"Saved plots to: {OUT}")

if __name__ == "__main__":
    main()
