# scripts/plot_param_sensitivity.py
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

IN_CSV  = Path("results/sensitivity/runs.csv")
OUT_DIR = Path("results/sensitivity/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Helper to plot one sweep (AP vs param) for two datasets
def plot_sweep(df: pd.DataFrame, selector: str, x_col: str, x_order, title: str, outfile: Path):
    import numpy as np
    plt.figure(figsize=(6.0, 4.0))
    for ds in ["CollegeMsg", "Enron"]:   # or ["CollegeMsg","MOOC"] if that’s your pair
        dsd = df[
            df["tag"].str.contains(selector, case=False, regex=True)
            & df["csv"].str.contains(ds, case=False, regex=False)   # <-- case-insensitive, no regex
        ].copy()
        if dsd.empty:
            print(f"[warn] no rows for {ds} in {selector}")
            continue
        dsd = dsd.sort_values(x_col)
        # enforce x order
        dsd[x_col] = pd.Categorical(dsd[x_col], categories=x_order, ordered=True)
        dsd = dsd.sort_values(x_col)
        plt.plot(dsd[x_col].astype(float), dsd["ap"].astype(float), marker="o", label=ds)

    plt.xlabel(x_col.replace("_", " "))
    plt.ylabel("Average Precision (AP)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()

def main():
    df = pd.read_csv(IN_CSV)

    # 1) alpha sweep
    sub = df[df["tag"].str.contains("sweep_alpha")]
    plot_sweep(
        sub.assign(alpha=sub["alpha"].round(3)),
        selector="sweep_alpha",
        x_col="alpha",
        x_order=[0.4, 0.5, 0.6],
        title="Temporal-biased Intensity α",
        outfile=OUT_DIR / "sweep_alpha_ap.png"
    )

    # 2) walks sweep
    sub = df[df["tag"].str.contains("sweep_walks")]
    plot_sweep(
        sub,
        selector="sweep_walks",
        x_col="walks",
        x_order=[16, 32, 64],
        title="Number of Temporal Walks C",
        outfile=OUT_DIR / "sweep_walks_ap.png"
    )

    # 3) walk_L sweep
    sub = df[df["tag"].str.contains("sweep_walkL")]
    plot_sweep(
        sub,
        selector="sweep_walkL",
        x_col="walk_L",
        x_order=[1, 2, 3, 4, 5, 6],
        title="Temporal Walk Length L",
        outfile=OUT_DIR / "sweep_walkL_ap.png"
    )

    # 4) neg_per_pos sweep
    sub = df[df["tag"].str.contains("sweep_neg")]
    plot_sweep(
        sub,
        selector="sweep_neg",
        x_col="neg_per_pos",
        x_order=[1, 3, 6, 9, 18],
        title="Negatives per Positive (k)",
        outfile=OUT_DIR / "sweep_neg_per_pos_ap.png"
    )

    print(f"Saved plots to: {OUT_DIR}")

if __name__ == "__main__":
    main()
