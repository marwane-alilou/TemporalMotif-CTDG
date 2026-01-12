#!/usr/bin/env python3
"""
Complexity & efficiency scaling study for *your method* vs. graph size.

What it does
------------
- Loads the full ml_*.csv (e.g., CollegeMsg), sorts by time, and creates
  time-prefix subsets at requested FRACTIONS (e.g., 0.1, 0.2, ..., 1.0).
- For each subset, calls runners/run_ours.py with your chosen hyperparams.
- Appends raw rows to results/complexity/raw_scale_ours.csv
- Produces a clean summary: results/complexity/summary_scale_ours.csv
- Fits log-log scaling exponents for:
    preprocess_s, time_train_s, time_total_s, peak_mem_mb, gpu_peak_mb
  vs number of edges |E| (and also vs the composite |E|*walks*walk_L*(neg+1))
- Optionally saves PNG plots showing observed scaling curves.

Example:
  python tools/scale_graphsize_ours.py \
    --repo_root . \
    --csv third_party/Neural-Temporal-Walks/data/ml_CollegeMsg.csv \
    --fractions 0.10 0.20 0.40 0.60 0.80 1.00 \
    --policy transductive \
    --encoder rnn --activation tanh --optimizer sgd --lr 1e-2 --batch 32 --epochs 60 \
    --walks 16 --walk_L 6 --neg_per_pos 1 \
    --make_plots
"""
from __future__ import annotations
import argparse, os, shlex, subprocess, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

def _ensure_dirs():
    out_root = Path("results/complexity")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "plots").mkdir(parents=True, exist_ok=True)
    (out_root / "tmp").mkdir(parents=True, exist_ok=True)
    return out_root

def _make_time_prefix_csv(full_csv: Path, frac: float, tmp_dir: Path) -> Path:
    """Write a time-sorted prefix (by 'ts' or 't') with ~frac edges to a temp CSV."""
    df = pd.read_csv(full_csv)
    # Support both (u,i,ts,label,idx) and (src,dst,t)
    time_col = "ts" if "ts" in df.columns else ("t" if "t" in df.columns else None)
    if time_col is None:
        raise ValueError("Input CSV must contain a timestamp column named 'ts' or 't'.")
    df = df.sort_values(time_col, kind="mergesort")  # stable
    n = max(1, int(len(df) * frac))
    df_sub = df.iloc[:n].copy()
    out = tmp_dir / f"{full_csv.stem}_frac{int(frac*100):02d}.csv"
    df_sub.to_csv(out, index=False)
    return out

def _run_one(repo_root: Path, sub_csv: Path, policy: str, mask_frac: float,
             encoder: str, activation: str, optimizer: str, lr: float,
             batch: int, epochs: int, seed: int,
             walks: int, walk_L: int, neg_per_pos: int,
             save_csv: Path):
    cmd = f"""
    python runners/run_ours.py
      --csv {shlex.quote(str(sub_csv))}
      --split_policy {policy}
      --mask_frac {mask_frac}
      --walks {walks}
      --walk_L {walk_L}
      --neg_per_pos {neg_per_pos}
      --alpha 0.5
      --encoder {encoder}
      --activation {activation}
      --optimizer {optimizer}
      --lr {lr}
      --batch {batch}
      --epochs {epochs}
      --seed {seed}
      --save_csv {shlex.quote(str(save_csv))}
    """
    cmd = " ".join(cmd.split())
    print(f"[RUN] {cmd}", flush=True)
    ret = subprocess.run(cmd, cwd=repo_root, shell=True)
    if ret.returncode != 0:
        raise RuntimeError(f"run_ours failed with code {ret.returncode}")

def _fit_powerlaw(x, y):
    """Fit y ~ a * x^b using log-log linear regression. Returns (a, b)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Keep only positive pairs
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return (np.nan, np.nan)
    lx, ly = np.log(x[m]), np.log(y[m])
    b, loga = np.polyfit(lx, ly, 1)  # ly ≈ b*lx + loga
    a = np.exp(loga)
    return (float(a), float(b))

def _summarize(raw_csv: Path, out_csv: Path, plots_dir: Path, make_plots: bool):
    df = pd.read_csv(raw_csv)
    # derive #edges in subset: it's `num_edges_train` emitted by your runner (train split),
    # but for size scaling we want to reference the *subset total edges* too if present.
    # Fall back to num_edges_train for monotone size proxy.
    if "num_edges_train" in df.columns:
        df["E_train"] = df["num_edges_train"]
    else:
        # Conservative fallback
        df["E_train"] = np.nan

    # composite complexity proxy:
    if {"walks","walk_L","neg_per_pos"}.issubset(df.columns):
        df["comp_proxy"] = df["E_train"] * df["walks"] * df["walk_L"] * (df["neg_per_pos"] + 1)
    else:
        df["comp_proxy"] = np.nan

    # total preprocess
    if {"time_trw_s","time_motif_s","time_feat_s"}.issubset(df.columns):
        df["preprocess_s"] = df["time_trw_s"] + df["time_motif_s"] + df["time_feat_s"]

    # If runner added gpu_peak_mb; if not, fill zeros
    for col in ["gpu_peak_mb","peak_mem_mb"]:
        if col not in df.columns:
            df[col] = 0.0

    # Fit exponents vs E_train
    metrics = ["preprocess_s","time_train_s","time_total_s","peak_mem_mb","gpu_peak_mb"]
    fits_E = {}
    for m in metrics:
        if m in df.columns and "E_train" in df.columns:
            a, b = _fit_powerlaw(df["E_train"], df[m])
            fits_E[m] = (a, b)
        else:
            fits_E[m] = (np.nan, np.nan)

    # Also vs comp_proxy
    fits_comp = {}
    if "comp_proxy" in df.columns:
        for m in metrics:
            if m in df.columns:
                a, b = _fit_powerlaw(df["comp_proxy"], df[m])
                fits_comp[m] = (a, b)
            else:
                fits_comp[m] = (np.nan, np.nan)

    # Save tidy summary
    keep = [
        "csv","split_policy","mask_frac",
        "walks","walk_L","neg_per_pos",
        "encoder","activation","optimizer","lr","epochs","batch","seed",
        "num_nodes_train","num_edges_train","feat_dim",
        "motifs_k3","motifs_k4","motifs_k5","motifs_k6",
        "time_trw_s","time_motif_s","time_feat_s","preprocess_s",
        "time_train_s","time_total_s","peak_mem_mb","gpu_peak_mb",
        "auc","ap","val_auc","val_ap",
        "E_train","comp_proxy","tag"
    ]
    keep = [c for c in keep if c in df.columns]
    df[keep].to_csv(out_csv, index=False)
    print(f"[SAVED] summary -> {out_csv}")

    # Emit a small report with exponents
    txt = plots_dir.parent / "scaling_report.txt"
    with open(txt, "w") as f:
        f.write("# Scaling exponents (y ~ a * x^b)\n\n")
        f.write("## vs |E_train|\n")
        for m,(a,b) in fits_E.items():
            f.write(f"{m:>15s}: a={a:.4g}, b={b:.3f}\n")
        f.write("\n## vs comp_proxy = |E_train|*walks*walk_L*(neg+1)\n")
        for m,(a,b) in fits_comp.items():
            f.write(f"{m:>15s}: a={a:.4g}, b={b:.3f}\n")
    print(f"[SAVED] exponents -> {txt}")

    if make_plots:
        try:
            import matplotlib.pyplot as plt
            # 1) time vs E
            for m in ["preprocess_s","time_train_s","time_total_s"]:
                if m in df.columns:
                    plt.figure()
                    plt.scatter(df["E_train"], df[m])
                    plt.xscale("log"); plt.yscale("log")
                    a,b = fits_E.get(m,(np.nan,np.nan))
                    if np.isfinite(a) and np.isfinite(b):
                        xs = np.linspace(df["E_train"].min(), df["E_train"].max(), 100)
                        ys = a * xs**b
                        plt.plot(xs, ys)
                    plt.xlabel("|E_train| (log)"); plt.ylabel(f"{m} (s, log)")
                    plt.title(f"{m} vs |E_train|")
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"{m}_vs_edges.png", dpi=200)
                    plt.close()

            # 2) memory vs E
            for m in ["peak_mem_mb","gpu_peak_mb"]:
                if m in df.columns:
                    plt.figure()
                    plt.scatter(df["E_train"], df[m])
                    plt.xscale("log"); plt.yscale("log")
                    a,b = fits_E.get(m,(np.nan,np.nan))
                    if np.isfinite(a) and np.isfinite(b):
                        xs = np.linspace(df["E_train"].min(), df["E_train"].max(), 100)
                        ys = a * xs**b
                        plt.plot(xs, ys)
                    plt.xlabel("|E_train| (log)"); plt.ylabel(f"{m} (MiB, log)")
                    plt.title(f"{m} vs |E_train|")
                    plt.tight_layout()
                    plt.savefig(plots_dir / f"{m}_vs_edges.png", dpi=200)
                    plt.close()
            print(f"[SAVED] plots -> {plots_dir}")
        except Exception as e:
            print(f"[WARN] plotting failed: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=Path, default=Path("."))
    ap.add_argument("--csv", type=Path, required=True,
                    help="Full dataset CSV (e.g., ml_CollegeMsg.csv)")
    ap.add_argument("--fractions", nargs="+", type=float,
                    default=[0.10,0.20,0.40,0.60,0.80,1.00],
                    help="Time-prefix fractions to evaluate (0,1]")
    ap.add_argument("--policy", type=str, default="transductive",
                    choices=["none","transductive","inductive_atleast1","inductive_both"])
    ap.add_argument("--mask_frac", type=float, default=0.05)

    # your best hyperparams
    ap.add_argument("--encoder", type=str, default="rnn", choices=["rnn","lstm","transformer","hgnn"])
    ap.add_argument("--activation", type=str, default="tanh")
    ap.add_argument("--optimizer", type=str, default="sgd", choices=["sgd","adam","adamw"])
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)

    # TRW settings (kept fixed in the scaling test)
    ap.add_argument("--walks", type=int, default=16)
    ap.add_argument("--walk_L", type=int, default=6)
    ap.add_argument("--neg_per_pos", type=int, default=1)

    ap.add_argument("--raw_csv", type=Path, default=Path("results/complexity/raw_scale_ours.csv"))
    ap.add_argument("--summary_csv", type=Path, default=Path("results/complexity/summary_scale_ours.csv"))
    ap.add_argument("--make_plots", action="store_true")
    args = ap.parse_args()

    out_root = _ensure_dirs()
    tmp_dir = out_root / "tmp"

    # Run each fraction
    for frac in args.fractions:
        sub_csv = _make_time_prefix_csv(args.csv, frac, tmp_dir)
        _run_one(
            repo_root=args.repo_root, sub_csv=sub_csv,
            policy=args.policy, mask_frac=args.mask_frac,
            encoder=args.encoder, activation=args.activation, optimizer=args.optimizer, lr=args.lr,
            batch=args.batch, epochs=args.epochs, seed=args.seed,
            walks=args.walks, walk_L=args.walk_L, neg_per_pos=args.neg_per_pos,
            save_csv=args.raw_csv
        )

    # Summarize and fit exponents
    _summarize(args.raw_csv, args.summary_csv, out_root / "plots", args.make_plots)

if __name__ == "__main__":
    main()
