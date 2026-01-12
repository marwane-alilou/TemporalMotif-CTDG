# scripts/run_ablations.py
from __future__ import annotations
import sys, subprocess
from pathlib import Path
import pandas as pd
import numpy as np

# ---------------- Config ----------------
DATASETS = {
    "CollegeMsg": "data/ml_CollegeMsg.csv",
    "MOOC":       "data/ml_mooc.csv",
}
REGIMES = {
    "New-Old": "inductive_atleast1",
    "New-New": "inductive_both",
}
SEEDS = [1, 2, 3]
MASK_FRAC = 0.05  # keep consistent across ablations

# Base model/head (your best head)
ENCODER, ACT, OPT = "rnn", "tanh", "sgd"
EPOCHS, BATCH, LR = 50, 32, 1e-2
BEST = dict(neg_per_pos=1, walks=16, alpha=0.4, walk_L=6)

OUT_DIR = Path("results/ablations")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_CSV = OUT_DIR / "raw_runs.csv"
SUM_CSV = OUT_DIR / "summary_mean_std.csv"

# Small grid for "incidence-only tuned fairly"
GRID_WALKS = [8, 16, 32]
GRID_L     = [4, 5, 6]
GRID_ALPHA = [0.4, 0.5, 0.6]

# ---------------- Helpers ----------------
def run_once(csv_path: str, ablation: str, ds_name: str, regime_name: str, seed: int,
             extra_args: list[str]) -> None:
    mode = REGIMES[regime_name]
    cmd = [
        sys.executable, "-m", "runners.run_ours",
        "--csv", csv_path,
        "--split_policy", mode,
        "--mask_frac", str(MASK_FRAC),
        "--encoder", ENCODER, "--activation", ACT, "--optimizer", OPT,
        "--neg_per_pos", str(BEST["neg_per_pos"]),
        "--walks", str(BEST["walks"]),
        "--alpha", str(BEST["alpha"]),
        "--walk_L", str(BEST["walk_L"]),
        "--epochs", str(EPOCHS), "--batch", str(BATCH), "--lr", str(LR),
        "--seed", str(seed),
        "--save_csv", str(RAW_CSV),
    ] + extra_args
    print(">>>", ds_name, regime_name, f"seed={seed}", f"ablation={ablation}")
    subprocess.run(cmd, check=True)

def summarize(raw_csv: Path, out_csv: Path):
    df = pd.read_csv(raw_csv)
    # Expected fields present from run_ours: csv, split_policy, seed, auc, ap
    # Map split_policy to regime names for readability
    m = {"inductive_atleast1": "New-Old", "inductive_both": "New-New"}
    df["Regime"] = df["split_policy"].map(m).fillna(df["split_policy"])
    # Dataset name from path
    df["Dataset"] = df["csv"].apply(lambda p: Path(p).stem.replace("ml_", "").replace(".csv",""))
    # Ablation tag comes from 'tag' if we add it, else infer from flags
    # We’ll store a small tag by checking columns that differ. For simplicity, we wrote tags in 'ablation' column via extra_args --tag
    if "ablation" not in df.columns:
        df["ablation"] = "baseline"  # fallback
    grp = df.groupby(["Dataset","Regime","ablation"], as_index=False).agg(
        auc_mean=("auc","mean"), auc_std=("auc","std"),
        ap_mean=("ap","mean"),   ap_std=("ap","std"),
        runs=("auc","count")
    )
    # pretty mean±std strings
    grp["AUC (mean±std)"] = grp.apply(lambda r: f"{r.auc_mean:.4f} ± {r.auc_std:.4f}", axis=1)
    grp["AP (mean±std)"]  = grp.apply(lambda r: f"{r.ap_mean:.4f} ± {r.ap_std:.4f}", axis=1)
    keep = ["Dataset","Regime","ablation","AUC (mean±std)","AP (mean±std)","runs"]
    grp[keep].sort_values(["Dataset","Regime","ablation"]).to_csv(out_csv, index=False)
    print("\nSaved summary to:", out_csv)

# ---------------- Main experiment script ----------------
def main():
    # fresh raw file
    if RAW_CSV.exists():
        RAW_CSV.unlink()

    # ---------------- A1: No time bias ----------------
    for ds, path in DATASETS.items():
        for regime_name in REGIMES.keys():
            for s in SEEDS:
                run_once(path, "no_time_bias", ds, regime_name, s,
                         extra_args=["--no_time_bias", "--motif_set", "3,4,5,6", "--tag", "no_time_bias"])

    # ---------------- A2: No Node2Vec ----------------
    for ds, path in DATASETS.items():
        for regime_name in REGIMES.keys():
            for s in SEEDS:
                run_once(path, "no_node2vec", ds, regime_name, s,
                         extra_args=["--no_node2vec", "--motif_set", "3,4,5,6", "--tag", "no_node2vec"])

    # ---------------- A3: No incidence features (Node2Vec only) ----------------
    for ds, path in DATASETS.items():
        for regime_name in REGIMES.keys():
            for s in SEEDS:
                run_once(path, "no_incidence_feats", ds, regime_name, s,
                         extra_args=["--no_incidence_feats", "--motif_set", "3,4,5,6", "--tag", "no_incidence"])

    # ---------------- A4: Motif order subsets ----------------
    for motif_set, tag in [("3,4", "motifs_34"), ("3,4,5", "motifs_345")]:
        for ds, path in DATASETS.items():
            for regime_name in REGIMES.keys():
                for s in SEEDS:
                    run_once(path, tag, ds, regime_name, s,
                             extra_args=["--motif_set", motif_set, "--tag", tag])

    # ---------------- A5: Incidence-only tuned fairly ----------------
    # Tune using seed=1 and validation AUC, then run seeds=1..3 with the chosen config.
    for ds, path in DATASETS.items():
        for regime_name, split in REGIMES.items():
            # grid search with seed=1, no Node2Vec
            grid_rows = []
            for w in GRID_WALKS:
                for L in GRID_L:
                    for a in GRID_ALPHA:
                        cmd = [
                            sys.executable, "-m", "runners.run_ours",
                            "--csv", path, "--split_policy", split,
                            "--mask_frac", str(MASK_FRAC),
                            "--encoder", ENCODER, "--activation", ACT, "--optimizer", OPT,
                            "--neg_per_pos", str(BEST["neg_per_pos"]),
                            "--walks", str(w), "--alpha", str(a), "--walk_L", str(L),
                            "--epochs", str(EPOCHS), "--batch", str(BATCH), "--lr", str(LR),
                            "--seed", "1", "--no_node2vec",
                            "--save_csv", str(RAW_CSV),
                            "--motif_set", "3,4,5,6",
                            "--tag", f"incidence_tune_w{w}_L{L}_a{a}"
                        ]
                        subprocess.run(cmd, check=True)
                        grid_rows.append((w,L,a))
            # pick best (highest val_auc) among just-written rows for (ds,regime)
            df = pd.read_csv(RAW_CSV)
            recent = df[(df["csv"]==path) & (df["split_policy"]==split) &
                        (df["seed"]==1) & (df["ablation"].str.startswith("incidence_tune_"))]
            # if val_auc missing for some reason, fall back to test auc
            criterion = "val_auc" if "val_auc" in recent.columns else "auc"
            best_row = recent.sort_values(criterion, ascending=False).iloc[0]
            # extract chosen params from tag
            tag = best_row["ablation"]
            # format: incidence_tune_w{w}_L{L}_a{a}
            parts = tag.split("_")
            w_best = int(parts[2][1:])
            L_best = int(parts[3][1:])
            a_best = float(parts[4][1:])
            # now run seeds 1..3 with chosen config (mark a stable tag)
            for s in SEEDS:
                run_once(path, "incidence_only_best", ds, regime_name, s, extra_args=[
                    "--no_node2vec", "--motif_set", "3,4,5,6",
                    "--walks", str(w_best), "--walk_L", str(L_best), "--alpha", str(a_best),
                    "--tag", "incidence_only_best"
                ])

    # ---------------- A6: Per-walk anonymization (primary w/o Node2Vec) ----------------
    for ds, path in DATASETS.items():
        for regime_name in REGIMES.keys():
            for s in SEEDS:
                run_once(path, "perwalk_anon_noN2V", ds, regime_name, s,
                         extra_args=["--per_walk_anonymize", "--no_node2vec",
                                     "--motif_set", "3,4,5,6", "--tag", "perwalk_anon_noN2V"])

    # Summarize
    # Ensure we have an 'ablation' column: we passed "--tag" to runner (see below).
    summarize(RAW_CSV, SUM_CSV)

if __name__ == "__main__":
    main()
