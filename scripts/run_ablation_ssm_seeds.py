from __future__ import annotations
import subprocess, sys
from pathlib import Path
import pandas as pd

OUT = Path("results/ablations/ssm_runs.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

def run_one(csv, split, seed):
    cmd = [
        sys.executable, "-m", "runners.ablation_ssm_encoder",
        "--csv", csv,
        "--split_policy", split, "--mask_frac", "0.05",
        "--neg_per_pos", "1", "--walks", "16", "--alpha", "0.4", "--walk_L", "6",
        "--encoder", "rnn", "--activation", "tanh", "--optimizer", "sgd",
        "--epochs", "50", "--batch", "32", "--lr", "0.01",
        "--ssm_bins", "128", "--ssm_d_state", "64", "--ssm_d_out", "64", "--ssm_emb_dim", "64", "--ssm_epochs", "10",
        "--save_csv", str(OUT),
        "--seed", str(seed),
    ]
    subprocess.run(cmd, check=True)

def main():
    # clear old results
    if OUT.exists():
        OUT.unlink()

    jobs = [
        ("data/ml_CollegeMsg.csv", "inductive_atleast1"),
        ("data/ml_enron.csv",      "inductive_both"),
    ]
    seeds = [1,2,3]
    for csv, split in jobs:
        for s in seeds:
            run_one(csv, split, s)

    # aggregate mean±std for AUC/AP per (csv, split_policy)
    df = pd.read_csv(OUT)
    agg = df.groupby(["csv","split_policy"]).agg(
        auc_mean=("auc","mean"), auc_std=("auc","std"),
        ap_mean=("ap","mean"),   ap_std=("ap","std"),
    ).reset_index()
    print("\n=== AUC/AP mean±std over 3 seeds ===")
    for _, r in agg.iterrows():
        print(f"{r['csv']} | {r['split_policy']:>20} | "
              f"AUC {r['auc_mean']:.4f}±{(r['auc_std'] or 0):.4f} | "
              f"AP {r['ap_mean']:.4f}±{(r['ap_std'] or 0):.4f}")

if __name__ == "__main__":
    main()
