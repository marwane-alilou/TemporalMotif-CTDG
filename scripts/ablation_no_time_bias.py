# scripts/ablation_no_time_bias.py
from __future__ import annotations
import subprocess, sys
from pathlib import Path

DATASETS = {
    "CollegeMsg": "data/ml_CollegeMsg.csv",
    "MOOC":       "data/ml_mooc.csv",
}
SPLITS = ["inductive_atleast1", "inductive_both"]
SEEDS  = [1, 2, 3]

# best (or fixed) config you want to use for the ablation
CFG = dict(encoder="rnn", activation="tanh", optimizer="sgd",
           neg_per_pos=1, walks=16, alpha=0.4, walk_L=6,
           epochs=50, batch=32, lr=1e-2)

OUT = Path("results/ablations/no_time_bias.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)
if OUT.exists(): OUT.unlink()

def run_once(csv_path, ds_name, split, seed):
    cmd = [
        sys.executable, "-m", "runners.run_ours",
        "--csv", csv_path,
        "--split_policy", split,
        "--encoder", CFG["encoder"], "--activation", CFG["activation"], "--optimizer", CFG["optimizer"],
        "--neg_per_pos", str(CFG["neg_per_pos"]), "--walks", str(CFG["walks"]),
        "--alpha", str(CFG["alpha"]), "--walk_L", str(CFG["walk_L"]),
        "--epochs", str(CFG["epochs"]), "--batch", str(CFG["batch"]), "--lr", str(CFG["lr"]),
        "--no_time_bias",               # ← ablation switch
        "--tag", "no_time_bias",
        "--seed", str(seed),
        "--save_csv", str(OUT),
    ]
    print(">>>", ds_name, split, "seed", seed)
    subprocess.run(cmd, check=True)

def main():
    for name, csv in DATASETS.items():
        for split in SPLITS:
            for seed in SEEDS:
                run_once(csv, name, split, seed)
    print("\nWrote:", OUT)

if __name__ == "__main__":
    main()
