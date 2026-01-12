# scripts/ablation_incidence_only_tuned.py
from __future__ import annotations
import itertools, subprocess, sys
from pathlib import Path

DATASETS = {"CollegeMsg": "data/ml_CollegeMsg.csv", "MOOC": "data/ml_mooc.csv"}
SPLITS   = ["inductive_atleast1", "inductive_both"]
SEEDS    = [1, 2, 3]

# small fairness grid (keep it light)
GRID = dict(
    walks=[8, 16],
    alpha=[0.3, 0.5],
    walk_L=[4, 6],
)

FIXED = dict(encoder="rnn", activation="tanh", optimizer="sgd",
             neg_per_pos=1, epochs=50, batch=32, lr=1e-2)

OUT = Path("results/ablations/incidence_only_tuned.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)
if OUT.exists(): OUT.unlink()

def run_once(csv_path, ds_name, split, seed, walks, alpha, walk_L):
    cmd = [
        sys.executable, "-m", "runners.run_ours",
        "--csv", csv_path, "--split_policy", split,
        "--encoder", FIXED["encoder"], "--activation", FIXED["activation"], "--optimizer", FIXED["optimizer"],
        "--neg_per_pos", str(FIXED["neg_per_pos"]), "--walks", str(walks),
        "--alpha", str(alpha), "--walk_L", str(walk_L),
        "--no_node2vec",                 # incidence-only
        "--epochs", str(FIXED["epochs"]), "--batch", str(FIXED["batch"]), "--lr", str(FIXED["lr"]),
        "--tag", "incidence_only_tuned",
        "--seed", str(seed),
        "--save_csv", str(OUT),
    ]
    print(">>>", ds_name, split, f"walks={walks} alpha={alpha} L={walk_L}", "seed", seed)
    subprocess.run(cmd, check=True)

def main():
    for name, csv in DATASETS.items():
        for split in SPLITS:
            for walks, alpha, L in itertools.product(GRID["walks"], GRID["alpha"], GRID["walk_L"]):
                for seed in SEEDS:
                    run_once(csv, name, split, seed, walks, alpha, L)
    print("\nWrote:", OUT)

if __name__ == "__main__":
    main()
