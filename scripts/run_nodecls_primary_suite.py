# scripts/run_nodecls_primary_suite.py
from __future__ import annotations
import subprocess, sys
from pathlib import Path

DATASETS = {
    "Reddit":     "data/ml_reddit.csv",
    "Wikipedia":  "data/ml_wikipedia.csv",
}

SEEDS = [1, 2, 3]

# TRW hyperparams (primary protocol, transductive)
WALKS  = 16
WALK_L = 6
ALPHA  = 0.4

USE_N2V = True  # set False if you want to disable

OUT_DIR = Path("results/nodecls_primary")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "runs.csv"

def run_one(csv, seed):
    cmd = [
        sys.executable, "-m", "runners.run_nodecls_primary",
        "--csv", csv,
        "--walks", str(WALKS),
        "--alpha", str(ALPHA),
        "--walk_L", str(WALK_L),
        "--seed", str(seed),
        "--save_csv", str(OUT_CSV),
    ]
    if USE_N2V:
        cmd.append("--use_node2vec")
    subprocess.run(cmd, check=True)

def main():
    # (re)create the CSV header on first run
    if OUT_CSV.exists():
        OUT_CSV.unlink()

    for ds_name, csv in DATASETS.items():
        for seed in SEEDS:
            print(f">>> {ds_name} | transductive | seed={seed}")
            run_one(csv, seed)

if __name__ == "__main__":
    main()
