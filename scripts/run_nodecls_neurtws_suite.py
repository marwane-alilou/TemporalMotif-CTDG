# scripts/run_nodecls_neurtws_suite.py
from __future__ import annotations
import subprocess, sys
from pathlib import Path

# Best (paper-like) defaults
WALKS, ALPHA, WALK_L = 16, 0.4, 6
EPOCHS, BATCH = 50, 64
SEEDS = [1, 2, 3]

DATASETS = {
    "Reddit":     "data/reddit.csv",
    "Wikipedia":  "data/wikipedia.csv",
}

OUT_DIR = Path("results/nodecls_neurtws")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "runs.csv"

def run_one(csv_path: str, ds_name: str, seed: int, use_node2vec: bool = True):
    print(f">>> {ds_name} | transductive | seed={seed}")
    cmd = [
        sys.executable, "-m", "runners.run_nodecls",
        "--csv", csv_path,
        "--split_policy", "transductive",     # <- NeurTWs protocol (no inductive masking)
        "--walks", str(WALKS),
        "--alpha", str(ALPHA),
        "--walk_L", str(WALK_L),
        "--epochs", str(EPOCHS),
        "--batch", str(BATCH),
        "--seed", str(seed),
        "--save_csv", str(OUT_CSV),
    ]
    if use_node2vec:
        cmd.append("--use_node2vec")
    subprocess.run(cmd, check=True)

def main():
    # start a fresh CSV
    if OUT_CSV.exists():
        OUT_CSV.unlink()

    for ds_name, csv in DATASETS.items():
        for seed in SEEDS:
            run_one(csv, ds_name, seed, use_node2vec=True)

    print(f"\nAll rows appended to: {OUT_CSV}")

if __name__ == "__main__":
    main()
