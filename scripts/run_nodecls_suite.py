# scripts/run_nodecls_suite.py
from __future__ import annotations
import sys, subprocess
from pathlib import Path

# Adjust these to your actual files
DATASETS = {
    "Reddit":     "data/reddit.csv",
    "Wikipedia":  "data/wikipedia.csv",
}
SPLITS = ["transductive", "inductive_atleast1", "inductive_both"]
SEEDS  = [1,2,3]

OUT = Path("results/nodecls/runs.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)
if OUT.exists():
    OUT.unlink()

def run_one(csv, split, seed):
    print(f">>> {Path(csv).stem} | {split} | seed={seed}")
    cmd = [
        sys.executable, "-m", "runners.run_nodecls",
        "--csv", csv,
        "--split_policy", split,
        "--seed", str(seed),
        "--walks", "16", "--alpha", "0.4", "--walk_L", "6",
        "--save_csv", str(OUT),
        # keep Node2Vec off for speed/robustness; uncomment to enable:
        "--use_node2vec",
        # For inductive on small files, you can also lower the mask:
        "--mask_frac", "0.05",
    ]
    subprocess.run(cmd, check=True)

def main():
    for name, csv in DATASETS.items():
        for split in SPLITS:
            for s in SEEDS:
                run_one(csv, split, s)
    print(f"\nAll results → {OUT}")

if __name__ == "__main__":
    main()
