# scripts/run_ablation_per_walk_anon_seeds.py
from __future__ import annotations
import subprocess, sys
from pathlib import Path

# --------------------------- CONFIG ---------------------------
# Best config you’ve been using
BEST = dict(neg_per_pos=1, walks=16, alpha=0.4, walk_L=6)
ENCODER, ACT, OPT = "rnn", "tanh", "sgd"
EPOCHS, BATCH, LR = 50, 32, 1e-2

# Inductive regimes for the ablation (New-Old / New-New)
SPLITS = ["inductive_atleast1", "inductive_both"]

# Run on these datasets
DATASETS = {
    "CollegeMsg": "data/ml_CollegeMsg.csv",
    "Enron":      "data/ml_enron.csv",
}

# Seeds to average
SEEDS = [1, 2, 3]

# Inductive masking rate (as in NeurTWs)
MASK_FRAC = 0.05

# Output CSV (appended)
OUT_CSV = Path("results/ablations/per_walk_anon_runs.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
# --------------------------------------------------------------


def run_one(csv_path: str, ds_name: str, split: str, seed: int):
    tag = f"per_walk_anon"
    print(f">>> {ds_name} {split} seed={seed} [{tag}]")
    cmd = [
        sys.executable, "-m", "runners.run_ours",
        "--csv", csv_path,
        "--split_policy", split,
        "--mask_frac", str(MASK_FRAC),

        # model/training
        "--encoder", ENCODER, "--activation", ACT, "--optimizer", OPT,
        "--epochs", str(EPOCHS), "--batch", str(BATCH), "--lr", str(LR),

        # best walk params
        "--neg_per_pos", str(BEST["neg_per_pos"]),
        "--walks", str(BEST["walks"]),
        "--alpha", str(BEST["alpha"]),
        "--walk_L", str(BEST["walk_L"]),

        # A6 ablation toggles:
        "--per_walk_anonymize",     # <- turn ON anonymization per random walk
        "--no_node2vec",            # <- turn OFF Node2Vec to isolate effect

        # bookkeeping
        "--tag", tag,
        "--seed", str(seed),
        "--save_csv", str(OUT_CSV),
    ]
    subprocess.run(cmd, check=True)


def main():
    # fresh file each time you start (comment out if you want to append across sessions)
    if OUT_CSV.exists():
        OUT_CSV.unlink()

    for ds_name, csv in DATASETS.items():
        for split in SPLITS:
            for seed in SEEDS:
                run_one(csv, ds_name, split, seed)

    print(f"\nAll rows written to: {OUT_CSV}")


if __name__ == "__main__":
    main()
