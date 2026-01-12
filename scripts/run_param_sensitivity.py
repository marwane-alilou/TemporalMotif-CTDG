# scripts/run_param_sensitivity.py
from __future__ import annotations
import subprocess, sys
from pathlib import Path

# ----------- CONFIG -----------
DATASETS = {
    "CollegeMsg": "data/ml_CollegeMsg.csv",
    "Enron":       "data/ml_enron.csv",
}
SPLIT = "transductive"
SEED  = 1

# Base (kept constant unless we sweep it)
BASE = dict(neg_per_pos=1, walks=16, walk_L=6, alpha=0.4)
ENCODER, ACT, OPT = "rnn", "tanh", "sgd"
EPOCHS, BATCH, LR = 50, 32, 1e-2

# Sweeps you asked for
SWEEP_ALPHA       = [0.4, 0.5, 0.6]
SWEEP_WALKS       = [16, 32, 64]
SWEEP_WALK_L      = [1, 2, 3, 4, 5, 6]
SWEEP_NEG_PER_POS = [1, 3, 6, 9, 18]

OUT_DIR = Path("results/sensitivity")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "runs.csv"     # Append all runs here

def run_one(dataset_name: str, csv_path: str, tag: str, cfg: dict):
    # cfg carries neg_per_pos / walks / walk_L / alpha
    cmd = [
        sys.executable, "-m", "runners.run_ours",
        "--csv", csv_path,
        "--split_policy", SPLIT,
        "--encoder", ENCODER, "--activation", ACT, "--optimizer", OPT,
        "--epochs", str(EPOCHS), "--batch", str(BATCH), "--lr", str(LR),
        "--neg_per_pos", str(cfg["neg_per_pos"]),
        "--walks", str(cfg["walks"]),
        "--walk_L", str(cfg["walk_L"]),
        "--alpha", str(cfg["alpha"]),
        "--seed", str(SEED),
        "--save_csv", str(OUT_CSV),
        "--tag", tag,
    ]
    print(">>>", dataset_name, tag, cfg)
    subprocess.run(cmd, check=True)

def main():
    # start fresh
    if OUT_CSV.exists():
        OUT_CSV.unlink()

    for ds_name, csv in DATASETS.items():
        # 1) alpha sweep (walks=16, walk_L=6, neg_per_pos=1 constant)
        for a in SWEEP_ALPHA:
            cfg = dict(BASE); cfg["alpha"] = a
            tag = f"sweep_alpha_a{a}"
            run_one(ds_name, csv, tag, cfg)

        # 2) walks sweep (alpha=0.4, walk_L=6, neg_per_pos=1 constant)
        for w in SWEEP_WALKS:
            cfg = dict(BASE); cfg["walks"] = w
            tag = f"sweep_walks_c{w}"
            run_one(ds_name, csv, tag, cfg)

        # 3) walk_L sweep (alpha=0.4, walks=16, neg_per_pos=1 constant)
        for L in SWEEP_WALK_L:
            cfg = dict(BASE); cfg["walk_L"] = L
            tag = f"sweep_walkL_L{L}"
            run_one(ds_name, csv, tag, cfg)

        # 4) neg_per_pos sweep (walks=16, walk_L=6, alpha=0.4 constant)
        for k in SWEEP_NEG_PER_POS:
            cfg = dict(BASE); cfg["neg_per_pos"] = k
            tag = f"sweep_neg_k{k}"
            run_one(ds_name, csv, tag, cfg)

    print(f"\nAll runs written to: {OUT_CSV}")

if __name__ == "__main__":
    main()
