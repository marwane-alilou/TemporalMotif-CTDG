# scripts/sweep_stage2.py
from __future__ import annotations
import itertools, subprocess, sys
from pathlib import Path

CSV = "data/ml_CollegeMsg.csv"          # change if needed
OUT = "results/stage2_lstm.csv"
split_policy = "transductive"            # or "inductive_atleast1" for inductive runs
epochs = 50
batch  = 32
lr     = 1e-2
no_node2vec = False

# Fixed encoder
encoder = "lstm"
activation = "relu"
optimizer  = "sgd"

# Your grid
walk_L_list      = [1, 2, 3]
walks_list       = [16, 32, 64]
alpha_list       = [0.4, 0.5, 0.6]
neg_per_pos_list = [1, 3, 6, 9, 18]

def main():
    Path("results").mkdir(parents=True, exist_ok=True)
    # start with a clean file
    p = Path(OUT)
    if p.exists(): p.unlink()

    for (walk_L, walks, alpha, neg) in itertools.product(
        walk_L_list, walks_list, alpha_list, neg_per_pos_list
    ):
        cmd = [
            sys.executable, "-m", "runners.run_ours",
            "--csv", CSV,
            "--split_policy", split_policy,
            "--encoder", encoder,
            "--activation", activation,
            "--optimizer", optimizer,
            "--neg_per_pos", str(neg),
            "--walks", str(walks),
            "--alpha", str(alpha),
            "--walk_L", str(walk_L),
            "--epochs", str(epochs),
            "--batch", str(batch),
            "--lr", str(lr),
            "--save_csv", OUT
        ]
        if no_node2vec:
            cmd.append("--no_node2vec")

        print(">>>", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print(f"\nStage-2 sweep complete → {OUT}")

if __name__ == "__main__":
    main()
