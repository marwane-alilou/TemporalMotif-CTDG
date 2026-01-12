# scripts/sweep_grid.py
from __future__ import annotations
import itertools
import subprocess
import sys
from pathlib import Path

# ---- configure your grid here ----
CSV = "data/ml_CollegeMsg.csv"         # change as needed
RESULTS_CSV = "results/sweep_results.csv"

encoders = {
    "rnn":        {"activations": ["tanh", "relu"], "optimizers": ["sgd", "adam"]},
    "lstm":       {"activations": ["tanh", "relu"], "optimizers": ["sgd", "adam"]},
    "transformer":{"activations": ["relu", "gelu"], "optimizers": ["adam", "adamw"]},
    "hgnn":       {"activations": ["tanh", "relu"], "optimizers": ["adam", "sgd"]},
}

# Hyperparameters to sweep (edit to your desired grid)
neg_per_pos_list = [3]
walks_list       = [16]
alpha_list       = [0.5]
walk_L_list      = [2,3]

# Evaluation regime
split_policy = "transductive"  # strict transductive; change to "inductive_atleast1" if needed
epochs = 30
batch  = 32
lr     = 1e-2
no_node2vec = False   # set True to run the Node2Vec-off ablation sweep

def main():
    Path("results").mkdir(exist_ok=True, parents=True)
    # remove old CSV if present to start fresh header
    out = Path(RESULTS_CSV)
    if out.exists():
        out.unlink()

    # iterate encoders and their local act/opt combos
    for enc, cfg in encoders.items():
        acts = cfg["activations"]
        opts = cfg["optimizers"]
        for act in acts:
            for opt in opts:
                for neg_per_pos, walks, alpha, walk_L in itertools.product(
                    neg_per_pos_list, walks_list, alpha_list, walk_L_list
                ):
                    cmd = [
                        sys.executable, "-m", "runners.run_ours",
                        "--csv", CSV,
                        "--split_policy", split_policy,
                        "--encoder", enc,
                        "--activation", act,
                        "--optimizer", opt,
                        "--neg_per_pos", str(neg_per_pos),
                        "--walks", str(walks),
                        "--alpha", str(alpha),
                        "--walk_L", str(walk_L),
                        "--epochs", str(epochs),
                        "--batch", str(batch),
                        "--lr", str(lr),
                        "--save_csv", RESULTS_CSV
                    ]
                    if no_node2vec:
                        cmd.append("--no_node2vec")
                    print(">>>", " ".join(cmd))
                    subprocess.run(cmd, check=True)

    print(f"\nSweep complete. Aggregated results at: {RESULTS_CSV}")

if __name__ == "__main__":
    main()
