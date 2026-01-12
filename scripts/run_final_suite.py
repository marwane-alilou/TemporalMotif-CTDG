# scripts/run_final_suite.py
from __future__ import annotations
import subprocess, sys, shutil
from pathlib import Path
import pandas as pd

# ---------- CONFIG ----------
BEST = dict(neg_per_pos=1, walks=16, alpha=0.4, walk_L=6)
ENCODER, ACT, OPT = "rnn", "tanh", "sgd"
SPLITS = ["transductive", "inductive_atleast1", "inductive_both"]  # transductive / New-Old / New-New
SEEDS = [1, 2, 3]
EPOCHS, BATCH, LR = 50, 32, 1e-2

# dataset labels -> file paths
DATASETS = {
    #"CollegeMsg": "data/ml_CollegeMsg.csv",
    #"Enron":      "data/ml_enron.csv",
    "MOOC":       "data/ml_mooc.csv",
}

OUT_DIR = Path("results/final_suite/mooc_3_seeds")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "runs_mooc_3_seeds.csv"     # all runs append here

# ---------- helpers ----------
def run_once(csv_path: str, dataset_name: str, split: str, seed: int, extra_tag: str = ""):
    """Call runners.run_ours for a single config/seed and append to OUT_CSV."""
    tag = f"{dataset_name}_{split}_seed{seed}{('_'+extra_tag) if extra_tag else ''}"
    print(">>>", tag)

    cmd = [
        sys.executable, "-m", "runners.run_ours",
        "--csv", csv_path,
        "--split_policy", split,
        "--encoder", ENCODER, "--activation", ACT, "--optimizer", OPT,
        "--neg_per_pos", str(BEST["neg_per_pos"]),
        "--walks", str(BEST["walks"]),
        "--alpha", str(BEST["alpha"]),
        "--walk_L", str(BEST["walk_L"]),
        "--epochs", str(EPOCHS), "--batch", str(BATCH), "--lr", str(LR),
        "--save_csv", str(OUT_CSV),
        "--seed", str(seed),
    ]
    subprocess.run(cmd, check=True)

def make_timeprefix_subsample(src_csv: str, frac: float, dst_csv: Path):
    """Chronological prefix subsample by timestamp to preserve temporal structure."""
    df = pd.read_csv(src_csv)
    # Expect columns: u,i,ts,label,idx  (we only need u,i,ts)
    df = df.sort_values("ts").reset_index(drop=True)
    n = max(1, int(len(df) * frac))
    df.head(n).to_csv(dst_csv, index=False)

def main():
    # start fresh
    if OUT_CSV.exists():
        OUT_CSV.unlink()

    # 1) All datasets × regimes × seeds (best config)
    for ds_name, csv in DATASETS.items():
        for split in SPLITS:
            for seed in SEEDS:
                # Seed handled by TF/Numpy etc. inside your runner via --seed if you wire it;
                # if not, randomness only affects negative sampling, which your runner
                # seeds via np default RNG. If you added --seed, append it here.
                run_once(csv, ds_name, split, seed)

    # 2) CollegeMsg deep-dive: scaling curves (25/50/75/100%)
    cm_src = DATASETS["CollegeMsg"]
    scal_dir = OUT_DIR / "scaling_csv"
    scal_dir.mkdir(exist_ok=True, parents=True)
    for frac in [0.25, 0.50, 0.75, 1.00]:
        dst = scal_dir / f"CollegeMsg_prefix_{int(frac*100)}.csv"
        make_timeprefix_subsample(cm_src, frac, dst)
        # one representative regime is enough for scaling (use transductive)
        for seed in SEEDS:
            run_once(str(dst), "CollegeMsg", "transductive", seed, extra_tag=f"scale{int(frac*100)}")

    print(f"\nAll runs appended to: {OUT_CSV}")

if __name__ == "__main__":
    main()
