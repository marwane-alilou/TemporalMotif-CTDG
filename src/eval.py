from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import ttest_ind

def compute_auc_ap(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    return {
        "auc": roc_auc_score(y_true, y_score),
        "ap":  average_precision_score(y_true, y_score),
    }

def aggregate_over_seeds(seed_metrics: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    """
    Returns mean, std for each metric across seeds.
    """
    keys = seed_metrics[0].keys()
    out = {}
    for k in keys:
        arr = np.array([m[k] for m in seed_metrics], dtype=float)
        out[k] = (float(arr.mean()), float(arr.std(ddof=1) if len(arr) > 1 else 0.0))
    return out

def significance_test(
    scores_A: List[float], scores_B: List[float], alternative: str = "two-sided"
) -> Dict[str, float]:
    """
    Welch's t-test across seeds.
    """
    t, p = ttest_ind(scores_A, scores_B, equal_var=False, alternative=alternative)
    return {"t": float(t), "p": float(p)}

# Convenience pretty-printer
def format_summary(summary: Dict[str, Tuple[float, float]]) -> str:
    return " | ".join([f"{k}: {m:.4f}±{s:.4f}" for k,(m,s) in summary.items()])
