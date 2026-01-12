from __future__ import annotations
import dataclasses as dc
from typing import Dict, Tuple, List, Optional, Iterable
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from scipy.sparse import csr_matrix

# ========== Config loading ==========

@dc.dataclass
class LPConfig:
    seeds: List[int]
    neg_per_pos: int
    num_eval_negs: int
    metrics: List[str]
    mode: str  # "transductive" | "inductive"

@dc.dataclass
class Protocol:
    seeds: List[int]
    lp_transductive: LPConfig
    lp_inductive: LPConfig
    nodecls_cfg: Dict

def load_protocol(path: str | Path) -> Protocol:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    def lp(mode: str) -> LPConfig:
        section = cfg["link_prediction"][mode]
        return LPConfig(
            seeds=cfg["seeds"],
            neg_per_pos=cfg["link_prediction"]["negatives_per_pos"],
            num_eval_negs=cfg["link_prediction"]["num_eval_negatives"],
            metrics=cfg["link_prediction"]["metrics"],
            mode=mode
        )

    return Protocol(
        seeds=cfg["seeds"],
        lp_transductive=lp("transductive"),
        lp_inductive=lp("inductive"),
        nodecls_cfg=cfg["node_classification"],
    )

# ========== Split enforcement utilities ==========

def enforce_transductive_splits(
    edges: pd.DataFrame,
    t_train_end: int, t_val_end: int, t_test_end: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    edges columns: ['src','dst','t'] (int64). Returns (train, val, test).
    """
    tr = edges[edges["t"] <= t_train_end].copy()
    va = edges[(edges["t"] > t_train_end) & (edges["t"] <= t_val_end)].copy()
    te = edges[(edges["t"] > t_val_end) & (edges["t"] <= t_test_end)].copy()
    return tr.sort_values("t"), va.sort_values("t"), te.sort_values("t")

def enforce_inductive_test_nodes(
    tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Keeps only test edges where at least one endpoint is UNSEEN in train
    (strict inductive). Adjust if you need both endpoints unseen.
    """
    seen = set(tr["src"]).union(set(tr["dst"]))
    # require at least one unseen endpoint in TEST
    mask_te = ~te["src"].isin(seen) | ~te["dst"].isin(seen)
    te_ind = te[mask_te].copy()
    # keep val as chronological; optional: enforce partially seen nodes
    return tr, va, te_ind

# ========== Negative sampling ==========

def sample_negatives(
    pos_edges: np.ndarray, num_nodes: int, k: int,
    forbid: Optional[csr_matrix] = None, rng: np.random.Generator | None = None
) -> np.ndarray:
    """
    Uniform negative sampling. If 'forbid' is given (adjacency), avoid existing edges.
    pos_edges: (N,2) ndarray
    returns (N*k, 2)
    """
    rng = rng or np.random.default_rng()
    N = pos_edges.shape[0]
    total = N * k
    neg = np.empty((total, 2), dtype=np.int64)
    i = 0
    while i < total:
        s = rng.integers(0, num_nodes, size=total - i, dtype=np.int64)
        d = rng.integers(0, num_nodes, size=total - i, dtype=np.int64)
        if forbid is not None:
            ok = (s != d) & (forbid[s, d].A1 == 0)
        else:
            ok = (s != d)
        take = np.where(ok)[0]
        to_fill = min(len(take), total - i)
        neg[i:i+to_fill, 0] = s[take][:to_fill]
        neg[i:i+to_fill, 1] = d[take][:to_fill]
        i += to_fill
    return neg

# ========== Adapter placeholders (we wire these to your code) ==========

def load_splits_from_repo() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """
    ADAPTER NEEDED:
    Return (train_df, val_df, test_df, num_nodes).
    Each df must have columns ['src','dst','t'] as int64 and sorted by 't'.
    """
    raise NotImplementedError("wire to your split loader")

def get_adjacency_from_edges(edges: pd.DataFrame, num_nodes: int) -> csr_matrix:
    r = edges["src"].to_numpy(np.int64)
    c = edges["dst"].to_numpy(np.int64)
    data = np.ones_like(r, dtype=np.int8)
    return csr_matrix((data, (r, c)), shape=(num_nodes, num_nodes))
