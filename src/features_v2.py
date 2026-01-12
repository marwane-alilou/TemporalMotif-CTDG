from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy.sparse import csr_matrix, issparse

# --------- helpers ---------
def _row_sum(X: csr_matrix) -> np.ndarray:
    return np.asarray(X.sum(axis=1)).ravel()

def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    b = np.where(b == 0, 1.0, b)
    return a / b

def _normalize_rows(X: csr_matrix) -> csr_matrix:
    r = _row_sum(X).astype(np.float64)
    r[r == 0] = 1.0
    return X.multiply(1.0 / r[:, None])

# --------- TF-IDF over motif "types" (columns) ---------
def tfidf_over_types(A: csr_matrix) -> csr_matrix:
    """
    A: incidence (nodes x hyperedges), binary or counts.
    Returns TF-IDF normalized matrix (nodes x hyperedges).
    """
    A = A.tocsr().astype(np.float64)
    tf = _normalize_rows(A)
    # idf
    df = np.asarray((A > 0).sum(axis=0)).ravel().astype(np.float64)
    n_docs = A.shape[0]
    idf = np.log((1 + n_docs) / (1 + df)) + 1.0
    return tf.multiply(idf)

# --------- Entropy over motif-type participation ---------
def type_entropy(A: csr_matrix, eps: float = 1e-12) -> np.ndarray:
    """
    Collapse per hyperedge to per-type via column groups BEFORE calling this function if needed.
    Here we assume columns already group a single "type" granularity (e.g., order-3).
    """
    P = _normalize_rows(A).astype(np.float64)
    P.data = np.clip(P.data, eps, None)
    # H_i = - sum_j p_ij * log p_ij
    # compute via element-wise on data plus row sums
    # Map data to p*logp and aggregate by rows:
    X = P.copy()
    X.data = P.data * np.log(P.data)
    neg_sum = -np.asarray(X.sum(axis=1)).ravel()
    return neg_sum  # shape (n_nodes,)

# --------- Recency-weighted counts ---------
def recency_weighted_counts(
    A: csr_matrix,
    edge_timestamps: Optional[np.ndarray],
    beta: float = 1.0
) -> np.ndarray:
    """
    A: (N x M) incidence; edge_timestamps: (M,) increasing integers.
    Returns per-node scalar: sum_j A_ij * exp(-beta * (t_max - t_j))
    """
    if edge_timestamps is None:
        return _row_sum(A).astype(np.float64)
    t = edge_timestamps.astype(np.float64)
    t_max = float(t.max()) if len(t) else 0.0
    w = np.exp(-beta * (t_max - t))  # (M,)
    # for csr, multiply columns by weights:
    Aw = A @ w
    return np.asarray(Aw).ravel()  # (N,)

# --------- Participation across orders (role/participation profile) ---------
def participation_across_orders(A_list: List[csr_matrix]) -> np.ndarray:
    """
    A_list: [A3, A4, A5, A6] etc. Returns (N x len(A_list)) with row-normalized counts per order.
    """
    counts = [ _row_sum(A) for A in A_list ]
    counts = np.stack(counts, axis=1).astype(np.float64)  # (N, K)
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return counts / row_sums

# --------- Main feature builder (drop-in) ---------
def build_incidence_features_v2(
    Ak: List[csr_matrix],
    edge_ts: Optional[List[np.ndarray]] = None,
    beta: float = 1.0
) -> Tuple[np.ndarray, List[str]]:
    """
    Inputs:
      Ak: list of incidence matrices for motif orders (e.g., [A3, A4, A5, A6]).
      edge_ts: list of arrays, same length as Ak, edge timestamps aligned to columns of each A_k.
    Outputs:
      X: (N x D) feature matrix
      names: list of feature names
    """
    assert len(Ak) >= 1
    N = Ak[0].shape[0]
    edge_ts = edge_ts or [None] * len(Ak)

    feats = []
    names = []

    # 1) Per-order TF-IDF pooled to node-level (sum over hyperedges)
    for i, A in enumerate(Ak):
        T = tfidf_over_types(A)                 # N x M_k
        v = np.asarray(T.sum(axis=1)).ravel()   # N
        feats.append(v)
        names.append(f"tfidf_sum_k{i+3}")

    # 2) Entropy over participation per order
    for i, A in enumerate(Ak):
        H = type_entropy(A)                     # N
        feats.append(H)
        names.append(f"entropy_k{i+3}")

    # 3) Recency-weighted counts per order
    for i, (A, ts) in enumerate(zip(Ak, edge_ts)):
        rw = recency_weighted_counts(A, ts, beta=beta)
        feats.append(rw)
        names.append(f"recency_weighted_count_k{i+3}")

    # 4) Participation profile across orders (role/participation)
    part = participation_across_orders(Ak)      # N x K
    for j in range(part.shape[1]):
        feats.append(part[:, j])
        names.append(f"participation_order_{j+3}")

    # 5) (Optional) Raw per-order counts (kept for ablations; not “centralities”)
    for i, A in enumerate(Ak):
        deg = _row_sum(A).astype(np.float64)
        feats.append(deg)
        names.append(f"incidence_count_k{i+3}")

    X = np.stack(feats, axis=1)  # (N, D)
    return X, names
