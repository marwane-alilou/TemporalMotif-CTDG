from typing import Dict, List, Tuple
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

def extract_temporal_motifs(walks: List[List[int]], motif_sizes: List[int]) -> Dict[int, set]:
    """
    Simple extractor: from [v0, t1, v1, t2, v2, ...] take node subsequences of length 'size'
    (ignoring timestamps), sliding by two (node positions).
    Returns {size: set of tuple(node_ids)}.
    """
    motifs = {}
    for size in motif_sizes:
        size_motifs = set()
        for walk in walks:
            # node positions are 0,2,4,..
            node_seq = walk[0::2]
            for i in range(0, len(node_seq) - size + 1):
                motif = tuple(node_seq[i:i+size])
                size_motifs.add(motif)
        motifs[size] = size_motifs
    return motifs

def create_incidence_matrices_sparse(motifs: Dict[int, set]):
    """
    Returns {size: (A_k, vertices, hyperedges)} with A_k as csr_matrix (|V_k| x |E_k|).
    """
    incidence_matrices = {}
    for size, size_motifs in motifs.items():
        vertices = sorted(set(v for motif in size_motifs for v in motif))
        v_index = {v: i for i, v in enumerate(vertices)}
        num_vertices = len(vertices)
        num_hyperedges = len(size_motifs)

        A = lil_matrix((num_vertices, num_hyperedges), dtype=np.int8)
        hyperedges = []
        for j, motif in enumerate(size_motifs):
            hyperedges.append(motif)
            for v in motif:
                A[v_index[v], j] = 1
        incidence_matrices[size] = (csr_matrix(A), vertices, hyperedges)
    return incidence_matrices
