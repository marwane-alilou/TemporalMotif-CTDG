# src/sampling/trw_sampler_wo_bias.py
from __future__ import annotations
import numpy as np
import networkx as nx

class TemporalRandomWalkSampler:
    """
    Temporal random walks WITHOUT time-bias (uniform over valid future edges).
    Supports per-walk, center-anchored anonymization for ablations.
    """
    def __init__(self, edges_df, num_walks: int = 3, alpha: float = 0.0,
                 per_walk_anonymize: bool = False, **kwargs):
        self.num_walks = int(num_walks)
        self.alpha = float(alpha)  # ignored (kept for interface compatibility)
        self.per_walk_anonymize = bool(per_walk_anonymize)

        # Build undirected temporal graph with edge timestamps
        self.temporal_network = nx.Graph()
        for _, r in edges_df.iterrows():
            u, v, ts = int(r["src"]), int(r["dst"]), int(r["t"])
            self.temporal_network.add_edge(u, v, time=ts)

    def sample_temporal_random_walks(self, L: int = 7):
        node_sets = {}
        for node in self.temporal_network.nodes():
            walks = []
            for _ in range(self.num_walks):
                w = self._temporal_walk(self.temporal_network, start_node=node, L=L)
                if self.per_walk_anonymize and len(w) > 0:
                    w = self._anonymize_walk_centered(w)
                walks.append(w)
            node_sets[node] = walks
        return node_sets

    def _temporal_walk(self, G, start_node, L=7):
        cur = start_node
        walk = [cur]
        cur_t = 0
        for _ in range(1, L + 1):
            nbrs = [(nbr, G.edges[cur, nbr]['time']) for nbr in G.neighbors(cur)]
            valid = [(nbr, t) for nbr, t in nbrs if t > cur_t]
            if not valid:
                break
            # UNBIASED: choose uniformly among valid edges
            idx = np.random.randint(len(valid))
            nxt, t_nxt = valid[idx]
            walk.append(t_nxt)
            walk.append(nxt)
            cur = nxt
            cur_t = t_nxt
        return walk

    @staticmethod
    def _anonymize_walk_centered(walk):
        """
        Center-anchored per-walk anonymization:
        node tokens are at even positions (0,2,4,...) and are remapped per-walk.
        """
        mapping = {}
        next_id = 0
        out = []
        for i, tok in enumerate(walk):
            if i % 2 == 0:
                n = int(tok)
                if n not in mapping:
                    mapping[n] = next_id
                    next_id += 1
                out.append(mapping[n])
            else:
                out.append(tok)  # keep timestamps
        return out
