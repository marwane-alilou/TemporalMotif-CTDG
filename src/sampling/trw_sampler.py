# src/sampling/trw_sampler.py
import networkx as nx
import numpy as np
import random

class TemporalRandomWalkSampler:
    def __init__(self, edges_df=None, num_walks=3, alpha=0.5):
        """
        edges_df: DataFrame with columns ['src','dst','t'] int64, sorted by t.
        """
        self.num_walks = num_walks
        self.alpha = alpha
        self.temporal_network = nx.Graph()
        if edges_df is not None:
            self._load_from_df(edges_df)

    def _load_from_df(self, df):
        for s, d, t in zip(df["src"].values, df["dst"].values, df["t"].values):
            # if multi-edges exist, keep the earliest time (or store min)
            if self.temporal_network.has_edge(s, d):
                old_t = self.temporal_network[s][d].get("time", t)
                if t < old_t:
                    self.temporal_network[s][d]["time"] = int(t)
            else:
                self.temporal_network.add_edge(int(s), int(d), time=int(t))

    def sample_temporal_random_walks(self, L=7):
        node_sets = {}
        for node in self.temporal_network.nodes():
            node_sets[node] = [self._temporal_walk(node, L=L) for _ in range(self.num_walks)]
        return node_sets

    def _temporal_walk(self, start_node, L=7):
        G = self.temporal_network
        current_node = start_node
        walk = [current_node]
        current_time = -1  # allow the earliest edge
        for _ in range(L):
            neighbors = []
            for nb in G.neighbors(current_node):
                t = G.edges[current_node, nb]['time']
                if t > current_time:
                    neighbors.append((nb, t))
            if not neighbors:
                break
            # softmax over recency bias
            times = np.array([t for _, t in neighbors], dtype=np.float64)
            weights = np.exp(-self.alpha * (times.max() - times))
            if not np.isfinite(weights).any() or weights.sum() == 0:
                next_node, t_next = random.choice(neighbors)
            else:
                weights = weights / weights.sum()
                idx = np.random.choice(len(neighbors), p=weights)
                next_node, t_next = neighbors[idx]
            walk.extend([int(t_next), int(next_node)])
            current_node = next_node
            current_time = t_next
        return walk
