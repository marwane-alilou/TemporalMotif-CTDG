# runners/run_nodecls_primary.py
from __future__ import annotations
import argparse, os, random, time, platform
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

import tensorflow as tf

from src.dataio.csv_temporal import build_splits
from src.sampling.trw_sampler import TemporalRandomWalkSampler
from src.motifs import extract_temporal_motifs, create_incidence_matrices_sparse
from src.features_v2 import build_incidence_features_v2


# ---------- CSV normalization (unchanged) ----------
def _normalize_event_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    rename = {}
    if "user_id" in df.columns and "u" not in df.columns: rename["user_id"] = "u"
    if "user"    in df.columns and "u" not in df.columns: rename["user"]    = "u"
    if "src"     in df.columns and "u" not in df.columns: rename["src"]     = "u"

    if "item_id" in df.columns and "i" not in df.columns: rename["item_id"] = "i"
    if "item"    in df.columns and "i" not in df.columns: rename["item"]    = "i"
    if "dst"     in df.columns and "i" not in df.columns: rename["dst"]     = "i"

    if "timestamp" in df.columns and "ts" not in df.columns: rename["timestamp"] = "ts"
    if "time"      in df.columns and "ts" not in df.columns: rename["time"]      = "ts"

    if "state_label" in df.columns and "label" not in df.columns: rename["state_label"] = "label"
    if "y" in df.columns and "label" not in df.columns: rename["y"] = "label"

    if rename:
        df = df.rename(columns=rename)

    required = ["u", "i", "ts"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV missing required columns {missing}. Found {df.columns.tolist()}.\n"
            "Accepted variants: user_id/user/src->u, item_id/item/dst->i, timestamp/time->ts, "
            "optional state_label/label->label."
        )

    df["u"] = df["u"].astype(np.int64)
    df["i"] = df["i"].astype(np.int64)
    df["ts"] = df["ts"].astype(np.int64)
    if "label" in df.columns:
        df["label"] = df["label"].astype(np.int64)

    return df.sort_values("ts").reset_index(drop=True)


# ---------- Label builders ----------
def _majority_label_per_node(df_train_horizon: pd.DataFrame) -> dict[int, int]:
    """Use both u and i roles; majority label in TRAIN horizon."""
    if "label" not in df_train_horizon.columns:
        return {}

    # stack (node, label) from src and dst roles
    s_src = df_train_horizon[["u", "label"]].rename(columns={"u": "node"})
    s_dst = df_train_horizon[["i", "label"]].rename(columns={"i": "node"})
    s_all = pd.concat([s_src, s_dst], axis=0, ignore_index=True)

    # drop rows without labels, if any
    s_all = s_all.dropna(subset=["label"])

    if s_all.empty:
        return {}

    # majority per node
    def _mode(x):
        return int(pd.Series.mode(x)[0])

    g = s_all.groupby("node")["label"].agg(_mode)
    return {int(k): int(v) for k, v in g.items()}


def _last_seen_label_per_node(df_train_horizon: pd.DataFrame) -> dict[int, int]:
    """Fallback: node’s last-seen label in TRAIN horizon, using both u and i."""
    if "label" not in df_train_horizon.columns:
        return {}
    df = df_train_horizon.copy()

    # For src role
    s_src = df[["ts", "u", "label"]].rename(columns={"u": "node"}).sort_values(["node", "ts"])
    last_src = s_src.groupby("node").tail(1)

    # For dst role
    s_dst = df[["ts", "i", "label"]].rename(columns={"i": "node"}).sort_values(["node", "ts"])
    last_dst = s_dst.groupby("node").tail(1)

    merged = pd.concat([last_src, last_dst], ignore_index=True).sort_values(["node", "ts"])
    last = merged.groupby("node").tail(1)
    return {int(r["node"]): int(r["label"]) for _, r in last.iterrows()}


# ---------- TRW+motif features on TRAIN ----------
def _build_node_features_on_train(train_df: pd.DataFrame,
                                  walks: int, alpha: float, walk_L: int,
                                  use_node2vec: bool,
                                  n2v_dim: int, n2v_walk_len: int, n2v_num_walks: int, n2v_workers: int,
                                  seed: int):
    t0 = time.perf_counter()
    sampler = TemporalRandomWalkSampler(edges_df=train_df, num_walks=walks, alpha=alpha)
    node_sets = sampler.sample_temporal_random_walks(L=walk_L)
    t_trw = time.perf_counter() - t0

    t0 = time.perf_counter()
    all_walks = [w for walks in node_sets.values() for w in walks]
    motif_sizes = [3, 4, 5, 6]
    motifs = extract_temporal_motifs(all_walks, motif_sizes)
    incidence = create_incidence_matrices_sparse(motifs)
    t_motif = time.perf_counter() - t0

    t0 = time.perf_counter()
    node_order = sorted(list(sampler.temporal_network.nodes()))
    node_to_row = {int(n): i for i, n in enumerate(node_order)}
    N = len(node_order)

    from scipy.sparse import csr_matrix
    Ak, motif_counts = [], {k: 0 for k in motif_sizes}
    if len(incidence) > 0:
        for k in sorted(incidence.keys()):
            A_local, vertices_k, _ = incidence[k]
            vertices_k = np.asarray(vertices_k, dtype=np.int64)

            row_map = np.full(vertices_k.shape[0], -1, dtype=np.int64)
            present = np.isin(vertices_k, list(node_to_row.keys()))
            if present.any():
                row_map[present] = np.array([node_to_row[v] for v in vertices_k[present]], dtype=np.int64)

            A = A_local.tocoo()
            valid = row_map[A.row] >= 0
            rows = row_map[A.row[valid]]
            cols = A.col[valid].astype(np.int64)
            data = A.data[valid].astype(np.float64)
            A_global = csr_matrix((data, (rows, cols)), shape=(N, A.shape[1]))
            Ak.append(A_global)
            motif_counts[k] = int(A_global.shape[1])

    if len(Ak) > 0:
        X_node, _ = build_incidence_features_v2(Ak, edge_ts=None, beta=1.0)
    else:
        X_node = np.zeros((N, 1), dtype=np.float32)

    if use_node2vec:
        try:
            from node2vec import Node2Vec
            n2v_workers = max(1, min(n2v_workers, os.cpu_count() or 1))
            kw = dict(dimensions=n2v_dim, walk_length=n2v_walk_len, num_walks=n2v_num_walks, workers=n2v_workers)
            if "seed" in Node2Vec.__init__.__code__.co_varnames:
                kw["seed"] = seed
            n2v = Node2Vec(sampler.temporal_network, **kw)
            model = n2v.fit(window=10, min_count=1, batch_words=4)
            E = np.zeros((N, kw["dimensions"]), dtype=np.float32)
            for i, n in enumerate(node_order):
                key = str(n) if str(n) in model.wv else n
                E[i] = model.wv[key]
            X_node = np.concatenate([X_node, E], axis=1)
        except Exception as e:
            print(f"[WARN] Node2Vec failed: {e}. Continuing without it.")

    t_feat = time.perf_counter() - t0
    timings = dict(time_trw_s=t_trw, time_motif_s=t_motif, time_feat_s=t_feat)
    return node_order, X_node, motif_counts, timings


# ---------- safe stratified split ----------
def _safe_stratified_split(nodes: list[int], y: np.ndarray, seed: int,
                           test_size=0.30, val_size=0.50, retries=20):
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    if len(classes) < 2:
        return None  # impossible to train a classifier

    for _ in range(retries):
        nodes_train, nodes_tmp, y_train, y_tmp = train_test_split(
            nodes, y, test_size=test_size, random_state=rng, stratify=y
        )
        if len(np.unique(y_train)) < 2:
            continue
        nodes_val, nodes_test, y_val, y_test = train_test_split(
            nodes_tmp, y_tmp, test_size=val_size, random_state=rng, stratify=y_tmp
        )
        if len(np.unique(y_val)) < 1 or len(np.unique(y_test)) < 1:
            continue
        return (nodes_train, nodes_val, nodes_test, y_train, y_val, y_test)

    return None  # give up after retries


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--walks", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--walk_L", type=int, default=6)

    ap.add_argument("--use_node2vec", action="store_true")
    ap.add_argument("--n2v_dim", type=int, default=64)
    ap.add_argument("--n2v_walk_len", type=int, default=30)
    ap.add_argument("--n2v_num_walks", type=int, default=200)
    ap.add_argument("--n2v_workers", type=int, default=4)

    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save_csv", type=str, default="")
    args = ap.parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed); np.random.seed(args.seed)

    df_raw = _normalize_event_csv(args.csv)
    train_df, val_df, test_df, _ = build_splits(df_raw)  # gives ['src','dst','t']

    # TRAIN horizon bound
    t_cut = train_df["t"].max()
    df_train_horizon = df_raw[df_raw["ts"] <= t_cut].copy()

    # 1) majority labels (u+i). If only one class, 2) last-seen labels (u+i).
    node_labels = _majority_label_per_node(df_train_horizon)
    if len(set(node_labels.values())) < 2 and "label" in df_raw.columns:
        print("[WARN] Majority labels in TRAIN produced one class; trying last-seen labels.")
        node_labels = _last_seen_label_per_node(df_train_horizon)

    # Build TRAIN features
    node_order, X_node, motif_counts, timings = _build_node_features_on_train(
        train_df=train_df, walks=args.walks, alpha=args.alpha, walk_L=args.walk_L,
        use_node2vec=args.use_node2vec,
        n2v_dim=args.n2v_dim, n2v_walk_len=args.n2v_walk_len,
        n2v_num_walks=args.n2v_num_walks, n2v_workers=args.n2v_workers,
        seed=args.seed
    )

    # No labels at all? Save features and exit.
    if not node_labels:
        print("[INFO] No labels detected; skipping probe.")
        if args.save_csv:
            out = Path(args.save_csv); out.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "csv": args.csv, "walks": args.walks, "alpha": args.alpha, "walk_L": args.walk_L,
                "use_node2vec": bool(args.use_node2vec),
                "n2v_dim": int(args.n2v_dim), "n2v_walk_len": int(args.n2v_walk_len),
                "n2v_num_walks": int(args.n2v_num_walks), "n2v_workers": int(args.n2v_workers),
                "seed": int(args.seed),
                "num_nodes_train_graph": len(node_order),
                "feat_dim": int(X_node.shape[1]),
                "motifs_k3": motif_counts.get(3, 0), "motifs_k4": motif_counts.get(4, 0),
                "motifs_k5": motif_counts.get(5, 0), "motifs_k6": motif_counts.get(6, 0),
                **timings,
                "cpu": platform.processor(),
                "gpu_count": len(tf.config.list_physical_devices('GPU')),
                "gpu_names": ";".join([getattr(d, "name", "GPU") for d in tf.config.list_physical_devices('GPU')]),
            }
            pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
        return

    # Align labels/features on nodes present in TRAIN graph
    idx_map = {n: i for i, n in enumerate(node_order)}
    labeled_nodes = [n for n in node_order if n in node_labels]
    y_all = np.array([node_labels[n] for n in labeled_nodes], dtype=int)
    X_all = np.stack([X_node[idx_map[n]] for n in labeled_nodes], axis=0)

    # Safe stratified split
    split = _safe_stratified_split(labeled_nodes, y_all, seed=args.seed, test_size=0.30, val_size=0.50)
    if split is None:
        # truly one-class after all attempts -> informative message and graceful exit
        uniq = np.unique(y_all)
        print(f"[WARN] Labeled nodes in TRAIN horizon have one class only: {uniq.tolist()}. "
              f"Cannot train a classifier. Skipping probe.")
        acc = mf1 = auc = ap = 0.0
        nodes_train = nodes_val = nodes_test = []
    else:
        nodes_train, nodes_val, nodes_test, y_train, y_val, y_test = split
        X_train = np.stack([X_node[idx_map[n]] for n in nodes_train], axis=0)
        X_val   = np.stack([X_node[idx_map[n]] for n in nodes_val],   axis=0)
        X_test  = np.stack([X_node[idx_map[n]] for n in nodes_test],  axis=0)

        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)
        X_test  = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=200, multi_class="auto", class_weight=None, random_state=args.seed)
        clf.fit(X_train, y_train)

        yhat = clf.predict(X_test)
        acc = float(accuracy_score(y_test, yhat))
        mf1 = float(f1_score(y_test, yhat, average="macro"))

        classes = np.unique(y_all)
        proba = clf.predict_proba(X_test)
        if len(classes) == 2:
            pos_col = list(clf.classes_).index(classes[1])
            auc = float(roc_auc_score(y_test, proba[:, pos_col]))
            ap  = float(average_precision_score(y_test, proba[:, pos_col]))
        else:
            aucs, aps = [], []
            for c in classes:
                y_true = (y_test == c).astype(int)
                col = list(clf.classes_).index(c)
                y_score = proba[:, col]
                try:
                    aucs.append(roc_auc_score(y_true, y_score))
                except ValueError:
                    pass
                aps.append(average_precision_score(y_true, y_score))
            auc = float(np.mean(aucs)) if len(aucs) else 0.0
            ap  = float(np.mean(aps))  if len(aps)  else 0.0

    print("\n=== Node Classification (primary) ===")
    print(f"Nodes (train/val/test) = {len(nodes_train)}/{len(nodes_val)}/{len(nodes_test)}")
    print(f"Feat dim = {X_node.shape[1]}, Motifs k3/k4/k5/k6 = "
          f"{motif_counts.get(3,0)}/{motif_counts.get(4,0)}/{motif_counts.get(5,0)}/{motif_counts.get(6,0)}")
    print(f"Acc = {acc:.4f} | Macro-F1 = {mf1:.4f} | Macro-OVR AUC = {auc:.4f} | Macro-OVR AP = {ap:.4f}")

    if args.save_csv:
        out = Path(args.save_csv); out.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "csv": args.csv, "walks": int(args.walks), "alpha": float(args.alpha), "walk_L": int(args.walk_L),
            "use_node2vec": bool(args.use_node2vec), "n2v_dim": int(args.n2v_dim),
            "n2v_walk_len": int(args.n2v_walk_len), "n2v_num_walks": int(args.n2v_num_walks),
            "n2v_workers": int(args.n2v_workers), "seed": int(args.seed),
            "num_nodes_train_graph": len(node_order), "feat_dim": int(X_node.shape[1]),
            "motifs_k3": motif_counts.get(3, 0), "motifs_k4": motif_counts.get(4, 0),
            "motifs_k5": motif_counts.get(5, 0), "motifs_k6": motif_counts.get(6, 0),
            "probe_train_nodes": len(nodes_train), "probe_val_nodes": len(nodes_val), "probe_test_nodes": len(nodes_test),
            "acc": acc, "macro_f1": mf1, "auc": auc, "ap": ap,
            **timings,
            "cpu": platform.processor(),
            "gpu_count": len(tf.config.list_physical_devices('GPU')),
            "gpu_names": ";".join([getattr(d, "name", "GPU") for d in tf.config.list_physical_devices('GPU')]),
        }
        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)


if __name__ == "__main__":
    main()
