# runners/run_ours.py
from __future__ import annotations
import argparse, os, time, psutil, platform, random
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LSTM, SimpleRNN, LayerNormalization,
    GlobalAveragePooling1D, MultiHeadAttention, Add
)
from tensorflow.keras.callbacks import EarlyStopping
from scipy.sparse import csr_matrix

# local
from src.dataio.csv_temporal import read_temporal_csv, build_splits
from src.motifs import extract_temporal_motifs, create_incidence_matrices_sparse
from src.features_v2 import build_incidence_features_v2
from src.protocols import get_adjacency_from_edges


# ---------------- seen-policy filters ----------------
def filter_edges_by_seen_policy(edges: np.ndarray, seen_nodes: set[int], policy: str) -> np.ndarray:
    if policy == 'none':
        return edges
    s = set(seen_nodes)
    keep = []
    for (u, v) in edges:
        a, b = (u in s), (v in s)
        if policy == 'transductive' and (a and b):
            keep.append((u, v))
        elif policy == 'inductive_atleast1' and (a ^ b):    # exactly one unseen
            keep.append((u, v))
        elif policy == 'inductive_both' and (not a and not b):
            keep.append((u, v))
    return np.array(keep, dtype=np.int64)


# ---------------- negatives ----------------
def sample_negatives_from_pool(
    pos_edges: np.ndarray,
    node_pool: np.ndarray,
    k: int,
    forbid: csr_matrix | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    total = int(pos_edges.shape[0] * k)
    if total == 0:
        return np.empty((0, 2), dtype=np.int64)
    neg = np.empty((total, 2), dtype=np.int64)
    m = len(node_pool)
    i = 0
    while i < total:
        s_idx = rng.integers(0, m, size=total - i, dtype=np.int64)
        d_idx = rng.integers(0, m, size=total - i, dtype=np.int64)
        s = node_pool[s_idx]
        d = node_pool[d_idx]
        ok = (s != d)
        if forbid is not None:
            ok &= (forbid[s, d].A1 == 0)
        take = np.where(ok)[0]
        fill = min(len(take), total - i)
        if fill == 0:
            continue
        neg[i:i+fill, 0] = s[take][:fill]
        neg[i:i+fill, 1] = d[take][:fill]
        i += fill
    return neg


def concatenate_pair_features(edges: np.ndarray, node_features: np.ndarray, node_order: list[int], concat_order: str = "src_dst") -> np.ndarray:
    idx = {n: i for i, n in enumerate(node_order)}
    D = node_features.shape[1]
    zero = np.zeros(D, dtype=node_features.dtype)
    X = np.empty((edges.shape[0], 2 * D), dtype=np.float32)
    for k, (s, d) in enumerate(edges):
        fs = node_features[idx.get(int(s), -1)] if int(s) in idx else zero
        fd = node_features[idx.get(int(d), -1)] if int(d) in idx else zero
        if isinstance(fs, int): fs = zero
        if isinstance(fd, int): fd = zero
        X[k] = np.concatenate([fs, fd] if concat_order == "src_dst" else [fd, fs], axis=0)
    return X


def build_lp_arrays(
    train_pos: np.ndarray,
    val_pos: np.ndarray,
    test_pos: np.ndarray,
    num_nodes: int,
    node_features: np.ndarray,
    node_order: list[int],
    neg_per_pos: int = 1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, ...]:
    rng = rng or np.random.default_rng()
    node_pool = np.array(node_order, dtype=np.int64)

    # forbid negatives that are true train positives
    train_df_edges = pd.DataFrame({"src": train_pos[:, 0], "dst": train_pos[:, 1]}, dtype=np.int64)
    forbid_adj = get_adjacency_from_edges(train_df_edges, num_nodes=max(node_pool) + 1)

    train_neg = sample_negatives_from_pool(train_pos, node_pool, neg_per_pos, forbid=forbid_adj, rng=rng)
    val_neg   = sample_negatives_from_pool(val_pos,   node_pool, neg_per_pos, forbid=None, rng=rng)
    test_neg  = sample_negatives_from_pool(test_pos,  node_pool, neg_per_pos, forbid=None, rng=rng)

    Xtr_pos = concatenate_pair_features(train_pos, node_features, node_order)
    Xva_pos = concatenate_pair_features(val_pos,   node_features, node_order)
    Xte_pos = concatenate_pair_features(test_pos,  node_features, node_order)

    Xtr_neg = concatenate_pair_features(train_neg, node_features, node_order)
    Xva_neg = concatenate_pair_features(val_neg,   node_features, node_order)
    Xte_neg = concatenate_pair_features(test_neg,  node_features, node_order)

    X_train = np.vstack([Xtr_pos, Xtr_neg]); y_train = np.hstack([np.ones(len(Xtr_pos), dtype=int), np.zeros(len(Xtr_neg), dtype=int)])
    X_val   = np.vstack([Xva_pos, Xva_neg]); y_val   = np.hstack([np.ones(len(Xva_pos), dtype=int), np.zeros(len(Xva_neg), dtype=int)])
    X_test  = np.vstack([Xte_pos, Xte_neg]); y_test  = np.hstack([np.ones(len(Xte_pos), dtype=int), np.zeros(len(Xte_neg), dtype=int)])
    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------- model factory ----------------
def make_optimizer(name: str, lr: float):
    name = name.lower()
    if name == "sgd":  return tf.keras.optimizers.SGD(learning_rate=lr)
    if name == "adam": return tf.keras.optimizers.Adam(learning_rate=lr)
    if name == "adamw":return tf.keras.optimizers.AdamW(learning_rate=lr)
    raise ValueError(f"Unknown optimizer: {name}")

def make_activation(name: str):
    name = name.lower()
    if name in ["tanh", "relu", "gelu", "selu", "elu", "sigmoid", "linear", "leakyrelu"]:
        return name
    raise ValueError(f"Unsupported activation: {name}")

def build_model(encoder: str, input_dim: int, activation: str, optimizer: str, lr: float):
    activation = make_activation(activation)
    opt = make_optimizer(optimizer, lr)

    if encoder == "rnn":
        model = Sequential([
            Input(shape=(1, input_dim)),
            SimpleRNN(128, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
            Dropout(0.5),
            Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
            Dense(1, activation='sigmoid')
        ])
    elif encoder == "lstm":
        model = Sequential([
            Input(shape=(1, input_dim)),
            LSTM(128, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
            Dropout(0.5),
            Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3)),
            Dense(1, activation='sigmoid')
        ])
    elif encoder == "transformer":
        x_in = Input(shape=(1, input_dim))
        h = Dense(128)(x_in)
        attn = MultiHeadAttention(num_heads=4, key_dim=32)(h, h)
        h = Add()([h, attn]); h = LayerNormalization()(h)
        h = Dense(128, activation=activation)(h)
        h = GlobalAveragePooling1D()(h)
        h = Dropout(0.5)(h)
        h = Dense(64, activation='relu')
        y = Dense(1, activation='sigmoid')(h)
        model = Model(inputs=x_in, outputs=y)
    elif encoder == "hgnn":  # simple MLP baseline
        model = Sequential([
            Input(shape=(1, input_dim)),
            Dense(128, activation=activation),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
    else:
        raise ValueError(f"Unknown encoder: {encoder}")

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--split_policy", type=str, default="transductive",
                    choices=["none", "transductive", "inductive_atleast1", "inductive_both"])
    ap.add_argument("--mask_frac", type=float, default=0.05)

    # TRW & motifs
    ap.add_argument("--neg_per_pos", type=int, default=1)
    ap.add_argument("--walks", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--walk_L", type=int, default=7)
    ap.add_argument("--motif_set", type=str, default="3,4,5,6")

    # Ablations / toggles
    ap.add_argument("--no_time_bias", action="store_true",
                    help="Use uniform (no-bias) TRW by importing trw_sampler_wo_bias if available.")
    ap.add_argument("--per_walk_anonymize", action="store_true",
                    help="Anonymize node ids independently per random walk (A6 ablation).")
    ap.add_argument("--no_node2vec", action="store_true")
    ap.add_argument("--no_incidence_feats", action="store_true")

    # Node2Vec controls
    ap.add_argument("--n2v_dim", type=int, default=64)
    ap.add_argument("--n2v_walk_len", type=int, default=30)
    ap.add_argument("--n2v_num_walks", type=int, default=200)
    ap.add_argument("--n2v_workers", type=int, default=4)

    # model
    ap.add_argument("--encoder", type=str, default="lstm", choices=["rnn","lstm","transformer","hgnn"])
    ap.add_argument("--activation", type=str, default="tanh")
    ap.add_argument("--optimizer", type=str, default="sgd", choices=["sgd","adam","adamw"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-2)

    # logging
    ap.add_argument("--save_csv", type=str, default="")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # seed
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed); np.random.seed(args.seed); tf.keras.utils.set_random_seed(args.seed)
    try: tf.config.experimental.enable_op_determinism()
    except Exception: pass
    rng = np.random.default_rng(args.seed)

    # timings
    t0_total = time.perf_counter()

    # splits
    df = read_temporal_csv(args.csv)
    train_df, val_df, test_df, num_nodes = build_splits(df)

    # NeurTWs-like inductive masking
    INDUCTIVE = args.split_policy in ("inductive_atleast1","inductive_both")
    if INDUCTIVE:
        all_nodes = pd.Index(pd.unique(df[["u","i"]].to_numpy().reshape(-1))).to_numpy(dtype=int)
        nmask = max(1, int(len(all_nodes) * args.mask_frac))
        masked = set(np.random.default_rng(args.seed).choice(all_nodes, size=nmask, replace=False))

        # train: remove any edge touching masked nodes
        train_df = train_df[(~train_df["src"].isin(masked)) & (~train_df["dst"].isin(masked))].copy()

        def _filter(df_in: pd.DataFrame) -> pd.DataFrame:
            a = df_in["src"].isin(masked); b = df_in["dst"].isin(masked)
            if args.split_policy == "inductive_atleast1":
                keep = (a ^ b)       # exactly one masked
            else:
                keep = (a & b)       # both masked
            return df_in[keep].copy()

        val_df  = _filter(val_df)
        test_df = _filter(test_df)

    # ensure non-empty splits
    if len(val_df)==0 or len(test_df)==0:
        def carve_tail(src_df, frac, at_least=1):
            if len(src_df)==0: return src_df, src_df
            n = max(at_least, int(len(src_df)*frac))
            tail = src_df.sort_values("t").tail(n)
            head = src_df.drop(tail.index)
            return head, tail
        if len(val_df)==0:  train_df, val_df  = carve_tail(train_df, 0.10, max(1, min(100, len(train_df)//10)))
        if len(test_df)==0: train_df, test_df = carve_tail(train_df, 0.10, max(1, min(100, len(train_df)//10)))

    tr_pos = train_df[["src","dst"]].to_numpy(np.int64)
    va_pos = val_df[["src","dst"]].to_numpy(np.int64)
    te_pos = test_df[["src","dst"]].to_numpy(np.int64)

    # Sampler (with switches)
    motif_set = sorted({int(x) for x in args.motif_set.split(",") if x.strip()})
    if args.no_time_bias:
        try:
            from src.sampling.trw_sampler_wo_bias import TemporalRandomWalkSampler
        except Exception:
            from src.sampling.trw_sampler import TemporalRandomWalkSampler
    else:
        from src.sampling.trw_sampler import TemporalRandomWalkSampler

    t0_trw = time.perf_counter()
    try:
        # newer signature supports per_walk_anonymize
        sampler = TemporalRandomWalkSampler(
            edges_df=train_df,
            num_walks=args.walks,
            alpha=args.alpha,
            per_walk_anonymize=args.per_walk_anonymize,
        )
    except TypeError:
        # older signature: ignore anonymization flag
        sampler = TemporalRandomWalkSampler(
            edges_df=train_df,
            num_walks=args.walks,
            alpha=args.alpha,
        )
    node_sets = sampler.sample_temporal_random_walks(L=args.walk_L)
    t_trw = time.perf_counter() - t0_trw

    # motifs & incidence
    t0_motif = time.perf_counter()
    all_walks = [w for walks in node_sets.values() for w in walks]
    motifs = extract_temporal_motifs(all_walks, motif_set)
    incidence = create_incidence_matrices_sparse(motifs)
    t_motif = time.perf_counter() - t0_motif

    # node order (train graph nodes)
    t0_feat = time.perf_counter()
    node_order = sorted(list(sampler.temporal_network.nodes()))
    node_to_row = {int(n): i for i, n in enumerate(node_order)}
    N = len(node_order)

    Ak = []
    motifs_k3 = motifs_k4 = motifs_k5 = motifs_k6 = 0
    for k in sorted(incidence.keys()):
        A_local, vertices_k, _ = incidence[k]  # (|V_k| x |E_k|)
        A_local = A_local.tocoo()
        vertices_k = np.asarray(vertices_k, dtype=np.int64)

        map_arr = np.full(len(vertices_k), -1, dtype=np.int64)
        present = np.isin(vertices_k, list(node_to_row.keys()))
        if present.any():
            map_arr[present] = np.array([node_to_row[v] for v in vertices_k[present]], dtype=np.int64)

        valid_mask = map_arr[A_local.row] >= 0
        rows = map_arr[A_local.row[valid_mask]]
        cols = A_local.col[valid_mask].astype(np.int64)
        data = A_local.data[valid_mask].astype(np.float64)
        A_global = csr_matrix((data, (rows, cols)), shape=(N, A_local.shape[1]))
        Ak.append(A_global)

        # count per k
        if   k == 3: motifs_k3 = int(A_global.shape[1])
        elif k == 4: motifs_k4 = int(A_global.shape[1])
        elif k == 5: motifs_k5 = int(A_global.shape[1])
        elif k == 6: motifs_k6 = int(A_global.shape[1])

    # features
    X_node = None
    if not args.no_incidence_feats:
        if len(Ak) == 0:
            X_node = np.zeros((N, 1), dtype=np.float32)
        else:
            X_node, _ = build_incidence_features_v2(Ak, edge_ts=None, beta=1.0)

    # Node2Vec (optional)
    if not args.no_node2vec:
        from node2vec import Node2Vec
        n2v_dim, n2v_walk_len, n2v_num_walks = args.n2v_dim, args.n2v_walk_len, args.n2v_num_walks
        n2v_workers = max(1, min(args.n2v_workers, os.cpu_count() or 1))
        try:
            if "seed" in Node2Vec.__init__.__code__.co_varnames:
                n2v = Node2Vec(sampler.temporal_network, dimensions=n2v_dim,
                               walk_length=n2v_walk_len, num_walks=n2v_num_walks,
                               workers=n2v_workers, seed=args.seed)
            else:
                n2v = Node2Vec(sampler.temporal_network, dimensions=n2v_dim,
                               walk_length=n2v_walk_len, num_walks=n2v_num_walks,
                               workers=n2v_workers)
            model_n2v = n2v.fit(window=10, min_count=1, batch_words=4)
        except Exception as e:
            print(f"[WARN] Node2Vec failed ({e}). Falling back to no_node2vec.")
            model_n2v = None

        if model_n2v is not None:
            E = np.zeros((N, n2v_dim), dtype=np.float32)
            for i, n in enumerate(node_order):
                key = str(n) if str(n) in model_n2v.wv else n
                E[i] = model_n2v.wv[key]
            X_node = E if X_node is None else np.concatenate([X_node, E], axis=1)

    # fallback features if both off
    if X_node is None:
        deg = np.zeros((N,1), dtype=np.float32)
        for (u,v) in tr_pos:
            iu = node_to_row.get(int(u)); iv = node_to_row.get(int(v))
            if iu is not None: deg[iu] += 1
            if iv is not None: deg[iv] += 1
        X_node = deg

    t_feat = time.perf_counter() - t0_feat

    # safety seen policy on val/test for inductive
    seen = set(node_order)
    if INDUCTIVE:
        va_pos = filter_edges_by_seen_policy(va_pos, seen, args.split_policy)
        te_pos = filter_edges_by_seen_policy(te_pos, seen, args.split_policy)
        assert len(va_pos) and len(te_pos), "All val/test edges filtered out."

    # arrays
    X_train, y_train, X_val, y_val, X_test, y_test = build_lp_arrays(
        tr_pos, va_pos, te_pos, num_nodes, X_node, node_order,
        neg_per_pos=args.neg_per_pos, rng=rng
    )

    # scale & reshape
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train); X_val = scaler.transform(X_val); X_test = scaler.transform(X_test)
    X_train_3d = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_val_3d   = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
    X_test_3d  = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # model
    model = build_model(args.encoder, X_train.shape[1], args.activation, args.optimizer, args.lr)
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    t0_train = time.perf_counter()
    model.fit(X_train_3d, y_train, validation_data=(X_val_3d, y_val),
              epochs=args.epochs, batch_size=args.batch, callbacks=[es], verbose=1)
    t_train = time.perf_counter() - t0_train

    # eval
    y_prob = model.predict(X_test_3d, verbose=0).ravel()
    y_hat  = (y_prob > 0.5).astype(int)
    precision = float(precision_score(y_test, y_hat))
    recall    = float(recall_score(y_test, y_hat))
    f1        = float(f1_score(y_test, y_hat))
    auc       = float(roc_auc_score(y_test, y_prob))
    ap        = float(average_precision_score(y_test, y_prob))

    y_prob_val = model.predict(X_val_3d, verbose=0).ravel()
    val_auc = float(roc_auc_score(y_val, y_prob_val))
    val_ap  = float(average_precision_score(y_val, y_prob_val))

    print("\n=== Test Metrics ===")
    print(pd.DataFrame({"Metric": ["precision","recall","f1","auc","ap"],
                        "Score": [precision,recall,f1,auc,ap]}).to_string(index=False))

    # save row
    if args.save_csv:
        out = Path(args.save_csv); out.parent.mkdir(parents=True, exist_ok=True)
        row = {
            # config
            "csv": args.csv, "split_policy": args.split_policy, "mask_frac": float(args.mask_frac),
            "encoder": args.encoder, "activation": args.activation, "optimizer": args.optimizer,
            "neg_per_pos": int(args.neg_per_pos), "walks": int(args.walks),
            "alpha": float(args.alpha), "walk_L": int(args.walk_L),
            "motif_set": ",".join(map(str, motif_set)),
            "no_time_bias": bool(args.no_time_bias),
            "per_walk_anonymize": bool(args.per_walk_anonymize),
            "no_node2vec": bool(args.no_node2vec), "no_incidence_feats": bool(args.no_incidence_feats),
            "n2v_dim": int(args.n2v_dim), "n2v_walk_len": int(args.n2v_walk_len),
            "n2v_num_walks": int(args.n2v_num_walks), "n2v_workers": int(args.n2v_workers),
            "seed": int(args.seed), "tag": args.tag,

            # sizes / features
            "num_nodes_train": int(N),
            "num_edges_train": int(len(train_df)),
            "feat_dim": int(X_node.shape[1]) if X_node is not None else 0,
            "motifs_k3": int(motifs_k3), "motifs_k4": int(motifs_k4),
            "motifs_k5": int(motifs_k5), "motifs_k6": int(motifs_k6),

            # metrics
            "precision": precision, "recall": recall, "f1": f1,
            "auc": auc, "ap": ap, "val_auc": val_auc, "val_ap": val_ap,

            # complexity
            "time_trw_s": t_trw, "time_motif_s": t_motif, "time_feat_s": t_feat,
            "time_train_s": t_train, "time_total_s": (time.perf_counter() - t0_total),
            "peak_mem_mb": float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)),

            # hardware snapshot
            "cpu": platform.processor(),
            "gpu_count": len(tf.config.list_physical_devices('GPU')),
            "gpu_names": ";".join([getattr(d, "name", "GPU") for d in tf.config.list_physical_devices('GPU')]),
        }
        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)


if __name__ == "__main__":
    main()
