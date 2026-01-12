# runners/ablation_ssm_encoder.py
from __future__ import annotations
import argparse, os, time, random, platform
from pathlib import Path
import numpy as np
import pandas as pd
import psutil

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv1D, DepthwiseConv1D, Dense, Dropout, LayerNormalization,
    GlobalAveragePooling1D, Multiply, Activation, Add
)
from tensorflow.keras.callbacks import EarlyStopping
from scipy.sparse import csr_matrix

# project imports
from src.dataio.csv_temporal import read_temporal_csv, build_splits
from src.motifs import extract_temporal_motifs, create_incidence_matrices_sparse
from src.features_v2 import build_incidence_features_v2
from src.protocols import get_adjacency_from_edges
from src.sampling.trw_sampler import TemporalRandomWalkSampler


# ------------------------- utils: negatives & filtering -------------------------
def filter_edges_by_seen_policy(edges: np.ndarray, seen_nodes: set[int], policy: str) -> np.ndarray:
    if policy == "none":
        return edges
    s = set(seen_nodes)
    keep = []
    for (u, v) in edges:
        a, b = (u in s), (v in s)
        if policy == "transductive" and (a and b):
            keep.append((u, v))
        elif policy == "inductive_atleast1" and (a ^ b):    # New–Old
            keep.append((u, v))
        elif policy == "inductive_both" and (not a and not b):  # New–New
            keep.append((u, v))
    return np.array(keep, dtype=np.int64)

def sample_negatives_from_pool(
    pos_edges: np.ndarray,
    node_pool: np.ndarray,
    k: int,
    forbid: csr_matrix | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    total = pos_edges.shape[0] * k
    if total == 0:
        return np.empty((0, 2), dtype=np.int64)
    neg = np.empty((total, 2), dtype=np.int64)
    m = len(node_pool)
    i = 0
    while i < total:
        s_idx = rng.integers(0, m, size=total - i, dtype=np.int64)
        d_idx = rng.integers(0, m, size=total - i, dtype=np.int64)
        s = node_pool[s_idx]; d = node_pool[d_idx]
        ok = (s != d)
        if forbid is not None:
            ok &= (forbid[s, d].A1 == 0)
        take = np.where(ok)[0]
        fill = min(len(take), total - i)
        neg[i:i+fill, 0] = s[take][:fill]
        neg[i:i+fill, 1] = d[take][:fill]
        i += fill
    return neg

def concatenate_pair_features(edges: np.ndarray, node_features: np.ndarray, node_order: list[int]) -> np.ndarray:
    idx = {n: i for i, n in enumerate(node_order)}
    D = node_features.shape[1]
    zero = np.zeros(D, dtype=node_features.dtype)
    X = np.empty((edges.shape[0], 2 * D), dtype=np.float32)
    for k, (s, d) in enumerate(edges):
        fs = node_features[idx.get(int(s), -1)] if int(s) in idx else zero
        fd = node_features[idx.get(int(d), -1)] if int(d) in idx else zero
        if isinstance(fs, int): fs = zero
        if isinstance(fd, int): fd = zero
        X[k] = np.concatenate([fs, fd], axis=0)
    return X

def build_lp_arrays(train_pos, val_pos, test_pos, num_nodes, node_features, node_order, neg_per_pos, rng):
    node_pool = np.array(node_order, dtype=np.int64)
    forbid_adj = get_adjacency_from_edges(pd.DataFrame({"src": train_pos[:,0], "dst": train_pos[:,1]}, dtype=np.int64),
                                          num_nodes=max(node_pool)+1)
    tr_neg = sample_negatives_from_pool(train_pos, node_pool, neg_per_pos, forbid=forbid_adj, rng=rng)
    va_neg = sample_negatives_from_pool(val_pos,   node_pool, neg_per_pos, forbid=None, rng=rng)
    te_neg = sample_negatives_from_pool(test_pos,  node_pool, neg_per_pos, forbid=None, rng=rng)

    Xtr = np.vstack([concatenate_pair_features(train_pos, node_features, node_order),
                     concatenate_pair_features(tr_neg,   node_features, node_order)])
    ytr = np.hstack([np.ones(len(train_pos), dtype=int), np.zeros(len(tr_neg), dtype=int)])

    Xva = np.vstack([concatenate_pair_features(val_pos, node_features, node_order),
                     concatenate_pair_features(va_neg,  node_features, node_order)])
    yva = np.hstack([np.ones(len(val_pos), dtype=int), np.zeros(len(va_neg), dtype=int)])

    Xte = np.vstack([concatenate_pair_features(test_pos, node_features, node_order),
                     concatenate_pair_features(te_neg,  node_features, node_order)])
    yte = np.hstack([np.ones(len(test_pos), dtype=int), np.zeros(len(te_neg), dtype=int)])
    return Xtr, ytr, Xva, yva, Xte, yte


# ------------------------- temporal series + SSM encoder -------------------------
def build_binned_series(train_df: pd.DataFrame, node_order: list[int], num_bins: int) -> np.ndarray:
    """Return (N, num_bins, 1) float32 normalized series (counts per bin)."""
    tmin = int(train_df["t"].min()); tmax = int(train_df["t"].max())
    edges = np.linspace(tmin, tmax + 1, num_bins + 1, dtype=np.float64)

    N = len(node_order)
    series = np.zeros((N, num_bins, 1), dtype=np.float32)
    node_to_row = {int(n): i for i, n in enumerate(node_order)}

    ts = train_df["t"].to_numpy(np.int64)
    bins = np.clip(np.digitize(ts, edges) - 1, 0, num_bins - 1)

    for role in ("src", "dst"):
        nodes = train_df[role].to_numpy(np.int64)
        for n, b in zip(nodes, bins):
            i = node_to_row.get(int(n))
            if i is not None:
                series[i, b, 0] += 1.0

    denom = series.sum(axis=1, keepdims=True) + 1e-6
    series = series / denom
    return series

def make_ssm_encoder(num_bins: int, emb_dim: int, d_state: int, d_out: int, dropout: float) -> Model:
    """
    A small SSM-inspired encoder that is robust in TF/Keras:
    DepthwiseConv1D (FIR-like) + gated mixing + pointwise conv + pooling + Dense.
    """
    x_in = Input(shape=(num_bins, 1))
    # depthwise temporal filter bank
    h = DepthwiseConv1D(kernel_size=5, padding="same", depth_multiplier=1)(x_in)
    h = LayerNormalization()(h)
    # simple gating
    g = Conv1D(filters=d_state, kernel_size=1, padding="same", activation="sigmoid")(h)
    v = Conv1D(filters=d_state, kernel_size=1, padding="same", activation="tanh")(h)
    h = Multiply()([g, v])
    # pointwise mixing to d_out
    h = Conv1D(filters=d_out, kernel_size=1, padding="same", activation="gelu")(h)
    h = Dropout(dropout)(h)
    # global pooling over time
    h = GlobalAveragePooling1D()(h)
    # final embedding
    z = Dense(emb_dim, activation=None)(h)

    # train with simple reconstruction target-zero (it just learns to extract stable patterns)
    model = Model(inputs=x_in, outputs=z)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return model


# ------------------------- downstream models -------------------------
def make_optimizer(name: str, lr: float):
    name = name.lower()
    if name == "sgd":  return tf.keras.optimizers.SGD(learning_rate=lr)
    if name == "adam": return tf.keras.optimizers.Adam(learning_rate=lr)
    if name == "adamw":return tf.keras.optimizers.AdamW(learning_rate=lr)
    raise ValueError(f"Unknown optimizer: {name}")

def build_predictor(encoder: str, input_dim: int, activation: str, optimizer: str, lr: float) -> Model:
    opt = make_optimizer(optimizer, lr)
    activation = activation.lower()
    if encoder == "rnn":
        from tensorflow.keras.layers import SimpleRNN, Input
        x = Input(shape=(1, input_dim))
        h = SimpleRNN(128, activation=activation,
                      kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
        h = Dropout(0.5)(h)
        h = Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-3))(h)
        y = Dense(1, activation="sigmoid")(h)
        m = Model(x, y)
    elif encoder == "lstm":
        from tensorflow.keras.layers import LSTM, Input
        x = Input(shape=(1, input_dim))
        h = LSTM(128, activation=activation,
                 kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
        h = Dropout(0.5)(h)
        h = Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-3))(h)
        y = Dense(1, activation="sigmoid")(h)
        m = Model(x, y)
    else:
        # simple MLP fallback
        x = Input(shape=(1, input_dim))
        h = Dense(128, activation=activation)(x)
        h = Dropout(0.5)(h)
        h = Dense(64, activation="relu")(h)
        y = Dense(1, activation="sigmoid")(h)
        m = Model(x, y)

    m.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return m


# ------------------------------------- main -------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--split_policy", type=str, default="transductive",
                    choices=["none","transductive","inductive_atleast1","inductive_both"])
    ap.add_argument("--mask_frac", type=float, default=0.10)

    # best config knobs
    ap.add_argument("--neg_per_pos", type=int, default=1)
    ap.add_argument("--walks", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--walk_L", type=int, default=6)

    ap.add_argument("--encoder", type=str, default="rnn")
    ap.add_argument("--activation", type=str, default="tanh")
    ap.add_argument("--optimizer", type=str, default="sgd", choices=["sgd","adam","adamw"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-2)

    # SSM params
    ap.add_argument("--ssm_bins", type=int, default=64)
    ap.add_argument("--ssm_d_state", type=int, default=64)
    ap.add_argument("--ssm_d_out", type=int, default=64)
    ap.add_argument("--ssm_emb_dim", type=int, default=64)
    ap.add_argument("--ssm_epochs", type=int, default=10)
    ap.add_argument("--ssm_dropout", type=float, default=0.1)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_csv", type=str, default="")
    args = ap.parse_args()

    # deterministic seeds
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed); np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    try: tf.config.experimental.enable_op_determinism()
    except Exception: pass
    rng = np.random.default_rng(args.seed)

    t0_total = time.perf_counter()

    # 1) load + splits
    df = read_temporal_csv(args.csv)
    train_df, val_df, test_df, num_nodes = build_splits(df)

    # NeurTWs-style inductive masking
    if args.split_policy in ("inductive_atleast1","inductive_both"):
        all_nodes = pd.Index(pd.unique(df[["u","i"]].to_numpy().reshape(-1))).to_numpy(dtype=int)
        nmask = max(1, int(len(all_nodes) * args.mask_frac))
        masked = set(rng.choice(all_nodes, size=nmask, replace=False))

        train_df = train_df[(~train_df["src"].isin(masked)) & (~train_df["dst"].isin(masked))].copy()

        def _filter(df_in):
            a = df_in["src"].isin(masked); b = df_in["dst"].isin(masked)
            if args.split_policy == "inductive_atleast1":
                return df_in[(a ^ b)].copy()
            else:
                return df_in[(a & b)].copy()

        val_df  = _filter(val_df)
        test_df = _filter(test_df)

    # ensure we still have data
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

    # 2) TRWs on train graph
    t0_trw = time.perf_counter()
    sampler = TemporalRandomWalkSampler(train_df, num_walks=args.walks, alpha=args.alpha)
    node_sets = sampler.sample_temporal_random_walks(L=args.walk_L)
    t_trw = time.perf_counter() - t0_trw

    # 3) Motifs -> incidence features
    t0_motif = time.perf_counter()
    all_walks = [w for walks in node_sets.values() for w in walks]
    motifs = extract_temporal_motifs(all_walks, [3,4,5,6])
    incidence = create_incidence_matrices_sparse(motifs)
    t_motif = time.perf_counter() - t0_motif

    # Node order from train graph
    t0_feat = time.perf_counter()
    node_order = sorted(list(sampler.temporal_network.nodes()))
    node_to_row = {int(n): i for i, n in enumerate(node_order)}
    N = len(node_order)

    Ak = []
    for k in sorted(incidence.keys()):
        A_local, vertices_k, _ = incidence[k]
        A_local = A_local.tocoo(); vertices_k = np.asarray(vertices_k, dtype=np.int64)

        map_arr = np.full(len(vertices_k), -1, dtype=np.int64)
        present = np.isin(vertices_k, list(node_to_row.keys()))
        if present.any():
            map_arr[present] = np.array([node_to_row[v] for v in vertices_k[present]], dtype=np.int64)

        valid = map_arr[A_local.row] >= 0
        rows = map_arr[A_local.row[valid]]
        cols = A_local.col[valid].astype(np.int64)
        data = A_local.data[valid].astype(np.float64)
        A_global = csr_matrix((data, (rows, cols)), shape=(N, A_local.shape[1]))
        Ak.append(A_global)

    # incidence node features
    if len(Ak) == 0:
        X_node = np.zeros((N, 1), dtype=np.float32)
    else:
        X_node, _ = build_incidence_features_v2(Ak, edge_ts=None, beta=1.0)

    # 4) Build SSM series & embeddings, then concat to features
    series = build_binned_series(train_df, node_order, num_bins=args.ssm_bins)  # (N, bins, 1)
    ssm = make_ssm_encoder(args.ssm_bins, args.ssm_emb_dim, args.ssm_d_state, args.ssm_d_out, args.ssm_dropout)
    # Train briefly (unsupervised target ~ 0 encourages small but structured embeddings)
    ssm.fit(series, np.zeros((series.shape[0], args.ssm_emb_dim), dtype=np.float32),
            epochs=args.ssm_epochs, batch_size=128, verbose=0)
    Z_ssm = ssm.predict(series, batch_size=256, verbose=0).astype(np.float32)  # (N, emb_dim)

    X_node = np.concatenate([X_node, Z_ssm], axis=1) if X_node is not None else Z_ssm
    t_feat = time.perf_counter() - t0_feat

    # policy filter for val/test
    seen = set(node_order)
    va_pos = filter_edges_by_seen_policy(va_pos, seen, args.split_policy)
    te_pos = filter_edges_by_seen_policy(te_pos, seen, args.split_policy)

    # 5) build arrays
    Xtr, ytr, Xva, yva, Xte, yte = build_lp_arrays(
        tr_pos, va_pos, te_pos, num_nodes, X_node, node_order,
        neg_per_pos=args.neg_per_pos, rng=rng
    )

    # scale & 3D for sequence encoders
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr); Xva = scaler.transform(Xva); Xte = scaler.transform(Xte)
    Xtr3 = Xtr.reshape((Xtr.shape[0], 1, Xtr.shape[1]))
    Xva3 = Xva.reshape((Xva.shape[0], 1, Xva.shape[1]))
    Xte3 = Xte.reshape((Xte.shape[0], 1, Xte.shape[1]))

    # 6) predictor
    model = build_predictor(args.encoder, Xtr.shape[1], args.activation, args.optimizer, args.lr)
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    t0_train = time.perf_counter()
    model.fit(Xtr3, ytr, validation_data=(Xva3, yva), epochs=args.epochs, batch_size=args.batch,
              callbacks=[es], verbose=1)
    t_train = time.perf_counter() - t0_train

    # 7) evaluate
    yprob = model.predict(Xte3, verbose=0).ravel()
    yhat  = (yprob > 0.5).astype(int)
    auc = float(roc_auc_score(yte, yprob)); ap = float(average_precision_score(yte, yprob))
    prec = float(precision_score(yte, yhat)); rec = float(recall_score(yte, yhat)); f1 = float(f1_score(yte, yhat))
    yprob_val = model.predict(Xva3, verbose=0).ravel()
    val_auc = float(roc_auc_score(yva, yprob_val)); val_ap = float(average_precision_score(yva, yprob_val))

    print("\n=== Test Metrics (SSM ablation) ===")
    print(pd.DataFrame({"metric":["AUC","AP","precision","recall","f1"],
                        "score":[auc, ap, prec, rec, f1]}).to_string(index=False))

    # 8) save row
    if args.save_csv:
        out = Path(args.save_csv); out.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "csv": args.csv, "split_policy": args.split_policy, "mask_frac": float(args.mask_frac),
            "encoder": args.encoder, "activation": args.activation, "optimizer": args.optimizer,
            "neg_per_pos": int(args.neg_per_pos), "walks": int(args.walks), "alpha": float(args.alpha),
            "walk_L": int(args.walk_L), "seed": int(args.seed),

            # SSM settings
            "ssm_bins": int(args.ssm_bins), "ssm_d_state": int(args.ssm_d_state),
            "ssm_d_out": int(args.ssm_d_out), "ssm_emb_dim": int(args.ssm_emb_dim),
            "ssm_epochs": int(args.ssm_epochs), "ssm_dropout": float(args.ssm_dropout),

            # sizes
            "num_nodes_train": len(node_order), "num_edges_train": len(train_df),
            "feat_dim": int(X_node.shape[1]),

            # metrics
            "auc": auc, "ap": ap, "val_auc": val_auc, "val_ap": val_ap,
            "precision": prec, "recall": rec, "f1": f1,

            # times
            "time_trw_s": t_trw, "time_motif_s": t_motif, "time_feat_s": t_feat,
            "time_train_s": t_train, "time_total_s": (time.perf_counter() - t0_total),

            # env
            "cpu": platform.processor(),
            "gpu_count": len(tf.config.list_physical_devices("GPU")),
        }
        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)


if __name__ == "__main__":
    main()
