# runners/run_nodecls.py
from __future__ import annotations
import argparse, os, time, random, psutil, platform
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from scipy.sparse import csr_matrix

# local
from src.dataio.csv_temporal import read_temporal_csv, build_splits
from src.sampling.trw_sampler import TemporalRandomWalkSampler
from src.motifs import extract_temporal_motifs, create_incidence_matrices_sparse
from src.features_v2 import build_incidence_features_v2


# -------------------- utilities --------------------
def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def last_state_per_user(df: pd.DataFrame) -> pd.Series:
    """
    Return a Series mapping user (column 'src') -> last state in this split.
    Expects a column named 'state_label' (or 'label' fallback).
    """
    lab_col = "state_label" if "state_label" in df.columns else ("label" if "label" in df.columns else None)
    if lab_col is None:
        raise ValueError("CSV must contain 'state_label' (or 'label') for node classification.")
    g = df.sort_values("t").groupby("src")[lab_col].tail(1)
    # g keeps original index; make (src -> label)
    s = df.loc[g.index, ["src", lab_col]].drop_duplicates("src").set_index("src")[lab_col]
    return s.astype(int)


def make_node_features_from_trw(train_df: pd.DataFrame,
                                walks: int, alpha: float, walk_L: int,
                                use_node2vec: bool,
                                seed: int) -> tuple[np.ndarray, list[int], dict[int, int]]:
    """
    Build node features from TRAIN graph only (TRW motifs + optional Node2Vec).
    Returns (X_node, node_order, node_to_row)
    """
    sampler = TemporalRandomWalkSampler(edges_df=train_df, num_walks=walks, alpha=alpha)
    node_sets = sampler.sample_temporal_random_walks(L=walk_L)

    # Motifs / incidence on all sampled walks
    all_walks = [w for walks in node_sets.values() for w in walks]
    motif_sizes = [3, 4, 5, 6]
    motifs = extract_temporal_motifs(all_walks, motif_sizes)
    incidence = create_incidence_matrices_sparse(motifs)

    # Node order from TRAIN graph
    node_order = sorted(list(sampler.temporal_network.nodes()))
    node_to_row = {int(n): i for i, n in enumerate(node_order)}
    N = len(node_order)

    # Align incidence matrices to global node rows
    Ak = []
    for k in sorted(incidence.keys()):
        A_local, vertices_k, _ = incidence[k]  # (|V_k| x |E_k|)
        A_local = A_local.tocoo()
        vertices_k = np.asarray(vertices_k, dtype=np.int64)

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

    # Incidence features
    if len(Ak) == 0:
        X_node = np.zeros((N, 1), dtype=np.float32)
    else:
        X_node, _ = build_incidence_features_v2(Ak, edge_ts=None, beta=1.0)  # (N, D_inc)

    # Optional Node2Vec (train graph only)
    if use_node2vec:
        from node2vec import Node2Vec
        try:
            n2v = Node2Vec(sampler.temporal_network, dimensions=64,
                           walk_length=30, num_walks=200, workers=4, seed=seed)
        except TypeError:
            n2v = Node2Vec(sampler.temporal_network, dimensions=64,
                           walk_length=30, num_walks=200, workers=4)
        model = n2v.fit(window=10, min_count=1, batch_words=4)
        E = np.zeros((N, 64), dtype=np.float32)
        for i, n in enumerate(node_order):
            key = str(n) if str(n) in model.wv else n
            E[i] = model.wv[key]
        X_node = np.concatenate([X_node, E], axis=1)

    return X_node.astype(np.float32), node_order, node_to_row


def gather_split_matrix(X_node: np.ndarray, node_to_row: dict[int, int], label_s: pd.Series):
    """
    Build (X, y) for a split given labels Series 'label_s' indexed by user id.
    Only keep nodes present in node_to_row (others get dropped).
    """
    keep_nodes, y = [], []
    for u, yv in label_s.items():
        iu = node_to_row.get(int(u))
        if iu is not None:
            keep_nodes.append(iu)
            y.append(int(yv))
    if len(keep_nodes) == 0:
        return np.empty((0, X_node.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)
    X = X_node[np.array(keep_nodes, dtype=np.int64)]
    y = np.array(y, dtype=np.int64)
    return X, y


def make_mlp(input_dim: int, num_classes: int) -> tf.keras.Model:
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax" if num_classes > 2 else "sigmoid")
    ])
    loss = "sparse_categorical_crossentropy" if num_classes > 2 else "binary_crossentropy"
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss, metrics=["accuracy"])
    return model


def evaluate_probs(y_true: np.ndarray, P: np.ndarray) -> tuple[float, float, float, float]:
    """
    Return (acc, macro_f1, auc, ap). Supports binary or multi-class (macro OVR).
    P: probabilities with shape (N, C) for C>2; for binary, P.shape==(N,) or (N,1) prob of class 1.
    """
    if P.ndim == 1 or P.shape[1] == 1:  # binary
        p1 = P.ravel()
        y_pred = (p1 >= 0.5).astype(int)
        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, average="macro")
        try:
            auc = roc_auc_score(y_true, p1)
        except ValueError:
            auc = float("nan")
        try:
            ap  = average_precision_score(y_true, p1)
        except ValueError:
            ap = float("nan")
        return acc, f1, auc, ap

    # multi-class
    C = P.shape[1]
    y_pred = P.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    classes = np.unique(y_true)
    # binarize only over classes present in y_true to avoid degenerate columns
    Y_true = label_binarize(y_true, classes=classes)
    P_sel  = P[:, classes]  # align columns
    try:
        auc = roc_auc_score(Y_true, P_sel, average="macro", multi_class="ovr")
    except ValueError:
        auc = float("nan")
    try:
        ap  = average_precision_score(Y_true, P_sel, average="macro")
    except ValueError:
        ap = float("nan")
    return acc, f1, auc, ap


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--split_policy", type=str, default="transductive", choices=["transductive"],
                    help="NeurTWs node classification is transductive only.")
    ap.add_argument("--walks", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--walk_L", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--use_node2vec", action="store_true")
    ap.add_argument("--save_csv", type=str, default="")
    args = ap.parse_args()

    set_all_seeds(args.seed)

    # 1) Read + time splits (train/val/test)
    df = read_temporal_csv(args.csv)   # expects columns unified to ['src','dst','t'] (+ state_label/label)
    train_df, val_df, test_df, _ = build_splits(df)

    # 2) Build node features from TRAIN graph only
    t0_feat = time.perf_counter()
    X_node, node_order, node_to_row = make_node_features_from_trw(
        train_df, walks=args.walks, alpha=args.alpha, walk_L=args.walk_L,
        use_node2vec=args.use_node2vec, seed=args.seed
    )
    t_feat = time.perf_counter() - t0_feat

    # 3) Labels per split (last state in split per user)
    y_train_s = last_state_per_user(train_df)
    y_val_s   = last_state_per_user(val_df)
    y_test_s  = last_state_per_user(test_df)

    # 4) Assemble matrices per split (drop users without features)
    Xtr, ytr = gather_split_matrix(X_node, node_to_row, y_train_s)
    Xva, yva = gather_split_matrix(X_node, node_to_row, y_val_s)
    Xte, yte = gather_split_matrix(X_node, node_to_row, y_test_s)
    if len(Xtr) == 0 or len(Xva) == 0 or len(Xte) == 0:
        raise RuntimeError("Empty train/val/test after alignment. Check that labels exist and nodes overlap the train graph.")

    # 5) Normalize
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr); Xva = scaler.transform(Xva); Xte = scaler.transform(Xte)

    # 6) Classifier
    n_classes = int(np.unique(ytr).size)
    # If any unseen classes appear in val/test, we filter them (consistent with train-only class space)
    mask_va = np.isin(yva, np.unique(ytr)); Xva, yva = Xva[mask_va], yva[mask_va]
    mask_te = np.isin(yte, np.unique(ytr)); Xte, yte = Xte[mask_te], yte[mask_te]

    model = make_mlp(input_dim=Xtr.shape[1], num_classes=(n_classes if n_classes > 2 else 1))
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(Xtr, (ytr if n_classes > 2 else (ytr > 0).astype(int)),
              validation_data=(Xva, (yva if n_classes > 2 else (yva > 0).astype(int))),
              epochs=args.epochs, batch_size=args.batch, verbose=0, callbacks=[es])

    # 7) Evaluate
    # Probabilities
    if n_classes > 2:
        P_val = model.predict(Xva, verbose=0)
        P_te  = model.predict(Xte, verbose=0)
    else:
        p_val = model.predict(Xva, verbose=0).ravel()
        p_te  = model.predict(Xte, verbose=0).ravel()
        P_val = p_val
        P_te  = p_te

    acc_v, f1_v, auc_v, ap_v = evaluate_probs(yva, P_val)
    acc_t, f1_t, auc_t, ap_t = evaluate_probs(yte, P_te)

    print("\n=== Validation ===")
    print(f"Acc: {acc_v:.4f} | Macro-F1: {f1_v:.4f} | AUC: {auc_v:.4f} | AP: {ap_v:.4f}")
    print("=== Test ===")
    print(f"Acc: {acc_t:.4f} | Macro-F1: {f1_t:.4f} | AUC: {auc_t:.4f} | AP: {ap_t:.4f}")

    # 8) Save row
    if args.save_csv:
        out = Path(args.save_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "csv": args.csv,
            "split_policy": "transductive",
            "walks": int(args.walks),
            "alpha": float(args.alpha),
            "walk_L": int(args.walk_L),
            "use_node2vec": bool(args.use_node2vec),
            "seed": int(args.seed),
            "feat_dim": int(X_node.shape[1]),
            "num_users_train": int(len(y_train_s)),
            "num_users_val":   int(len(y_val_s)),
            "num_users_test":  int(len(y_test_s)),
            # metrics
            "val_acc": float(acc_v), "val_f1": float(f1_v), "val_auc": float(auc_v), "val_ap": float(ap_v),
            "test_acc": float(acc_t), "test_f1": float(f1_t), "test_auc": float(auc_t), "test_ap": float(ap_t),
            # timing / env (optional)
            "time_feat_s": t_feat,
            "cpu": platform.processor(),
            "gpu_count": len(tf.config.list_physical_devices("GPU")),
        }
        header = not out.exists()
        pd.DataFrame([row]).to_csv(out, mode="a", index=False, header=header)


if __name__ == "__main__":
    main()
