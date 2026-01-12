# src/dataio/csv_temporal.py
from __future__ import annotations
import pandas as pd
import numpy as np

# canonical names we want
U_CANON, I_CANON, TS_CANON = "u", "i", "ts"

# accepted aliases → canonical
ALIAS_MAP = {
    # users
    "u": U_CANON, "user": U_CANON, "user_id": U_CANON, "src": U_CANON,
    # items
    "i": I_CANON, "item": I_CANON, "item_id": I_CANON, "dst": I_CANON,
    # timestamps
    "ts": TS_CANON, "time": TS_CANON, "timestamp": TS_CANON, "t": TS_CANON,
    # state label (kept as-is)
    "state_label": "state_label", "state": "state_label", "label": "state_label", "y": "state_label",
    # optional extras (we keep but don’t rely on them)
    "comma_separated_list_of_features": "comma_separated_list_of_features",
    "features": "comma_separated_list_of_features",
}

REQUIRED = [U_CANON, I_CANON, TS_CANON]

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_lower = [c.strip().lower() for c in df.columns]
    rename = {}
    for orig, low in zip(df.columns, cols_lower):
        if low in ALIAS_MAP:
            rename[orig] = ALIAS_MAP[low]
    if rename:
        df = df.rename(columns=rename)
    return df

def read_temporal_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_columns(df)

    # Validate required columns
    for c in REQUIRED:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column '{c}'. Found: {df.columns.tolist()}")

    # typing
    df[U_CANON]  = df[U_CANON].astype(np.int64)
    df[I_CANON]  = df[I_CANON].astype(np.int64)

    # timestamps may be float; cast safely to int64
    if np.issubdtype(df[TS_CANON].dtype, np.number):
        df[TS_CANON] = df[TS_CANON].astype(np.int64)
    else:
        # try parsing datetime-like strings
        df[TS_CANON] = pd.to_datetime(df[TS_CANON]).astype(np.int64) // 10**9

    if "state_label" in df.columns:
        df["state_label"] = df["state_label"].astype(np.int64)

    if "idx" in df.columns:
        df["idx"] = df["idx"].astype(np.int64)

    # return in ascending time
    return df.sort_values(TS_CANON).reset_index(drop=True)

def build_splits(df: pd.DataFrame):
    """
    Returns (train_df, val_df, test_df, num_nodes).
    Uses timestamp quantiles 70/85/100 as in the rest of the codebase.
    Output columns are standardized to ['src','dst','t'] for downstream code.
    """
    # map to unified names for downstream
    out = df.copy()
    out = out.rename(columns={U_CANON: "src", I_CANON: "dst", TS_CANON: "t"})

    q1 = out["t"].quantile(0.70)
    q2 = out["t"].quantile(0.85)
    tr = out[out["t"] <= q1]
    va = out[(out["t"] > q1) & (out["t"] <= q2)]
    te = out[out["t"] > q2]

    tr = tr.sort_values("t").reset_index(drop=True)
    va = va.sort_values("t").reset_index(drop=True)
    te = te.sort_values("t").reset_index(drop=True)

    num_nodes = int(max(df[U_CANON].max(), df[I_CANON].max())) + 1
    return tr, va, te, num_nodes
