from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def filter_reviews(jsonl_path: str | Path, min_interactions: int) -> pd.DataFrame:
    """Load a .jsonl review file and apply iterative k-core filtering.

    Reads only user_id, parent_asin, and timestamp fields. Repeatedly removes
    users and items with fewer than min_interactions interactions until
    convergence.

    Args:
        jsonl_path: Path to the decompressed .jsonl file.
        min_interactions: Minimum number of interactions for users and items.

    Returns:
        Filtered DataFrame with columns [user_id, parent_asin, timestamp].
    """
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append({
                "user_id": str(obj["user_id"]),
                "parent_asin": str(obj["parent_asin"]),
                "timestamp": int(obj["timestamp"]),
            })
    df = pd.DataFrame(rows, columns=["user_id", "parent_asin", "timestamp"])
    df = df.astype({"user_id": object, "parent_asin": object, "timestamp": "int64"})

    while True:
        user_counts = df["user_id"].value_counts()
        item_counts = df["parent_asin"].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        valid_items = item_counts[item_counts >= min_interactions].index
        filtered = df[
            df["user_id"].isin(valid_users) & df["parent_asin"].isin(valid_items)
        ]
        if len(filtered) == len(df):
            break
        df = filtered.reset_index(drop=True)

    return df
