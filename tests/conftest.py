import pandas as pd
import pytest


@pytest.fixture
def small_interactions_df():
    """5 users, each with exactly 7 interactions over 7 items. All pass 5-core."""
    rows = []
    for user_idx in range(5):
        for item_idx in range(7):
            rows.append({
                "user_id": f"user_{user_idx}",
                "parent_asin": f"item_{item_idx}",
                "timestamp": item_idx * 1000,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sparse_interactions_df(small_interactions_df):
    """Same as small_interactions_df but with an extra sparse user and sparse item."""
    sparse = pd.DataFrame([
        {"user_id": "sparse_user", "parent_asin": "item_0", "timestamp": 9999},
        {"user_id": "user_0", "parent_asin": "sparse_item", "timestamp": 9999},
    ])
    return pd.concat([small_interactions_df, sparse], ignore_index=True)
