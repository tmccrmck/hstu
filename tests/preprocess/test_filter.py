import json
from pathlib import Path
import pandas as pd
import pytest
from hstu_rec.preprocess.filter import filter_reviews


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


@pytest.fixture
def dense_jsonl(tmp_path):
    """5 users × 7 items = 35 rows; all pass 5-core."""
    rows = []
    for u in range(5):
        for i in range(7):
            rows.append({"user_id": f"u{u}", "parent_asin": f"item{i}", "timestamp": i})
    path = tmp_path / "reviews.jsonl"
    _write_jsonl(path, rows)
    return path


@pytest.fixture
def sparse_jsonl(tmp_path, dense_jsonl):
    """dense_jsonl + a sparse user (1 interaction) and sparse item (1 interaction)."""
    extra = [
        {"user_id": "sparse_user", "parent_asin": "item0", "timestamp": 999},
        {"user_id": "u0", "parent_asin": "sparse_item", "timestamp": 999},
    ]
    path = tmp_path / "sparse.jsonl"
    rows = []
    with open(dense_jsonl) as f:
        for line in f:
            rows.append(json.loads(line))
    _write_jsonl(path, rows + extra)
    return path


def test_filter_reviews_returns_dataframe_with_correct_columns(dense_jsonl):
    df = filter_reviews(str(dense_jsonl), min_interactions=5)
    assert list(df.columns) == ["user_id", "parent_asin", "timestamp"]


def test_filter_reviews_passes_dense_dataset_unchanged(dense_jsonl):
    df = filter_reviews(str(dense_jsonl), min_interactions=5)
    assert df["user_id"].nunique() == 5
    assert df["parent_asin"].nunique() == 7
    assert len(df) == 35


def test_filter_reviews_removes_sparse_users_and_items(sparse_jsonl):
    df = filter_reviews(str(sparse_jsonl), min_interactions=5)
    assert "sparse_user" not in df["user_id"].values
    assert "sparse_item" not in df["parent_asin"].values


def test_filter_reviews_converges_iteratively(tmp_path):
    """Removing a sparse user may drop an item below threshold, which cascades."""
    rows = []
    # 5 users × 6 common items (survive filtering)
    for u in range(5):
        for i in range(6):
            rows.append({"user_id": f"u{u}", "parent_asin": f"item{i}", "timestamp": i})
    # 1 user with 5 interactions, but one of their items appears only for them
    rows += [
        {"user_id": "cascade_user", "parent_asin": "item0", "timestamp": 0},
        {"user_id": "cascade_user", "parent_asin": "item1", "timestamp": 1},
        {"user_id": "cascade_user", "parent_asin": "item2", "timestamp": 2},
        {"user_id": "cascade_user", "parent_asin": "item3", "timestamp": 3},
        {"user_id": "cascade_user", "parent_asin": "cascade_only_item", "timestamp": 4},
    ]
    path = tmp_path / "cascade.jsonl"
    _write_jsonl(path, rows)
    df = filter_reviews(str(path), min_interactions=5)
    assert "cascade_user" not in df["user_id"].values
    assert "cascade_only_item" not in df["parent_asin"].values


def test_filter_reviews_column_types(dense_jsonl):
    df = filter_reviews(str(dense_jsonl), min_interactions=5)
    assert df["user_id"].dtype == object   # str
    assert df["parent_asin"].dtype == object  # str
    assert pd.api.types.is_integer_dtype(df["timestamp"])
