"""TFRecord writer tests.

Tests marked @pytest.mark.slow call write_tfrecords (which imports TensorFlow)
and are skipped locally by default. Run on the GPU machine with:
    uv run pytest -m slow
"""
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def five_users_df():
    """5 users × 7 items each. All pass 5-core. Items shared across users."""
    rows = []
    for u in range(5):
        for i in range(7):
            rows.append({"user_id": f"u{u}", "parent_asin": f"ASIN{i}", "timestamp": i * 1000})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fast tests — pure Python, no TF import
# ---------------------------------------------------------------------------

def test_pad_left_short_sequence():
    from hstu_rec.preprocess.tfrecords import _pad_left
    assert _pad_left([1, 2, 3], 5) == [0, 0, 1, 2, 3]


def test_pad_left_exact_length():
    from hstu_rec.preprocess.tfrecords import _pad_left
    assert _pad_left([1, 2, 3], 3) == [1, 2, 3]


def test_pad_left_truncates_from_left():
    from hstu_rec.preprocess.tfrecords import _pad_left
    assert _pad_left([1, 2, 3, 4, 5], 3) == [3, 4, 5]


# ---------------------------------------------------------------------------
# Slow tests — call write_tfrecords which imports TensorFlow
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_write_tfrecords_creates_expected_files(five_users_df, tmp_path):
    from hstu_rec.preprocess.tfrecords import write_tfrecords
    write_tfrecords(five_users_df, max_seq_len=50, output_dir=str(tmp_path))
    assert (tmp_path / "train.tfrecord").exists()
    assert (tmp_path / "val.tfrecord").exists()
    assert (tmp_path / "test.tfrecord").exists()
    assert (tmp_path / "item_map.parquet").exists()
    assert (tmp_path / "vocab_size.txt").exists()


@pytest.mark.slow
def test_write_tfrecords_vocab_size(five_users_df, tmp_path):
    from hstu_rec.preprocess.tfrecords import write_tfrecords
    write_tfrecords(five_users_df, max_seq_len=50, output_dir=str(tmp_path))
    assert int((tmp_path / "vocab_size.txt").read_text().strip()) == 7


@pytest.mark.slow
def test_write_tfrecords_item_map(five_users_df, tmp_path):
    from hstu_rec.preprocess.tfrecords import write_tfrecords
    write_tfrecords(five_users_df, max_seq_len=50, output_dir=str(tmp_path))
    item_df = pd.read_parquet(tmp_path / "item_map.parquet")
    item_map = dict(zip(item_df["parent_asin"], item_df["item_id"]))
    assert set(item_map.keys()) == {f"ASIN{i}" for i in range(7)}
    assert min(item_map.values()) == 1
    assert max(item_map.values()) == 7
    assert len(set(item_map.values())) == 7


@pytest.mark.slow
def test_val_has_one_example_per_user(five_users_df, tmp_path):
    from hstu_rec.preprocess.tfrecords import write_tfrecords
    import tensorflow as tf
    write_tfrecords(five_users_df, max_seq_len=50, output_dir=str(tmp_path))
    ds = tf.data.TFRecordDataset(str(tmp_path / "val.tfrecord"))
    assert sum(1 for _ in ds) == 5


@pytest.mark.slow
def test_test_has_one_example_per_user(five_users_df, tmp_path):
    from hstu_rec.preprocess.tfrecords import write_tfrecords
    import tensorflow as tf
    write_tfrecords(five_users_df, max_seq_len=50, output_dir=str(tmp_path))
    ds = tf.data.TFRecordDataset(str(tmp_path / "test.tfrecord"))
    assert sum(1 for _ in ds) == 5


@pytest.mark.slow
def test_train_example_count(five_users_df, tmp_path):
    # L=7 → range(1, L-2) = range(1,5) → 4 examples per user
    from hstu_rec.preprocess.tfrecords import write_tfrecords
    import tensorflow as tf
    write_tfrecords(five_users_df, max_seq_len=50, output_dir=str(tmp_path))
    ds = tf.data.TFRecordDataset(str(tmp_path / "train.tfrecord"))
    assert sum(1 for _ in ds) == 5 * 4


@pytest.mark.slow
def test_input_ids_left_padded_and_targets_real(five_users_df, tmp_path):
    from hstu_rec.preprocess.tfrecords import write_tfrecords
    import tensorflow as tf
    write_tfrecords(five_users_df, max_seq_len=50, output_dir=str(tmp_path))
    features = {
        "input_ids": tf.io.FixedLenFeature([50], tf.int64),
        "target_id": tf.io.FixedLenFeature([1], tf.int64),
    }
    train_ds = tf.data.TFRecordDataset(str(tmp_path / "train.tfrecord"))
    first_padded = 0
    for raw in train_ds:
        parsed = tf.io.parse_single_example(raw, features)
        ids = parsed["input_ids"].numpy().tolist()
        target = int(parsed["target_id"].numpy()[0])
        assert target >= 1  # never padding token
        if ids.count(0) == 49:
            first_padded += 1
    assert first_padded == 5  # one first-example per user
