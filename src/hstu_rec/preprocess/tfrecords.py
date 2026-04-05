from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _pad_left(seq: list[int], max_len: int) -> list[int]:
    """Left-pad sequence with zeros to max_len, or truncate from the left."""
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq


def write_tfrecords(df: pd.DataFrame, max_seq_len: int, output_dir: str | Path) -> None:
    """Write train/val/test TFRecords and metadata from a filtered review DataFrame.

    Leave-last-out split per user:
      - test:  input = last max_seq_len items of sequence[:-1], target = sequence[-1]
      - val:   input = last max_seq_len items of sequence[:-2], target = sequence[-2]
      - train: one example per position j in range(1, L-2):
               input = last max_seq_len items of sequence[:j], target = sequence[j]

    Args:
        df: Filtered DataFrame with columns [user_id, parent_asin, timestamp].
        max_seq_len: Length of input_ids in each TFRecord example.
        output_dir: Directory to write all output files.
    """
    import tensorflow as tf  # lazy import — keeps module importable without TF on PATH

    def _make_example(input_ids: list[int], target_id: int) -> tf.train.Example:
        return tf.train.Example(features=tf.train.Features(feature={
            "input_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=input_ids)),
            "target_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[target_id])),
        }))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Assign item IDs: 0 = padding, real items start at 1 (sorted for determinism)
    unique_asins = sorted(df["parent_asin"].unique())
    item_map = {asin: idx + 1 for idx, asin in enumerate(unique_asins)}
    vocab_size = len(item_map)

    (out / "item_map.json").write_text(json.dumps(item_map))
    (out / "vocab_size.txt").write_text(str(vocab_size))

    train_writer = tf.io.TFRecordWriter(str(out / "train.tfrecord"))
    val_writer = tf.io.TFRecordWriter(str(out / "val.tfrecord"))
    test_writer = tf.io.TFRecordWriter(str(out / "test.tfrecord"))

    for _user_id, user_df in df.groupby("user_id"):
        seq = [
            item_map[asin]
            for asin in user_df.sort_values("timestamp")["parent_asin"]
        ]
        L = len(seq)

        test_writer.write(
            _make_example(_pad_left(seq[:-1], max_seq_len), seq[-1]).SerializeToString()
        )
        val_writer.write(
            _make_example(_pad_left(seq[:-2], max_seq_len), seq[-2]).SerializeToString()
        )
        for j in range(1, L - 2):
            train_writer.write(
                _make_example(_pad_left(seq[:j], max_seq_len), seq[j]).SerializeToString()
            )

    train_writer.close()
    val_writer.close()
    test_writer.close()
