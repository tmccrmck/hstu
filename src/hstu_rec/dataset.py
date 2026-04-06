from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable

import yaml


@dataclasses.dataclass
class DatasetConfig:
    name: str
    review_url: str
    min_interactions: int


@dataclasses.dataclass
class ModelConfig:
    vocab_size: int | None
    max_sequence_length: int
    model_dim: int
    num_heads: int
    num_layers: int
    dropout: float
    learning_rate: float


@dataclasses.dataclass
class TrainingConfig:
    batch_size: int
    train_steps: int
    steps_per_eval: int
    steps_per_loop: int
    model_dir: str
    num_sampled: int = 10000


@dataclasses.dataclass
class Config:
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig


@dataclasses.dataclass
class TFRecordDataFactory:
    """Builds a tf.data.Dataset from a TFRecord file for a given split.

    TensorFlow is imported lazily inside make() so this module stays
    importable on machines without TF installed.
    """

    path: str
    batch_size: int
    max_sequence_length: int
    is_training: bool

    def make(self) -> "tf.data.Dataset":
        import tensorflow as tf  # lazy import

        ds = tf.data.TFRecordDataset(
            tf.data.Dataset.list_files(self.path, shuffle=self.is_training),
            num_parallel_reads=tf.data.AUTOTUNE,
        )
        parse_fn = parse_tfrecord_fn(self.max_sequence_length)
        ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
        if self.is_training:
            ds = ds.shuffle(buffer_size=10_000).repeat()
        # drop_remainder only during training (fixed batch shapes for XLA);
        # eval/test must not discard the final partial batch
        return ds.batch(self.batch_size, drop_remainder=self.is_training).prefetch(tf.data.AUTOTUNE)


def parse_tfrecord_fn(max_sequence_length: int) -> Callable:
    """Return a function that deserialises one TFRecord example.

    Returns:
        fn(serialized) -> ({"input_ids": int32[max_seq_len]}, int32[])
    """
    import tensorflow as tf  # lazy import

    feature_spec = {
        "input_ids": tf.io.FixedLenFeature([max_sequence_length], tf.int64),
        "target_id": tf.io.FixedLenFeature([1], tf.int64),
    }

    def _parse(serialized: "tf.Tensor"):
        parsed = tf.io.parse_single_example(serialized, feature_spec)
        input_ids = tf.cast(parsed["input_ids"], tf.int32)
        target_id = tf.cast(tf.squeeze(parsed["target_id"], axis=0), tf.int32)
        return {"input_ids": input_ids}, target_id

    return _parse


def make_data_factory(config: Config, data_dir: str, split: str) -> TFRecordDataFactory:
    """Construct a TFRecordDataFactory for the given split.

    Args:
        config: Loaded Config object.
        data_dir: Directory containing train/val/test.tfrecord files.
        split: One of "train", "val", or "test".
    """
    return TFRecordDataFactory(
        path=f"{data_dir}/{split}.tfrecord",
        batch_size=config.training.batch_size,
        max_sequence_length=config.model.max_sequence_length,
        is_training=(split == "train"),
    )


def load_config(path: str | Path) -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(p) as f:
        raw = yaml.safe_load(f)
    return Config(
        dataset=DatasetConfig(**raw["dataset"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
    )
