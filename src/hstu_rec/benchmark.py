"""Throughput benchmark for the HSTU recommender.

Measures forward-pass (eval) and training-step throughput using
synthetic in-memory data — no TFRecords or real dataset required.

Usage::

    uv run benchmark --config configs/video_games.yaml

Options:
    --config   Path to YAML config (required)
    --vocab    Synthetic vocab size (default: 1000)
    --batch    Batch size override; defaults to config value
    --warmup   Warm-up steps before timing (default: 5, allows JIT to compile)
    --steps    Number of timed steps (default: 50)
"""
from __future__ import annotations

import argparse
import time


def benchmark(
    config_path: str,
    vocab_size: int = 1000,
    batch_size: int | None = None,
    warmup_steps: int = 5,
    steps: int = 50,
) -> None:
    import numpy as np
    import tensorflow as tf
    from hstu_rec.dataset import load_config
    from hstu_rec.train import build_model

    config = load_config(config_path)
    if batch_size is not None:
        config.training.batch_size = batch_size
    bs = config.training.batch_size
    seq_len = config.model.max_sequence_length

    model = build_model(
        vocab_size=vocab_size,
        max_sequence_length=seq_len,
        model_dim=config.model.model_dim,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        dropout=0.0,
        learning_rate=config.model.learning_rate,
        num_sampled=config.training.num_sampled,
        use_timestamps=config.model.use_timestamps,
    )

    rng = np.random.default_rng(42)

    def make_batch():
        ids = rng.integers(1, vocab_size + 1, size=(bs, seq_len), dtype=np.int32)
        x = {"input_ids": ids}
        if config.model.use_timestamps:
            x["timestamps"] = rng.integers(0, 10**9, size=(bs, seq_len), dtype=np.int32)
        y = rng.integers(1, vocab_size + 1, size=(bs,), dtype=np.int32)
        return x, y

    # Build the model (weight initialisation happens on first forward pass)
    first_batch, first_targets = make_batch()
    model(first_batch, training=False)
    n_params = model.count_params()

    print(f"\n{'='*60}")
    print(f"HSTU Recommender Benchmark")
    print(f"{'='*60}")
    print(f"  Parameters     : {n_params:,}")
    print(f"  vocab_size     : {vocab_size}")
    print(f"  seq_len        : {seq_len}")
    print(f"  model_dim      : {config.model.model_dim}")
    print(f"  num_heads      : {config.model.num_heads}")
    print(f"  num_layers     : {config.model.num_layers}")
    print(f"  batch_size     : {bs}")
    print(f"  use_timestamps : {config.model.use_timestamps}")
    print()

    # ------------------------------------------------------------------ eval
    print(f"[eval]  warming up ({warmup_steps} steps)...", end=" ", flush=True)
    for _ in range(warmup_steps):
        x, _ = make_batch()
        model(x, training=False)
    print("done")

    print(f"[eval]  timing ({steps} steps)...", end=" ", flush=True)
    t0 = time.perf_counter()
    for _ in range(steps):
        x, _ = make_batch()
        model(x, training=False)
    elapsed = time.perf_counter() - t0
    eval_tput = (steps * bs) / elapsed
    eval_ms = (elapsed / steps) * 1000
    print(f"{eval_tput:>10,.0f} examples/sec   ({eval_ms:.1f} ms/batch)")

    # ----------------------------------------------------------------- train
    # Wrap in a tf.data pipeline so model.fit works with the JAX backend
    dummy_ds = (
        tf.data.Dataset.from_tensors((first_batch, first_targets))
        .repeat()
    )

    print(f"[train] warming up ({warmup_steps} steps)...", end=" ", flush=True)
    model.fit(dummy_ds, steps_per_epoch=warmup_steps, verbose=0)
    print("done")

    print(f"[train] timing ({steps} steps)...", end=" ", flush=True)
    t0 = time.perf_counter()
    model.fit(dummy_ds, steps_per_epoch=steps, verbose=0)
    elapsed = time.perf_counter() - t0
    train_tput = (steps * bs) / elapsed
    train_ms = (elapsed / steps) * 1000
    print(f"{train_tput:>10,.0f} examples/sec   ({train_ms:.1f} ms/batch)")

    print()
    print(f"  eval/train throughput ratio: {eval_tput / train_tput:.2f}x")
    print(f"{'='*60}\n")


def main(
    config_path: str | None = None,
    vocab_size: int | None = None,
    batch_size: int | None = None,
    warmup_steps: int | None = None,
    steps: int | None = None,
) -> None:
    if config_path is None:
        parser = argparse.ArgumentParser(description="Benchmark HSTU recommender throughput")
        parser.add_argument("--config",  required=True, help="Path to YAML config")
        parser.add_argument("--vocab",   type=int, default=1000, help="Synthetic vocab size (default: 1000)")
        parser.add_argument("--batch",   type=int, default=None, help="Batch size override")
        parser.add_argument("--warmup",  type=int, default=5,    help="Warm-up steps (default: 5)")
        parser.add_argument("--steps",   type=int, default=50,   help="Timed steps (default: 50)")
        args = parser.parse_args()
        config_path  = args.config
        vocab_size   = args.vocab
        batch_size   = args.batch
        warmup_steps = args.warmup
        steps        = args.steps

    benchmark(
        config_path=config_path,
        vocab_size=vocab_size or 1000,
        batch_size=batch_size,
        warmup_steps=warmup_steps or 5,
        steps=steps or 50,
    )
