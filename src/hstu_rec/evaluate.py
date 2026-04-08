from __future__ import annotations

import argparse
from pathlib import Path


def evaluate(model, test_ds) -> dict[str, float]:
    """Compute HR@10 and NDCG@10 over an entire test dataset.

    Args:
        model: Compiled ``HSTURecommender``.  Called with ``training=False``
               so it returns full ``(batch, vocab_size+1)`` logits.
        test_ds: ``tf.data.Dataset`` yielding ``(x_dict, y)`` batches.
                 Should NOT be repeated — iteration stops naturally at
                 the end of the dataset.

    Returns:
        Dict with keys ``"hr@10"``, ``"ndcg@10"``, ``"n_users"``.
    """
    from hstu_rec.metrics import HRAtK, NDCGAtK

    hr = HRAtK(k=10)
    ndcg = NDCGAtK(k=10)
    n = 0

    for x, y in test_ds:
        logits = model(x, training=False)
        hr.update_state(y, logits)
        ndcg.update_state(y, logits)
        n += int(logits.shape[0])

    return {
        "hr@10": float(hr.result()),
        "ndcg@10": float(ndcg.result()),
        "n_users": n,
    }


def main(
    config_path: str | None = None,
    data_dir: str | None = None,
    model_path: str | None = None,
) -> None:
    """CLI entry point for evaluation.

    Rebuilds the model architecture from config, loads saved weights, then
    runs HR@10 and NDCG@10 over the full test split.

    Usage::

        uv run evaluate \\
            --config configs/video_games.yaml \\
            --data   data/video_games \\
            --model  runs/video_games/model.keras
    """
    if any(a is None for a in (config_path, data_dir, model_path)):
        parser = argparse.ArgumentParser(description="Evaluate HSTU recommender")
        parser.add_argument("--config", required=True, help="Path to YAML config")
        parser.add_argument("--data",   required=True, help="Directory with TFRecords")
        parser.add_argument("--model",  required=True, help="Path to saved .keras weights")
        args = parser.parse_args()
        config_path = args.config
        data_dir    = args.data
        model_path  = args.model

    import numpy as np
    from hstu_rec.dataset import load_config, make_data_factory
    from hstu_rec.train import build_model

    config = load_config(config_path)

    vocab_size_path = Path(data_dir) / "vocab_size.txt"
    if not vocab_size_path.exists():
        raise FileNotFoundError(
            f"vocab_size.txt not found in {data_dir}. Run preprocess first."
        )
    vocab_size = int(vocab_size_path.read_text().strip())

    model = build_model(
        vocab_size=vocab_size,
        max_sequence_length=config.model.max_sequence_length,
        model_dim=config.model.model_dim,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        dropout=0.0,  # no dropout at eval time
        learning_rate=config.model.learning_rate,
        num_sampled=config.training.num_sampled,
        use_timestamps=config.model.use_timestamps,
    )

    # Build weights by running a single dummy forward pass, then load checkpoint.
    dummy = {"input_ids": np.zeros((1, config.model.max_sequence_length), dtype=np.int32)}
    if config.model.use_timestamps:
        dummy["timestamps"] = np.zeros((1, config.model.max_sequence_length), dtype=np.int32)
    model(dummy, training=False)
    model.load_weights(model_path)

    test_ds = make_data_factory(config, data_dir, "test").make()
    results = evaluate(model, test_ds)

    print(f"Test users evaluated : {results['n_users']}")
    print(f"HR@10                : {results['hr@10']:.4f}")
    print(f"NDCG@10              : {results['ndcg@10']:.4f}")
