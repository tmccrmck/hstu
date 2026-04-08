"""Evaluate module tests.

All tests require Keras/TF and are marked @pytest.mark.slow.
Run on GPU machine with: uv run pytest -m slow
"""
import pytest


@pytest.mark.slow
def test_evaluate_returns_expected_keys(tmp_path):
    """evaluate() returns hr@10, ndcg@10, and n_users."""
    import pandas as pd
    from hstu_rec.preprocess.tfrecords import write_tfrecords
    from hstu_rec.dataset import load_config, make_data_factory
    from hstu_rec.train import build_model
    from hstu_rec.evaluate import evaluate
    from pathlib import Path

    config_path = Path(__file__).parent.parent / "configs" / "video_games.yaml"
    config = load_config(config_path)
    config.model.max_sequence_length = 4
    config.model.model_dim = 16
    config.model.num_heads = 2
    config.model.num_layers = 1
    config.training.batch_size = 2
    config.training.num_sampled = 50

    rows = [{"user_id": f"u{u}", "parent_asin": f"A{i}", "timestamp": i}
            for u in range(5) for i in range(6)]
    write_tfrecords(pd.DataFrame(rows), max_seq_len=4, output_dir=str(tmp_path))

    vocab_size = int((tmp_path / "vocab_size.txt").read_text().strip())
    model = build_model(
        vocab_size=vocab_size,
        max_sequence_length=config.model.max_sequence_length,
        model_dim=config.model.model_dim,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        dropout=0.0,
        learning_rate=config.model.learning_rate,
        num_sampled=config.training.num_sampled,
    )
    test_ds = make_data_factory(config, str(tmp_path), "test").make()
    results = evaluate(model, test_ds)

    assert set(results.keys()) == {"hr@10", "ndcg@10", "n_users"}
    assert 0.0 <= results["hr@10"] <= 1.0
    assert 0.0 <= results["ndcg@10"] <= 1.0
    assert results["ndcg@10"] <= results["hr@10"]  # NDCG discounts by rank


@pytest.mark.slow
def test_evaluate_counts_all_users(tmp_path):
    """n_users covers every user in the test split (no dropped remainder)."""
    import pandas as pd
    from hstu_rec.preprocess.tfrecords import write_tfrecords
    from hstu_rec.dataset import load_config, make_data_factory
    from hstu_rec.train import build_model
    from hstu_rec.evaluate import evaluate
    from pathlib import Path

    config_path = Path(__file__).parent.parent / "configs" / "video_games.yaml"
    config = load_config(config_path)
    config.model.max_sequence_length = 4
    config.model.model_dim = 16
    config.model.num_heads = 2
    config.model.num_layers = 1
    config.training.batch_size = 4   # 7 users → 1 full batch + 1 partial
    config.training.num_sampled = 50

    n_users = 7
    rows = [{"user_id": f"u{u}", "parent_asin": f"A{i}", "timestamp": i}
            for u in range(n_users) for i in range(6)]
    write_tfrecords(pd.DataFrame(rows), max_seq_len=4, output_dir=str(tmp_path))

    vocab_size = int((tmp_path / "vocab_size.txt").read_text().strip())
    model = build_model(
        vocab_size=vocab_size,
        max_sequence_length=config.model.max_sequence_length,
        model_dim=config.model.model_dim,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        dropout=0.0,
        learning_rate=config.model.learning_rate,
        num_sampled=config.training.num_sampled,
    )
    test_ds = make_data_factory(config, str(tmp_path), "test").make()
    results = evaluate(model, test_ds)

    assert results["n_users"] == n_users


@pytest.mark.slow
def test_evaluate_with_timestamps(tmp_path):
    """evaluate() works correctly when use_timestamps=True."""
    import pandas as pd
    from hstu_rec.preprocess.tfrecords import write_tfrecords
    from hstu_rec.dataset import load_config, make_data_factory
    from hstu_rec.train import build_model
    from hstu_rec.evaluate import evaluate
    from pathlib import Path

    config_path = Path(__file__).parent.parent / "configs" / "video_games.yaml"
    config = load_config(config_path)
    config.model.max_sequence_length = 4
    config.model.model_dim = 16
    config.model.num_heads = 2
    config.model.num_layers = 1
    config.model.use_timestamps = True
    config.training.batch_size = 2
    config.training.num_sampled = 50

    rows = [{"user_id": f"u{u}", "parent_asin": f"A{i}", "timestamp": i * 100}
            for u in range(5) for i in range(6)]
    write_tfrecords(pd.DataFrame(rows), max_seq_len=4, output_dir=str(tmp_path))

    vocab_size = int((tmp_path / "vocab_size.txt").read_text().strip())
    model = build_model(
        vocab_size=vocab_size,
        max_sequence_length=config.model.max_sequence_length,
        model_dim=config.model.model_dim,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        dropout=0.0,
        learning_rate=config.model.learning_rate,
        num_sampled=config.training.num_sampled,
        use_timestamps=True,
    )
    test_ds = make_data_factory(config, str(tmp_path), "test").make()
    results = evaluate(model, test_ds)

    assert set(results.keys()) == {"hr@10", "ndcg@10", "n_users"}
    assert results["n_users"] == 5
    assert 0.0 <= results["hr@10"] <= 1.0


@pytest.mark.slow
def test_evaluate_load_weights_roundtrip(tmp_path):
    """Save weights, reload, evaluate — results are identical."""
    import numpy as np
    import pandas as pd
    from hstu_rec.preprocess.tfrecords import write_tfrecords
    from hstu_rec.dataset import load_config, make_data_factory
    from hstu_rec.train import build_model
    from hstu_rec.evaluate import evaluate
    from pathlib import Path

    config_path = Path(__file__).parent.parent / "configs" / "video_games.yaml"
    config = load_config(config_path)
    config.model.max_sequence_length = 4
    config.model.model_dim = 16
    config.model.num_heads = 2
    config.model.num_layers = 1
    config.training.batch_size = 2
    config.training.num_sampled = 50

    rows = [{"user_id": f"u{u}", "parent_asin": f"A{i}", "timestamp": i}
            for u in range(5) for i in range(6)]
    write_tfrecords(pd.DataFrame(rows), max_seq_len=4, output_dir=str(tmp_path))

    vocab_size = int((tmp_path / "vocab_size.txt").read_text().strip())

    def _fresh_model():
        return build_model(
            vocab_size=vocab_size,
            max_sequence_length=config.model.max_sequence_length,
            model_dim=config.model.model_dim,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
            dropout=0.0,
            learning_rate=config.model.learning_rate,
            num_sampled=config.training.num_sampled,
        )

    model_a = _fresh_model()
    # Build then save weights
    dummy = {"input_ids": np.zeros((1, 4), dtype=np.int32)}
    model_a(dummy, training=False)
    weights_path = str(tmp_path / "model.weights.h5")
    model_a.save_weights(weights_path)

    # Reload into a fresh model
    model_b = _fresh_model()
    model_b(dummy, training=False)
    model_b.load_weights(weights_path)

    test_ds = make_data_factory(config, str(tmp_path), "test").make()
    res_a = evaluate(model_a, test_ds)
    res_b = evaluate(model_b, test_ds)

    assert res_a["hr@10"] == pytest.approx(res_b["hr@10"])
    assert res_a["ndcg@10"] == pytest.approx(res_b["ndcg@10"])
