import pytest
from pathlib import Path
from hstu_rec.dataset import (
    Config, DatasetConfig, ModelConfig, TrainingConfig,
    TFRecordDataFactory, load_config, make_data_factory,
)


@pytest.fixture
def video_games_config_path():
    return Path(__file__).parent.parent / "configs" / "video_games.yaml"


def test_load_config_returns_config_instance(video_games_config_path):
    config = load_config(video_games_config_path)
    assert isinstance(config, Config)
    assert isinstance(config.dataset, DatasetConfig)
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.training, TrainingConfig)


def test_load_config_dataset_fields(video_games_config_path):
    config = load_config(video_games_config_path)
    assert config.dataset.name == "video_games"
    assert "Video_Games.jsonl.gz" in config.dataset.review_url
    assert config.dataset.min_interactions == 5


def test_load_config_model_fields(video_games_config_path):
    config = load_config(video_games_config_path)
    assert config.model.vocab_size is None
    assert config.model.max_sequence_length == 50
    assert config.model.model_dim == 64
    assert config.model.num_heads == 4
    assert config.model.num_layers == 4
    assert config.model.dropout == 0.5
    assert config.model.learning_rate == pytest.approx(1e-3)
    assert config.model.use_timestamps is False


def test_load_config_training_fields(video_games_config_path):
    config = load_config(video_games_config_path)
    assert config.training.batch_size == 128
    assert config.training.train_steps == 10000
    assert config.training.steps_per_eval == 500
    assert config.training.steps_per_loop == 100
    assert config.training.model_dir == "runs/video_games"
    assert config.training.num_sampled == 1000


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")


# ---------------------------------------------------------------------------
# TFRecordDataFactory / make_data_factory — fast structural tests (no TF)
# ---------------------------------------------------------------------------

def test_make_data_factory_returns_factory(video_games_config_path, tmp_path):
    config = load_config(video_games_config_path)
    factory = make_data_factory(config, str(tmp_path), "train")
    assert isinstance(factory, TFRecordDataFactory)


def test_make_data_factory_path(video_games_config_path, tmp_path):
    config = load_config(video_games_config_path)
    factory = make_data_factory(config, str(tmp_path), "val")
    assert factory.path == f"{tmp_path}/val.tfrecord"


def test_make_data_factory_is_training_flag(video_games_config_path, tmp_path):
    config = load_config(video_games_config_path)
    assert make_data_factory(config, str(tmp_path), "train").is_training is True
    assert make_data_factory(config, str(tmp_path), "val").is_training is False
    assert make_data_factory(config, str(tmp_path), "test").is_training is False


def test_make_data_factory_batch_size(video_games_config_path, tmp_path):
    config = load_config(video_games_config_path)
    factory = make_data_factory(config, str(tmp_path), "train")
    assert factory.batch_size == config.training.batch_size
    assert factory.max_sequence_length == config.model.max_sequence_length


# ---------------------------------------------------------------------------
# Slow tests — parse_tfrecord_fn and TFRecordDataFactory.make() require TF
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_parse_tfrecord_fn_output_types(video_games_config_path, tmp_path):
    """Round-trip: write one TFRecord, parse it, check dtypes and shapes."""
    import pandas as pd
    import tensorflow as tf
    from hstu_rec.preprocess.tfrecords import write_tfrecords
    from hstu_rec.dataset import parse_tfrecord_fn

    config = load_config(video_games_config_path)
    config.model.max_sequence_length = 4
    rows = [{"user_id": f"u{u}", "parent_asin": f"A{i}", "timestamp": i}
            for u in range(3) for i in range(5)]
    write_tfrecords(pd.DataFrame(rows), max_seq_len=4, output_dir=str(tmp_path))

    parse_fn = parse_tfrecord_fn(4)
    ds = tf.data.TFRecordDataset(str(tmp_path / "val.tfrecord"))
    for raw in ds.take(1):
        x, y = parse_fn(raw)
        assert x["input_ids"].dtype == tf.int32
        assert x["input_ids"].shape == (4,)
        assert y.dtype == tf.int32
        assert y.shape == ()


@pytest.mark.slow
def test_factory_make_yields_correct_shapes(video_games_config_path, tmp_path):
    import pandas as pd
    import tensorflow as tf
    from hstu_rec.preprocess.tfrecords import write_tfrecords

    config = load_config(video_games_config_path)
    config.model.max_sequence_length = 4
    config.training.batch_size = 2
    rows = [{"user_id": f"u{u}", "parent_asin": f"A{i}", "timestamp": i}
            for u in range(3) for i in range(5)]
    write_tfrecords(pd.DataFrame(rows), max_seq_len=4, output_dir=str(tmp_path))

    factory = make_data_factory(config, str(tmp_path), "val")
    ds = factory.make()
    for x_batch, y_batch in ds.take(1):
        assert x_batch["input_ids"].shape == (2, 4)
        assert y_batch.shape == (2,)
        assert x_batch["input_ids"].dtype == tf.int32
        assert y_batch.dtype == tf.int32
