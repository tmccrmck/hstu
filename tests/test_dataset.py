import os
import pytest
import tempfile
import yaml
from hstu_rec.dataset import Config, DatasetConfig, ModelConfig, TrainingConfig, load_config


@pytest.fixture
def video_games_config_path():
    return os.path.join(os.path.dirname(__file__), "..", "configs", "video_games.yaml")


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


def test_load_config_training_fields(video_games_config_path):
    config = load_config(video_games_config_path)
    assert config.training.batch_size == 128
    assert config.training.train_steps == 10000
    assert config.training.steps_per_eval == 500
    assert config.training.steps_per_loop == 100
    assert config.training.model_dir == "runs/video_games"


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")
