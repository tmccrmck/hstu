from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import yaml


@dataclasses.dataclass
class DatasetConfig:
    name: str
    review_url: str
    min_interactions: int


@dataclasses.dataclass
class ModelConfig:
    vocab_size: Optional[int]
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


@dataclasses.dataclass
class Config:
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig


def load_config(path: str) -> Config:
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
