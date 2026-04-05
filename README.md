# HSTU Recommender

Next-item recommendation using Google's [RecML](https://github.com/AI-Hypercomputer/RecML) HSTU implementation on the [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) dataset.

## Quick start

```bash
# Install dependencies
uv sync

# Set environment (JAX backend for Keras, RecML on PYTHONPATH)
source .env

# 1. Preprocess: download, filter, write TFRecords
uv run preprocess --config configs/video_games.yaml --output data/video_games

# 2. Train
uv run train --config configs/video_games.yaml --data data/video_games
```

Trained model is saved to `runs/video_games/model.keras`, selected by best validation NDCG@10.

## Requirements

- Python 3.12+
- Local dev: CPU-only JAX (default in `pyproject.toml`)
- GPU training: swap `jax[cpu]` → `jax[cuda12]` in `pyproject.toml`, then `uv sync`

## Switching datasets

Copy `configs/video_games.yaml` to a new file (e.g. `configs/books.yaml`), update the `review_url` and tune `model` / `training` params. Everything else is config-driven.

## Project layout

```
configs/          YAML configs per dataset
src/hstu_rec/
  dataset.py      Config dataclasses, TFRecordDataFactory, parse_tfrecord_fn
  metrics.py      NDCGAtK Keras metric
  train.py        LastNonPaddingToken layer, build_model(), train CLI
  preprocess/
    download.py   Download + decompress review JSONL
    filter.py     Iterative 5-core filtering
    tfrecords.py  Write train/val/test TFRecords
tests/            Fast (default) and slow (@pytest.mark.slow) tests
vendor/RecML/     RecML git submodule
docs/             Design specs, implementation plans, architecture notes
```

## Running tests

```bash
# Fast tests only (default — no TF/JAX required)
uv run pytest

# All tests including TF/Keras/JAX (run on GPU machine)
uv run pytest -m slow
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for a full description of the model, data pipeline, and scaling notes.
