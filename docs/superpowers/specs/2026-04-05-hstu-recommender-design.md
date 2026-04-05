# HSTU Recommender System — Design Spec

**Date:** 2026-04-05
**Status:** Approved

## Overview

A sequential recommender system for the Amazon Reviews 2023 Video_Games dataset using Google's RecML HSTU (Hierarchical Sequential Transduction Units) implementation. The task is next-item prediction: given a user's interaction history, predict the next item they will interact with.

The project is structured as two independent CLI stages (preprocessing and training) driven by YAML dataset configs, making it straightforward to extend to other Amazon Reviews datasets (e.g. Books) without code changes.

## Architecture

**Approach:** Thin wrapper around RecML's reference Keras/JAX HSTU implementation. RecML is added as a git dependency via `uv`. We own the preprocessing pipeline and a minimal training harness; the HSTU model, trainer loop, and core metrics come from RecML unchanged.

**Technology stack:**
- Python package management: `uv`
- Model framework: Keras (JAX backend), via RecML
- Data pipelines: `tf.data` + TFRecords, as required by RecML
- Config: YAML files parsed into Python dataclasses
- Hardware target: GPU

## Project Structure

```
hstu/
├── pyproject.toml               # uv project; RecML as git dependency
├── configs/
│   ├── video_games.yaml         # Dataset + model + training config for Video_Games
│   └── books.yaml               # Same for Books (future)
├── src/
│   └── hstu_rec/
│       ├── preprocess/
│       │   ├── __init__.py
│       │   ├── download.py      # Fetch + decompress .jsonl.gz; skip if cached
│       │   ├── filter.py        # Iterative 5-core filtering
│       │   └── tfrecords.py     # Build sequences → write TFRecords + item map
│       ├── train.py             # Load config → construct RecML experiment → run
│       ├── dataset.py           # DatasetConfig dataclass + YAML loader
│       └── metrics.py           # NDCGAtK Keras metric (HR@K is RecML built-in)
└── docs/
    └── superpowers/specs/
```

## CLI Entry Points

```bash
# Stage 1: preprocess
uv run preprocess --config configs/video_games.yaml --output data/video_games/

# Stage 2: train + eval
uv run train --config configs/video_games.yaml --data data/video_games/ --model-dir runs/video_games/
```

Swapping to Books: `--config configs/books.yaml` — no code changes required.

## Data Flow

### Stage 1 — Preprocessing

1. **Download** (`download.py`) — fetch `Video_Games.jsonl.gz` from the URL in the YAML config. Decompress to a local cache dir. Skip if the file already exists.
2. **Filter** (`filter.py`) — apply iterative 5-core filtering: repeatedly remove users and items with fewer than `min_interactions` (default 5) interactions until convergence.
3. **Build sequences** (`tfrecords.py`) — sort each user's interactions by timestamp. Encode items to contiguous integer IDs (0 reserved for padding). Write the item→ID mapping as `item_map.json` in the output directory, along with `vocab_size.txt` (item count after filtering).
4. **Write TFRecords** (`tfrecords.py`) — use leave-last-out split: each user's last item is the test target, second-to-last is validation target, all preceding items form training sequences. Emit sliding-window examples (input = items[0..n-1], target = items[n]). Write to `train.tfrecord`, `val.tfrecord`, `test.tfrecord`.

### Stage 2 — Training

`train.py` loads the YAML config, reads `vocab_size.txt` from the data directory (overriding `model.vocab_size: null`), constructs `TFRecordDataFactory` instances and an `HSTUTask` (subclassing RecML's `KerasTask`), and delegates to RecML's `KerasTrainer`. Evaluation runs every `steps_per_eval` steps on the validation split.

## Configuration

```yaml
# configs/video_games.yaml
dataset:
  name: video_games
  review_url: https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Video_Games.jsonl.gz
  min_interactions: 5
  max_sequence_length: 50
  split_strategy: leave_last_out

model:
  vocab_size: null        # auto-populated from vocab_size.txt after preprocessing
  model_dim: 64
  num_heads: 4
  num_layers: 4
  dropout: 0.5
  learning_rate: 1e-3

training:
  batch_size: 128
  train_steps: 10000
  steps_per_eval: 500
  model_dir: runs/video_games
```

`vocab_size: null` decouples preprocessing from training config — the actual count is written by the preprocessor and read at training time.

## Metrics

- **HR@10** — Hit Rate at 10, provided by RecML's `SparseTopKCategoricalAccuracy(k=10)`
- **NDCG@10** — Normalized Discounted Cumulative Gain at 10, implemented as a custom `NDCGAtK` Keras metric in `metrics.py` and added to the model's compile call

Evaluation protocol: leave-last-out. The final item in each user's sequence is the ground truth; the model ranks it against all items in the vocabulary.

## Dataset Scope

- **Initial target:** Video_Games (2.8M users, 137K items, 4.6M ratings; after 5-core filtering: significantly smaller)
- **Future extension:** Books (10.3M users, 4.4M items, 29.5M ratings) — add `configs/books.yaml`, no code changes

## Out of Scope

- Item metadata (titles, descriptions, images) — interaction sequences only
- Rating prediction or explicit feedback modeling
- Inference serving / model export
- Hyperparameter search
