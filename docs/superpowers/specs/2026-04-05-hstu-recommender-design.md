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
- Data pipelines: `tf.data` + TFRecords
- Config: YAML files parsed into Python dataclasses
- Hardware target: GPU

## Project Structure

```
hstu/
├── pyproject.toml               # uv project; RecML as git dependency; CLI entry points
├── configs/
│   ├── video_games.yaml         # Dataset + model + training config for Video_Games
│   └── books.yaml               # Same for Books (future)
├── src/
│   └── hstu_rec/
│       ├── preprocess/
│       │   ├── __init__.py      # Exposes main() as CLI entry point
│       │   ├── download.py      # Fetch + decompress .jsonl.gz; skip if cached
│       │   ├── filter.py        # Iterative 5-core filtering
│       │   └── tfrecords.py     # Build sequences → write TFRecords + metadata
│       ├── train.py             # Exposes main(); HSTUTask; wires config → RecML experiment
│       ├── dataset.py           # Config dataclasses, YAML loader, TFRecordDataFactory, parse_tfrecord_fn
│       └── metrics.py           # NDCGAtK Keras metric
└── docs/
    └── superpowers/specs/
```

## CLI Entry Points

Registered in `pyproject.toml` under `[project.scripts]`:

```toml
[project.scripts]
preprocess = "hstu_rec.preprocess:main"
train      = "hstu_rec.train:main"
```

Usage:

```bash
# Stage 1: preprocess (writes TFRecords + metadata to OUTPUT_DIR)
uv run preprocess --config configs/video_games.yaml --output OUTPUT_DIR

# Stage 2: train + eval (reads TFRecords from DATA_DIR, which must equal OUTPUT_DIR above)
uv run train --config configs/video_games.yaml --data DATA_DIR [--model-dir MODEL_DIR]
```

**Important:** `--output` (preprocess) and `--data` (train) must point to the same directory. The train stage reads `vocab_size.txt` and TFRecord files written there by the preprocess stage.

`--model-dir` is optional; if provided, it overrides `training.model_dir` from the YAML config. Swapping to Books: `--config configs/books.yaml` — no code changes required.

## Configuration

```yaml
# configs/video_games.yaml
dataset:
  name: video_games
  review_url: https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Video_Games.jsonl.gz
  min_interactions: 5          # 5-core threshold

model:
  vocab_size: null             # auto-populated from vocab_size.txt written by preprocessor
  max_sequence_length: 50      # input sequence length; sequences truncated to this length
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

`vocab_size: null` decouples preprocessing from training config — the actual count is written by the preprocessor and read at training time. If `vocab_size.txt` is missing, `train.py` raises `FileNotFoundError` with a message instructing the user to run preprocessing first.

## Config Dataclasses (`dataset.py`)

```python
@dataclass
class DatasetConfig:
    name: str
    review_url: str
    min_interactions: int

@dataclass
class ModelConfig:
    vocab_size: int | None    # None until populated from vocab_size.txt at training time
    max_sequence_length: int
    model_dim: int
    num_heads: int
    num_layers: int
    dropout: float
    learning_rate: float

@dataclass
class TrainingConfig:
    batch_size: int
    train_steps: int
    steps_per_eval: int
    model_dir: str

@dataclass
class Config:
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig

def load_config(path: str) -> Config: ...
```

`load_config()` parses the YAML and returns a fully-populated `Config`. All three sub-configs are always present; no optional sections.

## Data Flow

### Stage 1 — Preprocessing

1. **Download** (`download.py`) — fetch the `.jsonl.gz` from `config.dataset.review_url`. Decompress to `{output_dir}/cache/{name}.jsonl` where `name` is derived from the filename in the URL (e.g. `Video_Games.jsonl.gz` → `Video_Games.jsonl`). Skip both download and decompression if `{output_dir}/cache/{name}.jsonl` already exists.

2. **Filter** (`filter.py`) — load the `.jsonl` file into a **pandas DataFrame** entirely in memory, with columns `user_id` (str), `parent_asin` (str, the Amazon ASIN — used as the item identifier, no normalization applied), and `timestamp` (int, Unix milliseconds as stored in the dataset). Parse only these three fields from each JSON line; ignore all other fields. Apply iterative 5-core filtering: repeatedly remove rows where any `user_id` or `parent_asin` has fewer than `min_interactions` occurrences until neither set changes. Return the filtered DataFrame with the same three columns. **Note:** for Video_Games (~4.6M rows), in-memory processing requires ~1–2 GB RAM. For Books (~29.5M rows), this may require 10+ GB. An engineer extending to Books should profile memory usage.

3. **Assign item IDs** (`tfrecords.py`) — assign contiguous integer IDs to unique `parent_asin` values in the filtered DataFrame: **0 is the padding token**, real items are assigned IDs 1 through `vocab_size` (in arbitrary order, e.g. sorted by first occurrence). `vocab_size` = number of unique `parent_asin` values after filtering. Sort each user's rows by `timestamp` to form an ordered integer sequence of item IDs.

4. **Split and write TFRecords** (`tfrecords.py`) — apply leave-last-out per user. Given a user's full item ID sequence of length L (L ≥ `min_interactions` ≥ 5 by construction):
   - **Test** (`test.tfrecord`): input = last `max_seq_len` items of `sequence[:-1]`, zero-padded on the left; target = `sequence[-1]`.
   - **Val** (`val.tfrecord`): input = last `max_seq_len` items of `sequence[:-2]`, zero-padded on the left; target = `sequence[-2]`.
   - **Train** (`train.tfrecord`): one example for each position `j` in `range(1, L - 2)`: input = last `max_seq_len` items of `sequence[:j]`, zero-padded on the left; target = `sequence[j]`. This yields `L - 3` training examples per user (minimum 2 for L=5).
   - `test.tfrecord` is written but not used during training; it is reserved for future final evaluation.

5. **Write metadata** to `--output`:
   - `item_map.json`: `{"<parent_asin_string>": <integer_id>, ...}`. Keys are raw ASIN strings from the dataset. Not read at training time; for post-hoc analysis only.
   - `vocab_size.txt`: single line, single integer — the count of real items (padding token not included). Example: `137000`.

### Stage 2 — Training

`train.py` sequence:
1. Parse CLI args: `--config`, `--data`, optional `--model-dir`.
2. Load `Config` via `dataset.load_config(config_path)`.
3. If `--model-dir` provided, override `config.training.model_dir`.
4. Read `vocab_size` from `{data_dir}/vocab_size.txt` (single integer, single line). Raise `FileNotFoundError` if missing.
5. Set `config.model.vocab_size = vocab_size`.
6. Construct `HSTUTask(config=config, data_dir=data_dir)`.
7. Construct `KerasTrainer(model_dir, train_steps, steps_per_eval)`.
8. Call `recml.core.run_experiment(Experiment(task, trainer), mode=TRAIN_AND_EVAL)`.

## Module Responsibilities

### `dataset.py`

Owns:
- Config dataclasses and `load_config()`.
- `TFRecordDataFactory` — a custom dataclass (not a RecML class) with fields `path: str`, `batch_size: int`, `max_sequence_length: int`, `is_training: bool`. Has a `make() -> tf.data.Dataset` method that: (a) lists files matching `path`, (b) reads them as `TFRecordDataset` with `num_parallel_reads=AUTOTUNE`, (c) applies `parse_tfrecord_fn`, (d) shuffles and repeats if `is_training`, (e) batches to `batch_size` with `drop_remainder=True`, (f) prefetches with `AUTOTUNE`.
- `make_data_factory(config: Config, data_dir: str, split: str) -> TFRecordDataFactory` — constructs a single factory for `split` ∈ `{"train", "val", "test"}`. Sets `path=f"{data_dir}/{split}.tfrecord"` and `is_training=(split == "train")`.
- `parse_tfrecord_fn(max_sequence_length: int)` — returns a callable `fn(serialized: tf.Tensor) -> (dict, tf.Tensor)` that deserializes one TFRecord example and returns `({"input_ids": int32[max_seq_len]}, int32[])`. The target is squeezed from shape `[1]` to scalar `[]` so it is compatible with `SparseCategoricalCrossentropy`.

Does **not** own: HSTU model construction, training loop, download/filtering logic.

### `train.py`

Owns: CLI entry point, `HSTUTask` class, `LastNonPaddingToken` layer, wiring of config → RecML experiment objects.

**`HSTUTask` constructor:**
```python
class HSTUTask(keras_trainer.KerasTask):
    def __init__(self, config: Config, data_dir: str): ...
```
Stores `config` and `data_dir`. Does **not** accept pre-built factories — it creates them internally in `create_dataset`.

**`HSTUTask.create_dataset(training: bool) -> tf.data.Dataset`:**
Calls `dataset.make_data_factory(self.config, self.data_dir, split="train" if training else "val")`, then calls `.make()` on the returned factory.

**`HSTUTask.create_model() -> keras.Model`:** See RecML Integration section.

Does **not** own: data factory construction logic or TFRecord parsing (both in `dataset.py`).

### `preprocess/`

`__init__.py` exposes `main()`. `download.py`, `filter.py`, `tfrecords.py` are internal helpers. `main()` orchestrates them as follows:

```
main(config_path: str, output_dir: str):
    config = load_config(config_path)
    jsonl_path = download.download(config.dataset.review_url, output_dir)
        # returns: Path to {output_dir}/cache/{name}.jsonl
    df = filter.filter_reviews(jsonl_path, config.dataset.min_interactions)
        # returns: pd.DataFrame with columns [user_id, parent_asin, timestamp]
    tfrecords.write_tfrecords(df, config.model.max_sequence_length, output_dir)
        # writes: train.tfrecord, val.tfrecord, test.tfrecord, item_map.json, vocab_size.txt
```

Each helper is a module-level function with no shared state.

## TFRecord Schema

All splits (train, val, test) use the same schema. Each record is a `tf.train.Example` with fixed-length features:

| Feature key  | `tf.io` type              | Stored dtype | Shape                   | Description                                        |
|-------------|--------------------------|-------------|-------------------------|----------------------------------------------------|
| `input_ids` | `FixedLenFeature(int64)` | int64        | `[max_sequence_length]` | Item ID sequence, zero-padded on the left          |
| `target_id` | `FixedLenFeature(int64)` | int64        | `[1]`                   | Ground-truth next item ID (always a real item ≥ 1) |

All sequences are padded to exactly `max_sequence_length` at write time; no runtime padding is needed. `parse_tfrecord_fn` must explicitly cast both features from int64 to int32 after parsing (TFRecords only support int64 for integers, but the Keras input layer expects int32). After parsing and casting: `input_ids` is `int32[max_sequence_length]`; `target_id` is squeezed from `int64[1]` to scalar `int32[]`.

## RecML Integration

### `HSTUTask.create_model()`

```python
inputs = keras.Input(shape=(max_sequence_length,), dtype="int32", name="input_ids")

# Padding mask derived inside the model (not a second input tensor)
padding_mask = keras.ops.cast(keras.ops.not_equal(inputs, 0), "int32")

# HSTU: add_head=True (default) → output shape [batch, seq_len, vocab_size+1]
seq_logits = hstu.HSTU(
    vocab_size=vocab_size + 1,
    max_positions=max_sequence_length,
    model_dim=model_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout=dropout,
)(inputs, padding_mask=padding_mask)

# Extract logit vector at last non-padding position → [batch, vocab_size+1]
output_logits = LastNonPaddingToken()([seq_logits, padding_mask])
output_logits = keras.layers.Activation("linear", dtype="float32")(output_logits)

model = keras.Model(inputs=inputs, outputs=output_logits)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.SparseTopKCategoricalAccuracy(k=10, name="HR@10"),
        NDCGAtK(k=10, name="NDCG@10"),
    ],
)
```

### `LastNonPaddingToken` layer (defined in `train.py`)

A `keras.layers.Layer` subclass. Inputs: `[seq_logits, padding_mask]` where `seq_logits` is `float32[batch, seq_len, vocab_size+1]` and `padding_mask` is `int32[batch, seq_len]`. Uses `keras.ops.*` throughout (not `tf.*`) to remain compatible with the JAX backend. Implementation:

```python
lengths = keras.ops.sum(keras.ops.cast(padding_mask, "int32"), axis=1)  # [batch]
last_indices = lengths - 1                                                # [batch]
# keras.ops.take equivalent to tf.gather with batch_dims=1:
# for each i: output[i] = seq_logits[i, last_indices[i], :]
return keras.ops.take_along_axis(
    seq_logits,
    keras.ops.expand_dims(keras.ops.expand_dims(last_indices, -1), -1)
    * keras.ops.ones((1, 1, seq_logits.shape[-1]), dtype="int32"),
    axis=1,
)[:, 0, :]                                                               # [batch, vocab_size+1]
```

### Constructor mapping: Config → RecML HSTU

| Config field                    | Passed to `HSTU` as |
|--------------------------------|---------------------|
| `model.vocab_size + 1`         | `vocab_size`        |
| `model.max_sequence_length`    | `max_positions`     |
| `model.model_dim`              | `model_dim`         |
| `model.num_heads`              | `num_heads`         |
| `model.num_layers`             | `num_layers`        |
| `model.dropout`                | `dropout`           |

### `KerasTrainer` constructor mapping

| Config field               | `KerasTrainer` arg  |
|---------------------------|---------------------|
| `training.model_dir`      | `model_dir`         |
| `training.train_steps`    | `train_steps`       |
| `training.steps_per_eval` | `steps_per_eval`    |

## Metrics

- **HR@10** — Hit Rate at 10, via RecML's `SparseTopKCategoricalAccuracy(k=10)`.
- **NDCG@10** — implemented as `NDCGAtK` in `metrics.py`.

**`NDCGAtK` input contract:**
- `y_true`: `int32[batch]` — integer item IDs (ground-truth targets, already squeezed to scalar per example).
- `y_pred`: `float32[batch, vocab_size+1]` — raw logits over the full vocabulary including padding token index 0.
- Computes the rank of each ground-truth item among all `vocab_size+1` positions (including index 0), then averages `1 / log2(rank + 1)` over the batch. Index 0 (padding token) is **not masked** during ranking — this is consistent with how `SparseTopKCategoricalAccuracy` operates and is the standard practice for full-vocabulary sequential recommendation evaluation.

**Evaluation protocol:** Full-vocabulary ranking — no sampled softmax. The model outputs logits over all `vocab_size+1` positions. Both HR@10 and NDCG@10 rank the ground-truth item against all positions including the padding token (index 0). Since all ground-truth targets are real items (ID ≥ 1), the padding logit acts as noise but does not bias the metric.

## Padding Convention

- Item ID `0` is the padding token. Real item IDs are in `[1, vocab_size]`.
- The HSTU embedding table size is `vocab_size + 1` (passed as `vocab_size` to RecML per the table above).
- `vocab_size.txt` stores the count of real items only. Example: `137000` means IDs 1–137000 are valid; embedding table has 137001 rows.
- The padding mask is computed inside the Keras model as `input_ids != 0`.

## Dataset Scope

- **Initial target:** Video_Games (2.8M users, 137K items, 4.6M ratings; significantly smaller after 5-core filtering).
- **Future extension:** Books (10.3M users, 4.4M items, 29.5M ratings) — add `configs/books.yaml` with the Books download URL; no code changes.

## Out of Scope

- Item metadata (titles, descriptions, images) — interaction sequences only.
- Rating prediction or explicit feedback modeling.
- Inference serving / model export.
- Hyperparameter search.
- Test-set evaluation (only val metrics reported during training; `test.tfrecord` is written but unused until a separate eval script is added).
