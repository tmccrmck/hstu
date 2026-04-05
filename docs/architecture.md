# Architecture

## Overview

Sequential next-item recommender built on Google's HSTU (Hierarchical Sequential Transduction Units) from [Actions Speak Louder than Words](https://arxiv.org/abs/2402.17152). Given a user's interaction history, the model predicts which item they will interact with next.

## Data pipeline

```
Amazon Reviews JSONL.gz
        │
        ▼
   download.py          HTTP download + decompress; skips if cached
        │
        ▼
   filter.py            Iterative 5-core: remove users/items with
                        fewer than min_interactions until convergence
        │
        ▼
   tfrecords.py         Leave-last-out split per user:
                          test  → seq[-1]          as target
                          val   → seq[-2]          as target
                          train → seq[j] for j in range(1, L-2) as targets
                        Writes train/val/test.tfrecord + item_map.json + vocab_size.txt
        │
        ▼
   TFRecordDataFactory  tf.data pipeline: parse → (optionally shuffle+repeat)
                        → batch → prefetch
```

**TFRecord schema** (per example):

| Field       | Type          | Shape          | Notes                          |
|-------------|---------------|----------------|--------------------------------|
| `input_ids` | int64         | `[seq_len]`    | Left-zero-padded item IDs      |
| `target_id` | int64         | `[1]`          | Next item to predict           |

Item IDs: `0` = padding, real items `1..vocab_size`. The embedding table has `vocab_size + 1` rows to accommodate the padding token.

## Model

```
input_ids  (batch, seq_len)
     │
     ▼
  HSTU (RecML)                      vocab_size+1 × model_dim tied embedding
     │  num_layers × HSTUBlock      causal pointwise attention
     │  add_head=True               projects back to vocab via tied embedding
     ▼
sequence logits  (batch, seq_len, vocab_size+1)
     │
     ▼
LastNonPaddingToken                 gather the slice at the last non-zero input position
     │
     ▼
item logits  (batch, vocab_size+1)
     │
     ▼
SparseCategoricalCrossentropy loss  trained against target_id
```

**Key design choices:**

- **Tied embeddings** (`add_head=True`): the output projection reuses the item embedding weights, which reduces parameter count and tends to improve generalisation on sparse recommendation data.
- **Causal (lower-triangular) attention mask**: generated from the padding mask so future positions can't attend to padding, and the model is autoregressive over the sequence.
- **Left-zero-padding**: each training example is a prefix of the user's history. Padding on the left means the model always sees the most recent context in the rightmost positions, which matters for the causal mask.
- **Leave-last-out**: standard evaluation protocol for sequential recommendation. Val and test each get exactly one example per user; training gets all intermediate positions.

## Evaluation

- **HR@10** (Hit Rate at 10): fraction of users where the true next item appears in the top-10 predicted items.
- **NDCG@10** (Normalised Discounted Cumulative Gain at 10): ranks the true item within the top-10 and discounts by position. Computed by `NDCGAtK` in `metrics.py`. Used as the checkpoint monitor during training.

Both metrics are computed over the full item catalogue (not sampled negatives), which is standard but expensive at very large vocab sizes — see scaling notes below.

## Configuration

All hyperparameters live in YAML under `configs/`. No code changes are needed to switch datasets.

```yaml
dataset:
  name: video_games
  review_url: "..."
  min_interactions: 5

model:
  vocab_size: null        # filled from vocab_size.txt at train time
  max_sequence_length: 50
  model_dim: 64
  num_heads: 4
  num_layers: 4
  dropout: 0.5
  learning_rate: 1e-3

training:
  batch_size: 128
  train_steps: 10000
  steps_per_eval: 500
  steps_per_loop: 100
  model_dir: runs/video_games
```

---

## Scaling to larger datasets

The current setup works well for Video Games (~500k reviews after filtering). The Books dataset is roughly 10× larger and exposes several bottlenecks. Here is where they appear and what to do about them.

### 1. GPU / accelerator

**Bottleneck:** `jax[cpu]` is fine for development but is far too slow for training at scale.

**Fix:** On RunPod or any CUDA machine:
1. In `pyproject.toml`, change `jax[cpu]` → `jax[cuda12]` (or `jax[tpu]` for TPUs).
2. Run `uv sync`.
3. Set `KERAS_BACKEND=jax` (already in `.env`).

No code changes required — Keras abstracts the backend.

### 2. Vocabulary size and softmax cost

**Bottleneck:** Full-softmax cross-entropy over `vocab_size+1` logits is O(vocab_size) per example. Books has ~3M unique items after filtering.

**Options (roughly in order of effort):**
- **Sampled softmax** (`keras.losses.CategoricalCrossentropy` with negative sampling): sample a subset of negatives per batch. Standard in large-scale rec systems; loses some accuracy but trains 10–100× faster.
- **Approximate nearest-neighbour retrieval at eval**: instead of scoring all items, use FAISS or ScaNN to retrieve candidates, then rerank. Eval NDCG@10 becomes approximate but is fast enough to run every few thousand steps.
- **In-batch negatives**: treat other items in the batch as negatives. Simple to implement and effectively gives a large batch of negatives for free.

### 3. Preprocessing memory

**Bottleneck:** `filter.py` loads the entire filtered DataFrame into RAM. Books JSONL is ~8 GB uncompressed; iterative 5-core on that can exceed 16 GB peak.

**Fix:** Process in chunks using `pandas.read_json(..., lines=True, chunksize=...)` or switch to Polars (lazy evaluation). Alternatively, run preprocessing on a cloud instance (even a small CPU VM with 32 GB RAM is cheap) and store the TFRecords in GCS/S3.

### 4. TFRecord sharding

**Bottleneck:** A single `train.tfrecord` file becomes a throughput bottleneck; `tf.data` can't parallelise reads across a single file.

**Fix:** Shard into N files (e.g. N = 128 or 256). Change the writer in `tfrecords.py` to round-robin across shards, then update `TFRecordDataFactory.path` to a glob pattern like `{data_dir}/train-*.tfrecord`. `tf.data.TFRecordDataset` and `list_files(..., shuffle=True)` handle the rest.

### 5. Sequence length

**Bottleneck:** Long-tail users in Books can have thousands of interactions. `max_sequence_length=50` truncates them; longer sequences improve accuracy but increase memory quadratically with HSTU's attention.

**Options:**
- Increase `max_sequence_length` (256–512) if GPU memory allows; HSTU's pointwise attention is more memory-efficient than standard softmax attention, so this is feasible.
- For very long sequences, apply HSTU's sliding-window or chunked attention variant (the RecML `hstu_ops.py` has causal block-sparse implementations).

### 6. Multi-GPU / multi-host training

**Bottleneck:** Single-device training won't utilise multi-GPU RunPod instances.

**Fix:** JAX has first-class data-parallel sharding via `jax.sharding`. The RecML `jax_trainer.py` already wires up `PartitionSpec`-based model parallelism; swap from `keras_trainer.py` to `jax_trainer.py` and configure a `Mesh`. No model code changes needed — just trainer config.

### Summary table

| Scale            | Key change                                      |
|------------------|-------------------------------------------------|
| ~1M items        | `jax[cuda12]`, single GPU                      |
| ~3M items        | Add sampled softmax or in-batch negatives       |
| >8 GB raw data   | Chunk-based preprocessing or cloud VM          |
| Large throughput | Shard TFRecords (128–256 shards)               |
| Long sequences   | Increase `max_sequence_length` or chunked attn |
| Multi-GPU        | JAX `jax_trainer.py` + device mesh             |
