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
                        Writes train/val/test.tfrecord + item_map.parquet + vocab_size.txt
        │
        ▼
   TFRecordDataFactory  tf.data pipeline: parse → (optionally shuffle+repeat)
                        → batch → prefetch
```

**TFRecord schema** (per example):

| Field        | Type          | Shape          | Notes                                      |
|--------------|---------------|----------------|--------------------------------------------|
| `input_ids`  | int64         | `[seq_len]`    | Left-zero-padded item IDs                  |
| `timestamps` | int64         | `[seq_len]`    | Left-zero-padded Unix timestamps (seconds) |
| `target_id`  | int64         | `[1]`          | Next item to predict                       |

Item IDs: `0` = padding, real items `1..vocab_size`. The embedding table has `vocab_size + 1` rows to accommodate the padding token.

`timestamps` are always written and parsed regardless of `use_timestamps`. This means preprocessing only needs to run once; flipping the flag in the config does not require regenerating TFRecords.

## Model

The forward pass has two modes that share all weights but differ in what they compute.

### Training path (sampled softmax)

```
input_ids  (batch, seq_len)
timestamps (batch, seq_len)  ← only when use_timestamps=True
     │
     ▼
[RelativeBucketedTimeAndPositionBasedBias]   log-bucketed pairwise time deltas
     │  → attention_bias (batch, seq_len, seq_len)   (skipped when use_timestamps=False)
     │
     ▼
  _TimestampHSTU (RecML HSTU subclass)        vocab_size+1 × model_dim tied embedding
     │  num_layers × HSTUBlock                causal pointwise attention + optional time bias
     │  add_head=False                         keeps model_dim representation
     ▼
sequence embeddings  (batch, seq_len, model_dim)
     │
     ▼
LastNonPaddingToken                 gather slice at last non-zero input position
     │
     ▼
user embedding  (batch, model_dim)
     │
     ▼
_sampled_softmax_loss               sample num_sampled negatives uniformly;
                                    compute logits for {true_item} ∪ {negatives};
                                    cross-entropy with true item at index 0
```

Cost: O(`num_sampled × model_dim`) per step instead of O(`vocab_size × model_dim`).

### Eval / inference path (full logits)

```
input_ids  (batch, seq_len)
timestamps (batch, seq_len)  ← only when use_timestamps=True
     │
     ▼
  _TimestampHSTU  (same weights, add_head=False)
     │
     ▼
sequence embeddings  (batch, seq_len, model_dim)
     │
     ▼
LastNonPaddingToken
     │
     ▼
user embedding  (batch, model_dim)
     │
     ▼
user_emb @ embedding_table.T        full tied-embedding projection
     │
     ▼
item logits  (batch, vocab_size+1)
     │
     ▼
NDCG@10 / HR@10                     ranked against the full catalogue
```

**Key design choices:**

- **Tied embeddings (`add_head=False` + manual projection):** the output projection reuses the item embedding weights, reducing parameter count. With `add_head=False` we control *when* the projection runs — only during eval — which is what enables sampled softmax during training.
- **Sampled softmax:** `num_sampled` negatives are drawn uniformly each step. The gradient is a biased estimator of the full-softmax gradient but converges to the same optimum and is `vocab_size / num_sampled` times cheaper per step. Configured via `training.num_sampled` in the YAML.
- **Optional timestamp bias (`use_timestamps`):** when enabled, `RelativeBucketedTimeAndPositionBasedBias` converts Unix timestamps to log-bucketed pairwise time differences and adds them as an additive attention bias. This lets the model weight recent interactions more heavily. Setting `use_timestamps: false` (the default) leaves the bias at zero and the model behaves identically to a pure ID-based HSTU. Timestamps are *always* stored in TFRecords and parsed by the data pipeline, so toggling the flag in the config requires no preprocessing rerun. Implemented via `_TimestampHSTU`, a thin subclass that forwards `attention_bias` to each `HSTUBlock` — the upstream `HSTU.call()` accepts the argument but never passes it through.
- **Causal (lower-triangular) attention mask**: generated from the padding mask so future positions can't attend to padding, and the model is autoregressive over the sequence.
- **Left-zero-padding**: each training example is a prefix of the user's history. Padding on the left means the model always sees the most recent context in the rightmost positions, which matters for the causal mask.
- **Leave-last-out**: standard evaluation protocol for sequential recommendation. Val and test each get exactly one example per user; training gets all intermediate positions.

## Known issues in upstream RecML HSTU (`vendor/RecML/recml/layers/keras/hstu.py`)

These are bugs in the upstream code that we work around or accept for now. If we fork RecML in the future, these should be the first fixes.

1. **`self.final_norm` is applied twice in `HSTU.call()` (lines 601 and 616).** The same `LayerNormalization` layer — one set of gamma/beta weights — is called both before the decoder block loop and after it. Those two points see completely different activation distributions (raw embeddings vs. post-attention outputs), so the shared weights receive conflicting gradient signals during training. Each `HSTUBlock` already applies its own `_input_layer_norm` internally, making the pre-block call redundant as well as harmful. **Fix:** remove the first call (line 601) and keep only the post-block final norm. Affects every forward pass regardless of config.

2. **`HSTU.call()` does not accept or forward `attention_bias` to decoder blocks.** `HSTUBlock.call()` supports an `attention_bias` parameter, and the same file ships `RelativeBucketedTimeAndPositionBasedBias` to produce one, but `HSTU.call()` never wires them together. We work around this with `_TimestampHSTU` in `train.py`, a thin subclass that copy-pastes the entire `HSTU.call()` forward logic to add `attention_bias` forwarding. **Fix:** add `attention_bias` to `HSTU.call()`'s signature and pass it to each block. Only affects `use_timestamps: true` — with the default `false`, `attention_bias` is `None` and the missing forwarding is a no-op.

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
  use_timestamps: false   # set true to add time-bucketed attention bias

training:
  batch_size: 128
  train_steps: 10000
  steps_per_eval: 500
  steps_per_loop: 100
  model_dir: runs/video_games
  num_sampled: 1000   # negatives per training step
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

**Training is already solved:** sampled softmax is implemented. Set `num_sampled` in the YAML (10 000 is a reasonable default for Books). The per-step cost drops from O(vocab_size) to O(num_sampled), typically 100–300× cheaper for a 3M-item catalogue.

**Remaining eval bottleneck:** NDCG@10 still scores all items during validation. For very large catalogues:
- **Approximate nearest-neighbour retrieval**: use FAISS or ScaNN to retrieve the top-1000 candidates via ANN search against pre-computed item embeddings, then compute NDCG@10 over candidates only. Eval becomes approximate but runs in sub-second per batch.
- **In-batch negatives** (alternative training approach): treat other items in the batch as negatives — free negatives, no sampling needed, but limited to `batch_size` candidates per step.

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

| Scale            | Key change                                           |
|------------------|------------------------------------------------------|
| ~1M items        | `jax[cuda12]`, single GPU                           |
| ~3M items        | Tune `num_sampled` (already implemented); ANN eval  |
| >8 GB raw data   | Chunk-based preprocessing or cloud VM               |
| Large throughput | Shard TFRecords (128–256 shards)                    |
| Long sequences   | Increase `max_sequence_length` or chunked attn      |
| Multi-GPU        | JAX `jax_trainer.py` + device mesh                  |
