# Design: Model-parallel item embedding

**Status:** future work, not implemented.
**Trigger:** vocabulary exceeds what fits replicated on one GPU, or per-device embedding memory becomes the dominant training constraint.

## Motivation

The item embedding is by far the largest tensor in this model. At Books-scale (3M items) with `model_dim=128` tied, it is ~1.5 GB in float32 — fine on any modern GPU. At 50M items it becomes ~25 GB and starts crowding out activations, optimizer state, and batch size. Beyond that, a single GPU can no longer hold the embedding replicated, and training requires model parallelism.

Everything else in HSTU is small relative to the embedding: the position embedding is bounded by `max_sequence_length × model_dim`, and each HSTUBlock's Q/K/V/U/output dense layers are `model_dim × model_dim × num_heads`-sized. Those scale with model depth and width, not with vocab. Sharding them buys nothing.

So the design is narrow: shard exactly one tensor — `hstu/item_embedding/embeddings` — along its vocab (row) axis, replicate everything else, and let Keras/JAX handle the rest.

## When to use it

| Scenario                              | Embedding memory (tied, float32) | Verdict                       |
|---------------------------------------|----------------------------------|-------------------------------|
| Video Games (~50k items, dim=64)      | ~13 MB                           | Use single-device data parallel |
| Books (~3M items, dim=128)            | ~1.5 GB                          | Use single-device data parallel |
| 10M items, dim=128                    | ~5 GB                            | Still fine replicated on A100 |
| 50M items, dim=256                    | ~51 GB                           | Model parallel starts helping |
| 500M items, dim=256                   | ~512 GB                          | Model parallel required       |

Rule of thumb: if the embedding fits comfortably in one GPU (under ~40% of device memory) alongside activations and optimizer state, don't bother with model parallelism — the collective overhead will hurt throughput without reducing memory pressure.

## API surface

**New config field** in `TrainingConfig`:

```yaml
training:
  num_model_shards: 1   # 1 = pure data parallel (default); >1 = model parallel over N devices
```

**New module** `src/hstu_rec/distribution.py`:

```python
def setup_model_parallel(num_model_shards: int) -> None:
    """Configure Keras/JAX to shard the item embedding across `num_model_shards` devices.

    Must be called before build_model(). No-op when num_model_shards == 1.
    """
    if num_model_shards <= 1:
        return

    import keras
    from keras.distribution import (
        DeviceMesh, LayoutMap, ModelParallel, list_devices, set_distribution,
    )

    devices = list_devices()
    if len(devices) % num_model_shards != 0:
        raise ValueError(
            f"num_devices ({len(devices)}) must be divisible by "
            f"num_model_shards ({num_model_shards})"
        )
    num_batch_shards = len(devices) // num_model_shards

    mesh = DeviceMesh(
        shape=(num_batch_shards, num_model_shards),
        axis_names=("batch", "model"),
        devices=devices,
    )
    layout_map = LayoutMap(mesh)
    layout_map[r".*item_embedding/embeddings"] = ("model", None)  # shard on vocab axis
    # Everything else defaults to replicated.

    set_distribution(ModelParallel(layout_map=layout_map, batch_dim_name="batch"))
```

**Wiring in `train.main()`:**

```python
from hstu_rec.distribution import setup_model_parallel
setup_model_parallel(config.training.num_model_shards)
# ...then build_model(...) as before
```

## What gets sharded

| Tensor                                 | Shape                     | Layout            |
|----------------------------------------|---------------------------|-------------------|
| `hstu/item_embedding/embeddings`       | `(vocab+1, model_dim)`    | `('model', None)` |
| AdamW `m` / `v` for the above          | `(vocab+1, model_dim)` each | inherits layout |
| `hstu/position_embedding/embeddings`   | `(max_pos, model_dim)`    | replicated        |
| `hstu/hstu_block_i/{q,k,v,u}_dense/*`  | `(model_dim, heads, dim)` | replicated        |
| `hstu/hstu_block_i/output_dense/*`     | `(heads, dim, model_dim)` | replicated        |
| All LayerNorm gamma/beta               | `(model_dim,)`            | replicated        |

The mesh also distributes data along the `batch` axis, so `num_batch_shards × num_model_shards = total devices`. Pure model parallel (`num_batch_shards = 1`) is a valid configuration but usually suboptimal — you want some data parallelism to keep devices busy.

## Operations that need to Just Work

1. **Sampled softmax negatives**: `keras.ops.take(emb_table, sampled_ids, axis=0)` is a gather along the sharded axis. JAX lowers this to an all-gather or sharded gather; throughput depends on how many sampled IDs land on each shard.
2. **Eval projection `user_emb @ emb_table.T`**: produces a `(batch, vocab)` tensor sharded along vocab. Top-k requires an all-reduce/all-gather. Keras handles this; expected to be fine at our scales.
3. **Gradient updates**: gradients for the sharded embedding are themselves sharded. Optimizer state (`m`, `v` for AdamW) inherits the layout automatically.

## Hard constraints and failure modes

- **JAX backend only.** Keras ModelParallel is JAX-only. We already use JAX — this is not a new requirement but it rules out ever switching backends.
- **`set_distribution()` must run before model construction.** Creating the model first and setting the distribution afterwards is undefined behaviour — weights may end up replicated even though the layout map says to shard them.
- **Regex path is fragile.** The layout map keys on variable paths (`.*item_embedding/embeddings`). If RecML renames the layer or restructures the hierarchy, our regex silently stops matching, the embedding falls back to replicated, and we OOM at scale. Mitigation: add a verification test that asserts `model.get_layer("hstu").item_embedding.embeddings` has the expected sharded layout after build. Fail loudly if it doesn't.
- **`tie_weights=True` is required**, not just preferred. With `tie_weights=False`, there's a second `(vocab+1, model_dim)` reverse-embedding tensor that also needs a layout entry (or gets replicated and blows memory). Tying the weights avoids the whole problem.

## What this explicitly does not solve

- **Activation memory.** Large batches × long sequences × many layers still need gradient checkpointing or smaller batches. ModelParallel shards weights, not activations.
- **Vocab-dependent computation cost.** Sampled softmax already removes this at training time. Eval still materializes `(batch, vocab)` logits; at 50M+ vocab this becomes expensive even when sharded. The real fix is ANN retrieval over pre-computed item embeddings (documented separately in `architecture.md` under "Scaling to larger datasets").
- **OOV / new items.** Orthogonal — model parallelism doesn't change the vocabulary contract.

## Sequencing against other scaling work

Cheaper and should land first:

1. **`tie_weights=True`** — halves embedding memory (~1.5 GB → ~750 MB at Books). Necessary prerequisite for model parallelism anyway.
2. **bfloat16 embeddings** — halves memory again, negligible quality cost for embeddings.
3. **ANN eval retrieval** — removes full-vocab matmul from evaluation, the other major vocab-dependent cost.

Model parallelism is the right tool only when those three together still leave the embedding as the dominant memory constraint.
