from __future__ import annotations

import argparse
import os
from pathlib import Path


def _sampled_softmax_loss(emb_table, user_emb, targets, num_sampled):
    """Sampled softmax loss using keras.ops (backend-agnostic).

    Samples ``num_sampled`` negatives uniformly at random, then minimises
    cross-entropy over ``{true_item} ∪ {negatives}`` with the true item
    always at index 0.  No frequency-correction is applied (uniform sampling).

    Args:
        emb_table:   float tensor ``(vocab_size+1, model_dim)``
        user_emb:    float tensor ``(batch, model_dim)``
        targets:     int tensor   ``(batch,)`` — 1-indexed true item IDs
        num_sampled: number of randomly-sampled negatives per step

    Returns:
        Scalar mean loss.
    """
    import keras

    vocab_size = keras.ops.shape(emb_table)[0]
    batch_size = keras.ops.shape(user_emb)[0]

    # Sample negatives uniformly: (num_sampled,)
    sampled_ids = keras.random.randint(
        shape=(num_sampled,), minval=0, maxval=vocab_size, dtype="int32"
    )

    # True item logits: dot(user_emb[i], emb_table[targets[i]])  → (batch, 1)
    true_emb = keras.ops.take(emb_table, targets, axis=0)
    true_logits = keras.ops.sum(user_emb * true_emb, axis=-1, keepdims=True)

    # Sampled logits: user_emb @ sampled_emb.T  → (batch, num_sampled)
    sampled_emb = keras.ops.take(emb_table, sampled_ids, axis=0)
    sampled_logits = keras.ops.matmul(user_emb, keras.ops.transpose(sampled_emb))

    # True item is always at position 0  → label is always 0
    all_logits = keras.ops.concatenate([true_logits, sampled_logits], axis=1)
    labels = keras.ops.zeros((batch_size,), dtype="int32")

    loss = keras.losses.sparse_categorical_crossentropy(labels, all_logits, from_logits=True)
    return keras.ops.mean(loss)


class LastNonPaddingToken:
    """Keras layer: extracts the hidden state at the last non-padding position.

    Padding token ID is 0.  Given HSTU output ``(batch, seq_len, dim)`` and
    the original ``input_ids`` ``(batch, seq_len)``, returns the slice at the
    last non-zero position per example → ``(batch, dim)``.
    """

    def __new__(cls, **kwargs):
        import keras

        class _LastNonPaddingToken(keras.layers.Layer):
            def call(self, inputs, input_ids):
                import keras as _keras

                mask = _keras.ops.cast(
                    _keras.ops.not_equal(input_ids, 0), dtype="int32"
                )  # (batch, seq_len)

                seq_len = _keras.ops.shape(input_ids)[1]
                positions = _keras.ops.arange(seq_len, dtype="int32")

                masked_positions = _keras.ops.where(
                    _keras.ops.cast(mask, "bool"),
                    positions,
                    _keras.ops.full_like(positions, -1),
                )  # (batch, seq_len)

                last_idx = _keras.ops.argmax(masked_positions, axis=1)  # (batch,)

                one_hot = _keras.ops.one_hot(
                    last_idx, seq_len, dtype=inputs.dtype
                )  # (batch, seq_len)
                one_hot = _keras.ops.expand_dims(one_hot, axis=-1)  # (batch, seq_len, 1)
                return _keras.ops.sum(inputs * one_hot, axis=1)  # (batch, dim)

        return _LastNonPaddingToken(**kwargs)


def build_model(
    vocab_size: int,
    max_sequence_length: int,
    model_dim: int,
    num_heads: int,
    num_layers: int,
    dropout: float,
    learning_rate: float,
    num_sampled: int = 10000,
) -> "keras.Model":
    """Build and compile the HSTU next-item prediction model.

    Training uses sampled softmax (``num_sampled`` negatives per step).
    Evaluation uses full-vocabulary logits and tracks NDCG@10.

    Args:
        vocab_size: Number of real items (padding token 0 is excluded).
        max_sequence_length: Length of ``input_ids`` sequences.
        model_dim: Embedding / hidden dimension.
        num_heads: HSTU attention heads.
        num_layers: HSTU blocks.
        dropout: Dropout rate.
        learning_rate: Adam learning rate.
        num_sampled: Negative samples per training step.

    Returns:
        Compiled Keras model.  Input: ``{"input_ids": int32[batch, max_seq_len]}``.
        Eval output: logits ``float32[batch, vocab_size+1]``.
    """
    import keras
    from recml.layers.keras.hstu import HSTU
    from hstu_rec.metrics import NDCGAtK

    vocab_size_with_pad = vocab_size + 1  # index 0 = padding

    class HSTURecommender(keras.Model):
        """HSTU recommender with sampled softmax training and NDCG@10 eval."""

        def __init__(self):
            super().__init__(name="hstu_rec")
            self._vocab_size = vocab_size
            self._model_dim = model_dim
            self.hstu = HSTU(
                vocab_size=vocab_size_with_pad,
                model_dim=model_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                add_head=False,  # keep model_dim output; we project manually
                name="hstu",
            )
            self.last_token = LastNonPaddingToken(name="last_token")
            self._loss_tracker = keras.metrics.Mean(name="loss")
            self._ndcg_metric = NDCGAtK(k=10)

        @property
        def embedding_table(self):
            """Item embedding matrix ``(vocab_size+1, model_dim)``."""
            return self.hstu.item_embedding.embeddings

        def call(self, inputs, training=False):
            """Forward pass.

            Returns:
                training=True:  user embedding ``(batch, model_dim)`` —
                    cheaper than a full vocab projection; used by
                    ``compute_loss`` for sampled softmax.
                training=False: full logits ``(batch, vocab_size+1)`` —
                    used for NDCG@10 and inference.
            """
            input_ids = inputs["input_ids"]
            padding_mask = keras.ops.cast(input_ids, "bool")
            seq_out = self.hstu(input_ids, padding_mask=padding_mask, training=training)
            user_emb = self.last_token(seq_out, input_ids)
            if training:
                return user_emb
            # Tied embedding projection: user_emb @ emb_table.T
            return keras.ops.matmul(user_emb, keras.ops.transpose(self.embedding_table))

        def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, training=True):
            """Loss computation — the primary gradient customisation point.

            During training ``y_pred`` is user embedding (batch, model_dim);
            sampled softmax avoids the full vocab projection.
            During eval ``y_pred`` is full logits (batch, vocab_size+1);
            standard sparse CE is used to track val_loss.
            """
            if training:
                loss = _sampled_softmax_loss(
                    self.embedding_table, y_pred, y, num_sampled
                )
            else:
                loss = keras.ops.mean(
                    keras.losses.sparse_categorical_crossentropy(
                        y, y_pred, from_logits=True
                    )
                )
            self._loss_tracker.update_state(loss)
            return loss

        def compute_metrics(self, x, y, y_pred, sample_weight=None):
            """Update NDCG@10 only when ``y_pred`` is full logits (eval mode)."""
            if y_pred.shape[-1] == self._vocab_size + 1:
                self._ndcg_metric.update_state(y, y_pred, sample_weight)
            return {m.name: m.result() for m in self.metrics}

        @property
        def metrics(self):
            return [self._loss_tracker, self._ndcg_metric]

    model = HSTURecommender()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    return model


def main(config_path: str | None = None, data_dir: str | None = None) -> None:
    """CLI entry point for training."""
    if config_path is None or data_dir is None:
        parser = argparse.ArgumentParser(description="Train HSTU recommender")
        parser.add_argument("--config", required=True, help="Path to YAML config")
        parser.add_argument("--data", required=True, help="Directory with TFRecords")
        args = parser.parse_args()
        config_path = args.config
        data_dir = args.data

    import keras
    from hstu_rec.dataset import load_config, make_data_factory

    config = load_config(config_path)

    vocab_size_path = Path(data_dir) / "vocab_size.txt"
    if not vocab_size_path.exists():
        raise FileNotFoundError(
            f"vocab_size.txt not found in {data_dir}. Run preprocess first."
        )
    vocab_size = int(vocab_size_path.read_text().strip())

    model = build_model(
        vocab_size=vocab_size,
        max_sequence_length=config.model.max_sequence_length,
        model_dim=config.model.model_dim,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        learning_rate=config.model.learning_rate,
        num_sampled=config.training.num_sampled,
    )

    train_ds = make_data_factory(config, data_dir, "train").make()
    val_ds = make_data_factory(config, data_dir, "val").make()

    os.makedirs(config.training.model_dir, exist_ok=True)

    model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=config.training.steps_per_loop,
        epochs=config.training.train_steps // config.training.steps_per_loop,
        validation_steps=config.training.steps_per_eval,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(config.training.model_dir, "model.keras"),
                save_best_only=True,
                monitor="val_ndcg_at_10",
                mode="max",
            ),
        ],
    )
    print(f"Training complete. Model saved to {config.training.model_dir}/model.keras")
