from __future__ import annotations

import argparse
import os
from pathlib import Path


class LastNonPaddingToken:
    """Keras layer that extracts the hidden state at the last non-padding position.

    Padding tokens have ID 0.  Given the HSTU output of shape
    ``(batch, seq_len, dim)`` and the original ``input_ids`` of shape
    ``(batch, seq_len)``, it returns the slice at the last non-zero position
    for each example in the batch, yielding ``(batch, dim)``.

    Usage::

        layer = LastNonPaddingToken()
        logits = layer(hstu_output, input_ids)  # (batch, dim)
    """

    def __new__(cls, **kwargs):
        import keras  # lazy import

        class _LastNonPaddingToken(keras.layers.Layer):
            def call(self, inputs, input_ids):
                """Extract last non-padding hidden state.

                Args:
                    inputs: float tensor ``(batch, seq_len, dim)``
                    input_ids: int tensor ``(batch, seq_len)``; 0 = padding

                Returns:
                    float tensor ``(batch, dim)``
                """
                import keras as _keras

                mask = _keras.ops.cast(
                    _keras.ops.not_equal(input_ids, 0), dtype="int32"
                )  # (batch, seq_len)

                seq_len = _keras.ops.shape(input_ids)[1]
                positions = _keras.ops.arange(seq_len, dtype="int32")  # (seq_len,)

                # -1 for padding positions so argmax finds the last real token
                masked_positions = _keras.ops.where(
                    _keras.ops.cast(mask, "bool"),
                    positions,
                    _keras.ops.full_like(positions, -1),
                )  # (batch, seq_len)

                last_idx = _keras.ops.argmax(masked_positions, axis=1)  # (batch,)

                # Gather: one-hot * inputs → sum over seq_len axis
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
) -> "keras.Model":
    """Build and compile the HSTU next-item prediction model.

    Args:
        vocab_size: Number of real items (padding token 0 not counted).
        max_sequence_length: Sequence length for input_ids.
        model_dim: Embedding / hidden dimension.
        num_heads: Number of HSTU attention heads.
        num_layers: Number of HSTU blocks.
        dropout: Dropout rate.
        learning_rate: Adam learning rate.

    Returns:
        Compiled Keras model.  Input: dict with key ``"input_ids"`` of shape
        ``(batch, max_sequence_length)``.  Output: logits of shape
        ``(batch, vocab_size + 1)`` (index 0 = padding; model never predicts it).
    """
    import keras  # lazy import
    from recml.layers.keras.hstu import HSTU

    input_ids = keras.Input(
        shape=(max_sequence_length,), dtype="int32", name="input_ids"
    )

    # HSTU expects vocab_size = embedding table rows.
    # We use vocab_size + 1 so that index 0 (padding) has its own row.
    hstu = HSTU(
        vocab_size=vocab_size + 1,
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        add_head=True,
        name="hstu",
    )

    # (batch, seq_len, vocab_size + 1)
    sequence_logits = hstu(input_ids, padding_mask=keras.ops.cast(input_ids, "bool"))

    # (batch, vocab_size + 1)
    logits = LastNonPaddingToken(name="last_token")(sequence_logits, input_ids)

    model = keras.Model(inputs={"input_ids": input_ids}, outputs=logits, name="hstu_rec")

    from hstu_rec.metrics import NDCGAtK

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[NDCGAtK(k=10)],
    )
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
