from __future__ import annotations

# keras and tensorflow are imported at the top level here — this module is
# always loaded inside TF-enabled function bodies (build_model / evaluate),
# never at package import time, so these imports are effectively lazy.
import keras
import tensorflow as tf


class NDCGAtK(keras.metrics.Metric):
    """Normalised Discounted Cumulative Gain @ K as a Keras metric.

    Expects:
        y_true: int32 tensor of shape (batch,) — ground-truth item IDs
        y_pred: float32 tensor of shape (batch, vocab_size+1) — logits
    """

    def __init__(self, k: int = 10, name: str | None = None, **kwargs):
        super().__init__(name=name or f"ndcg_at_{k}", **kwargs)
        self.k = k
        self._total = self.add_weight(name="total", initializer="zeros")
        self._count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        # top-k indices per example: (batch, k)
        _, top_k_ids = tf.math.top_k(y_pred, k=self.k)
        top_k_ids = tf.cast(top_k_ids, tf.int32)

        # position of true item in top-k (-1 if absent)
        y_true_exp = tf.expand_dims(y_true, axis=1)        # (batch, 1)
        matches = tf.equal(top_k_ids, y_true_exp)           # (batch, k)
        found = tf.reduce_any(matches, axis=1)              # (batch,)

        # rank (0-indexed) of the first match
        ranks = tf.argmax(tf.cast(matches, tf.int32), axis=1)  # (batch,)

        # NDCG contribution: log2(2) / log2(rank+2) = log(2) / log(rank+2)
        ranks_f = tf.cast(ranks, tf.float32)
        gain = tf.math.log(2.0) / tf.math.log(ranks_f + 2.0)
        gain = tf.where(found, gain, tf.zeros_like(gain))

        if sample_weight is not None:
            gain = gain * tf.cast(sample_weight, tf.float32)

        self._total.assign_add(tf.reduce_sum(gain))
        self._count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self._total, self._count)

    def reset_state(self):
        self._total.assign(0.0)
        self._count.assign(0.0)

    def get_config(self):
        return {"k": self.k, "name": self.name}


class HRAtK(keras.metrics.Metric):
    """Hit Rate @ K as a Keras metric.

    A hit is when the true item appears anywhere in the top-K predictions.

    Expects:
        y_true: int32 tensor of shape (batch,) — ground-truth item IDs
        y_pred: float32 tensor of shape (batch, vocab_size+1) — logits
    """

    def __init__(self, k: int = 10, name: str | None = None, **kwargs):
        super().__init__(name=name or f"hr_at_{k}", **kwargs)
        self.k = k
        self._total = self.add_weight(name="total", initializer="zeros")
        self._count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        _, top_k_ids = tf.math.top_k(y_pred, k=self.k)
        top_k_ids = tf.cast(top_k_ids, tf.int32)

        y_true_exp = tf.expand_dims(y_true, axis=1)           # (batch, 1)
        hits = tf.reduce_any(tf.equal(top_k_ids, y_true_exp), axis=1)  # (batch,)
        hits_f = tf.cast(hits, tf.float32)

        if sample_weight is not None:
            hits_f = hits_f * tf.cast(sample_weight, tf.float32)

        self._total.assign_add(tf.reduce_sum(hits_f))
        self._count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self._total, self._count)

    def reset_state(self):
        self._total.assign(0.0)
        self._count.assign(0.0)

    def get_config(self):
        return {"k": self.k, "name": self.name}
