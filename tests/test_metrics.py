"""NDCGAtK metric tests.

All tests require Keras/TF and are marked @pytest.mark.slow.
Run on GPU machine with: uv run pytest -m slow
"""
import pytest


@pytest.mark.slow
def test_ndcg_perfect_rank():
    """Target is the top-1 prediction → NDCG = 1.0."""
    import numpy as np
    import tensorflow as tf
    from hstu_rec.metrics import NDCGAtK

    metric = NDCGAtK(k=10)
    # batch of 2: target item is always index 0, logit is highest
    logits = np.zeros((2, 20), dtype="float32")
    logits[:, 0] = 10.0  # item 0 always ranked #1
    y_true = tf.constant([0, 0], dtype=tf.int32)
    metric.update_state(y_true, tf.constant(logits))
    assert metric.result().numpy() == pytest.approx(1.0)


@pytest.mark.slow
def test_ndcg_not_in_top_k():
    """Target is outside top-k → NDCG = 0.0."""
    import numpy as np
    import tensorflow as tf
    from hstu_rec.metrics import NDCGAtK

    metric = NDCGAtK(k=3)
    # 10 items; item 9 has highest logit, target is item 0 (rank >> 3)
    logits = np.arange(10, dtype="float32").reshape(1, 10)  # item 9 = highest
    y_true = tf.constant([0], dtype=tf.int32)
    metric.update_state(y_true, tf.constant(logits))
    assert metric.result().numpy() == pytest.approx(0.0)


@pytest.mark.slow
def test_ndcg_second_rank():
    """Target at rank 2 (0-indexed 1) → NDCG = log(2)/log(3)."""
    import math
    import numpy as np
    import tensorflow as tf
    from hstu_rec.metrics import NDCGAtK

    metric = NDCGAtK(k=10)
    logits = np.zeros((1, 10), dtype="float32")
    logits[0, 5] = 2.0  # rank 1 item
    logits[0, 3] = 1.0  # rank 2 item (target)
    y_true = tf.constant([3], dtype=tf.int32)
    metric.update_state(y_true, tf.constant(logits))
    expected = math.log(2) / math.log(3)
    assert metric.result().numpy() == pytest.approx(expected, rel=1e-5)


@pytest.mark.slow
def test_ndcg_reset_state():
    """reset_state() zeroes the accumulator."""
    import numpy as np
    import tensorflow as tf
    from hstu_rec.metrics import NDCGAtK

    metric = NDCGAtK(k=10)
    logits = np.zeros((2, 20), dtype="float32")
    logits[:, 0] = 10.0
    metric.update_state(tf.constant([0, 0]), tf.constant(logits))
    metric.reset_state()
    assert metric.result().numpy() == pytest.approx(0.0)


@pytest.mark.slow
def test_ndcg_name():
    from hstu_rec.metrics import NDCGAtK

    assert NDCGAtK(k=10).name == "ndcg_at_10"
    assert NDCGAtK(k=5, name="my_metric").name == "my_metric"


# ---------------------------------------------------------------------------
# HRAtK
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_hr_perfect():
    """True item is top-1 → HR@10 = 1.0."""
    import numpy as np
    import tensorflow as tf
    from hstu_rec.metrics import HRAtK

    metric = HRAtK(k=10)
    logits = np.zeros((3, 20), dtype="float32")
    logits[:, 0] = 10.0
    metric.update_state(tf.constant([0, 0, 0]), tf.constant(logits))
    assert metric.result().numpy() == pytest.approx(1.0)


@pytest.mark.slow
def test_hr_miss():
    """True item ranked outside top-3 → HR@3 = 0.0."""
    import numpy as np
    import tensorflow as tf
    from hstu_rec.metrics import HRAtK

    metric = HRAtK(k=3)
    logits = np.arange(10, dtype="float32").reshape(1, 10)  # item 9 highest
    metric.update_state(tf.constant([0]), tf.constant(logits))
    assert metric.result().numpy() == pytest.approx(0.0)


@pytest.mark.slow
def test_hr_partial():
    """Half of the batch hits → HR = 0.5."""
    import numpy as np
    import tensorflow as tf
    from hstu_rec.metrics import HRAtK

    metric = HRAtK(k=1)
    logits = np.zeros((2, 10), dtype="float32")
    logits[0, 3] = 10.0   # example 0: top-1 is item 3 ✓
    logits[1, 0] = 10.0   # example 1: top-1 is item 0, true is item 3 ✗
    metric.update_state(tf.constant([3, 3]), tf.constant(logits))
    assert metric.result().numpy() == pytest.approx(0.5)


@pytest.mark.slow
def test_hr_name():
    from hstu_rec.metrics import HRAtK

    assert HRAtK(k=10).name == "hr_at_10"
    assert HRAtK(k=5, name="my_hr").name == "my_hr"


@pytest.mark.slow
def test_metrics_are_keras_metric_instances():
    """NDCGAtK and HRAtK are proper keras.metrics.Metric subclasses."""
    import keras
    from hstu_rec.metrics import NDCGAtK, HRAtK

    assert isinstance(NDCGAtK(k=10), keras.metrics.Metric)
    assert isinstance(HRAtK(k=10), keras.metrics.Metric)


@pytest.mark.slow
def test_ndcg_batch_accumulation():
    """update_state can be called multiple times; result averages all batches."""
    import numpy as np
    import tensorflow as tf
    from hstu_rec.metrics import NDCGAtK

    metric = NDCGAtK(k=10)

    # Batch 1: perfect hit → gain = 1.0
    logits1 = np.zeros((2, 20), dtype="float32")
    logits1[:, 0] = 10.0
    metric.update_state(tf.constant([0, 0]), tf.constant(logits1))

    # Batch 2: complete miss → gain = 0.0
    logits2 = np.arange(20, dtype="float32").reshape(1, 20)  # item 19 is top
    metric.update_state(tf.constant([0]), tf.constant(logits2))  # item 0 is rank 20

    # Mean over 3 examples: (1 + 1 + 0) / 3
    assert metric.result().numpy() == pytest.approx(2 / 3, rel=1e-5)
