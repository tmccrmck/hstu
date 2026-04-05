# HSTU Recommender System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a next-item sequential recommender for the Amazon Reviews 2023 Video_Games dataset using RecML's HSTU layer, with a preprocessing CLI and a training CLI both driven by YAML config.

**Architecture:** Two independent CLI stages: `preprocess` (download → 5-core filter → TFRecords) and `train` (load TFRecords → run RecML KerasTrainer with HSTU). RecML is vendored as a git submodule. Config dataclasses loaded from YAML drive both stages; swapping to Books = swap `--config`.

**Tech Stack:** Python 3.12+, uv, JAX (GPU), Keras 3 (JAX backend), TensorFlow (data pipelines only), pandas, RecML (git submodule), PyYAML, requests.

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | uv project, deps, CLI entry points |
| `.env` | `KERAS_BACKEND=jax` + `PYTHONPATH=vendor/RecML` |
| `configs/video_games.yaml` | Dataset + model + training config |
| `configs/books.yaml` | Placeholder for future Books dataset |
| `vendor/RecML/` | RecML git submodule |
| `vendor/RecML/pyproject.toml` | Minimal stub making RecML pip-installable |
| `src/hstu_rec/__init__.py` | Empty package marker |
| `src/hstu_rec/dataset.py` | `DatasetConfig`, `ModelConfig`, `TrainingConfig`, `Config`, `load_config`, `TFRecordDataFactory`, `make_data_factory`, `parse_tfrecord_fn` |
| `src/hstu_rec/preprocess/__init__.py` | `main()` CLI entry point |
| `src/hstu_rec/preprocess/download.py` | `download(url, output_dir) -> Path` |
| `src/hstu_rec/preprocess/filter.py` | `filter_reviews(jsonl_path, min_interactions) -> pd.DataFrame` |
| `src/hstu_rec/preprocess/tfrecords.py` | `write_tfrecords(df, max_seq_len, output_dir)` |
| `src/hstu_rec/metrics.py` | `NDCGAtK` Keras metric |
| `src/hstu_rec/train.py` | `LastNonPaddingToken`, `HSTUTask`, `main()` |
| `tests/conftest.py` | Shared pytest fixtures |
| `tests/test_dataset.py` | Config loading, TFRecordDataFactory, parse_tfrecord_fn |
| `tests/preprocess/test_download.py` | download() with mocked network |
| `tests/preprocess/test_filter.py` | filter_reviews() with synthetic DataFrames |
| `tests/preprocess/test_tfrecords.py` | write_tfrecords() round-trip |
| `tests/test_metrics.py` | NDCGAtK correctness |
| `tests/test_train.py` | LastNonPaddingToken unit test; HSTUTask smoke test |

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.env`
- Create: `configs/video_games.yaml`
- Create: `configs/books.yaml`
- Create: `src/hstu_rec/__init__.py`
- Create: `src/hstu_rec/preprocess/__init__.py` (empty, populated in Task 6)
- Create: `tests/__init__.py`
- Create: `tests/preprocess/__init__.py`

- [ ] **Step 1: Add RecML as a git submodule**

```bash
git submodule add https://github.com/AI-Hypercomputer/RecML.git vendor/RecML
```

Expected: `vendor/RecML/` populated, `.gitmodules` created.

- [ ] **Step 2: Create a minimal pyproject.toml stub inside vendor/RecML**

RecML has no `pyproject.toml`, so uv cannot install it as a path dep without one.

```bash
cat > vendor/RecML/pyproject.toml << 'EOF'
[project]
name = "recml"
version = "0.1.0"
requires-python = ">=3.12"
EOF
```

- [ ] **Step 3: Create pyproject.toml**

```toml
[project]
name = "hstu-rec"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "keras>=3.9",
    "jax[cuda12]",
    "tensorflow>=2.19",
    "tensorflow-io-gcs-filesystem",
    "pandas>=2.0",
    "pyyaml>=6.0",
    "requests>=2.32",
    "tqdm>=4.67",
    "recml",
    # RecML transitive deps
    "absl-py>=2.0",
    "orbax-checkpoint>=0.11",
    "optax>=0.2",
    "fiddle>=0.3",
    "keras-hub>=0.20",
]

[project.scripts]
preprocess = "hstu_rec.preprocess:main"
train      = "hstu_rec.train:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/hstu_rec"]

[tool.uv.sources]
recml = { path = "vendor/RecML", editable = true }

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: Create .env**

```
KERAS_BACKEND=jax
PYTHONPATH=vendor/RecML
```

- [ ] **Step 5: Create configs/video_games.yaml**

```yaml
dataset:
  name: video_games
  review_url: https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Video_Games.jsonl.gz
  min_interactions: 5

model:
  vocab_size: null
  max_sequence_length: 50
  model_dim: 64
  num_heads: 4
  num_layers: 4
  dropout: 0.5
  learning_rate: 0.001

training:
  batch_size: 128
  train_steps: 10000
  steps_per_eval: 500
  steps_per_loop: 100
  model_dir: runs/video_games
```

- [ ] **Step 6: Create configs/books.yaml (placeholder)**

```yaml
dataset:
  name: books
  review_url: https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Books.jsonl.gz
  min_interactions: 5

model:
  vocab_size: null
  max_sequence_length: 50
  model_dim: 128
  num_heads: 4
  num_layers: 4
  dropout: 0.5
  learning_rate: 0.001

training:
  batch_size: 256
  train_steps: 50000
  steps_per_eval: 1000
  steps_per_loop: 100
  model_dir: runs/books
```

- [ ] **Step 7: Create empty package init files**

```bash
touch src/hstu_rec/__init__.py
mkdir -p src/hstu_rec/preprocess
touch src/hstu_rec/preprocess/__init__.py
mkdir -p tests/preprocess
touch tests/__init__.py tests/preprocess/__init__.py
```

- [ ] **Step 8: Install dependencies**

```bash
uv sync
```

Expected: all deps resolve and install without errors. If JAX CUDA wheels fail (no GPU on the build machine), use `jax[cpu]` temporarily and swap back before training.

- [ ] **Step 9: Verify RecML is importable**

```bash
uv run python -c "import recml; print(recml.KerasTask)"
```

Expected: `<class 'recml.core.training.keras_trainer.KerasTask'>`

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml .env configs/ src/ tests/ vendor/ .gitmodules
git commit -m "feat: project scaffold with uv, RecML submodule, configs"
```

---

## Task 2: Config Dataclasses and YAML Loader

**Files:**
- Create: `src/hstu_rec/dataset.py` (Config classes + `load_config` only)
- Create: `tests/test_dataset.py` (config loading tests)
- Create: `tests/conftest.py` (shared fixtures)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_dataset.py
import os
import pytest
import tempfile
import yaml
from hstu_rec.dataset import Config, DatasetConfig, ModelConfig, TrainingConfig, load_config


@pytest.fixture
def video_games_config_path():
    return os.path.join(os.path.dirname(__file__), "..", "configs", "video_games.yaml")


def test_load_config_returns_config_instance(video_games_config_path):
    config = load_config(video_games_config_path)
    assert isinstance(config, Config)
    assert isinstance(config.dataset, DatasetConfig)
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.training, TrainingConfig)


def test_load_config_dataset_fields(video_games_config_path):
    config = load_config(video_games_config_path)
    assert config.dataset.name == "video_games"
    assert "Video_Games.jsonl.gz" in config.dataset.review_url
    assert config.dataset.min_interactions == 5


def test_load_config_model_fields(video_games_config_path):
    config = load_config(video_games_config_path)
    assert config.model.vocab_size is None
    assert config.model.max_sequence_length == 50
    assert config.model.model_dim == 64
    assert config.model.num_heads == 4
    assert config.model.num_layers == 4
    assert config.model.dropout == 0.5
    assert config.model.learning_rate == pytest.approx(1e-3)


def test_load_config_training_fields(video_games_config_path):
    config = load_config(video_games_config_path)
    assert config.training.batch_size == 128
    assert config.training.train_steps == 10000
    assert config.training.steps_per_eval == 500
    assert config.training.model_dir == "runs/video_games"


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")
```

```python
# tests/conftest.py
import pandas as pd
import pytest


@pytest.fixture
def small_interactions_df():
    """5 users, each with exactly 7 interactions over 7 items. All pass 5-core."""
    rows = []
    for user_idx in range(5):
        for item_idx in range(7):
            rows.append({
                "user_id": f"user_{user_idx}",
                "parent_asin": f"item_{item_idx}",
                "timestamp": item_idx * 1000,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sparse_interactions_df(small_interactions_df):
    """Same as small_interactions_df but with an extra sparse user and sparse item."""
    sparse = pd.DataFrame([
        {"user_id": "sparse_user", "parent_asin": "item_0", "timestamp": 9999},
        {"user_id": "user_0", "parent_asin": "sparse_item", "timestamp": 9999},
    ])
    return pd.concat([small_interactions_df, sparse], ignore_index=True)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_dataset.py -v
```

Expected: `ImportError: cannot import name 'Config' from 'hstu_rec.dataset'`

- [ ] **Step 3: Implement dataset.py (config classes only)**

```python
# src/hstu_rec/dataset.py
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import yaml


@dataclasses.dataclass
class DatasetConfig:
    name: str
    review_url: str
    min_interactions: int


@dataclasses.dataclass
class ModelConfig:
    vocab_size: Optional[int]
    max_sequence_length: int
    model_dim: int
    num_heads: int
    num_layers: int
    dropout: float
    learning_rate: float


@dataclasses.dataclass
class TrainingConfig:
    batch_size: int
    train_steps: int
    steps_per_eval: int
    steps_per_loop: int
    model_dir: str


@dataclasses.dataclass
class Config:
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig


def load_config(path: str) -> Config:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(p) as f:
        raw = yaml.safe_load(f)
    return Config(
        dataset=DatasetConfig(**raw["dataset"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_dataset.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hstu_rec/dataset.py tests/test_dataset.py tests/conftest.py
git commit -m "feat: Config dataclasses and YAML loader"
```

---

## Task 3: Download Module

**Files:**
- Create: `src/hstu_rec/preprocess/download.py`
- Create: `tests/preprocess/test_download.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/preprocess/test_download.py
import gzip
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from hstu_rec.preprocess.download import download


def _make_gz_bytes(content: str) -> bytes:
    return gzip.compress(content.encode())


def test_download_creates_cache_dir_and_returns_path(tmp_path):
    url = "https://example.com/Video_Games.jsonl.gz"
    gz_bytes = _make_gz_bytes('{"user_id": "u1"}\n')

    with patch("hstu_rec.preprocess.download.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = [gz_bytes]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value.__enter__ = lambda s: mock_resp
        mock_get.return_value.__exit__ = MagicMock(return_value=False)

        result = download(url, str(tmp_path))

    assert result == tmp_path / "cache" / "Video_Games.jsonl"
    assert result.exists()
    assert result.read_text().strip() == '{"user_id": "u1"}'


def test_download_skips_if_already_exists(tmp_path):
    url = "https://example.com/Video_Games.jsonl.gz"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    existing = cache_dir / "Video_Games.jsonl"
    existing.write_text("existing content")

    with patch("hstu_rec.preprocess.download.requests.get") as mock_get:
        result = download(url, str(tmp_path))
        mock_get.assert_not_called()

    assert result == existing
    assert result.read_text() == "existing content"


def test_download_derives_filename_from_url(tmp_path):
    url = "https://example.com/some/path/Books.jsonl.gz"
    gz_bytes = _make_gz_bytes('{"x": 1}\n')

    with patch("hstu_rec.preprocess.download.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = [gz_bytes]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value.__enter__ = lambda s: mock_resp
        mock_get.return_value.__exit__ = MagicMock(return_value=False)

        result = download(url, str(tmp_path))

    assert result.name == "Books.jsonl"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/preprocess/test_download.py -v
```

Expected: `ImportError: cannot import name 'download'`

- [ ] **Step 3: Implement download.py**

```python
# src/hstu_rec/preprocess/download.py
from __future__ import annotations

import gzip
import shutil
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm


def download(url: str, output_dir: str) -> Path:
    """Download and decompress a .jsonl.gz file.

    Skips download if the decompressed .jsonl already exists.

    Args:
        url: URL to a .jsonl.gz file.
        output_dir: Root output directory. File is written to output_dir/cache/<name>.jsonl.

    Returns:
        Path to the decompressed .jsonl file.
    """
    filename_gz = Path(urlparse(url).path).name          # e.g. "Video_Games.jsonl.gz"
    filename_jsonl = filename_gz.replace(".jsonl.gz", ".jsonl")
    cache_dir = Path(output_dir) / "cache"
    jsonl_path = cache_dir / filename_jsonl

    if jsonl_path.exists():
        print(f"Cache hit: {jsonl_path}")
        return jsonl_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    gz_path = cache_dir / filename_gz

    print(f"Downloading {url} ...")
    with requests.get(url, stream=True) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(gz_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

    print(f"Decompressing to {jsonl_path} ...")
    with gzip.open(gz_path, "rb") as f_in, open(jsonl_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    gz_path.unlink()  # remove the .gz after decompression
    return jsonl_path
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/preprocess/test_download.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hstu_rec/preprocess/download.py tests/preprocess/test_download.py
git commit -m "feat: download module with cache-skip"
```

---

## Task 4: Filter Module

**Files:**
- Create: `src/hstu_rec/preprocess/filter.py`
- Create: `tests/preprocess/test_filter.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/preprocess/test_filter.py
import json
import tempfile
from pathlib import Path
import pandas as pd
import pytest
from hstu_rec.preprocess.filter import filter_reviews


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


@pytest.fixture
def dense_jsonl(tmp_path):
    """5 users × 7 items = 35 rows; all pass 5-core."""
    rows = []
    for u in range(5):
        for i in range(7):
            rows.append({"user_id": f"u{u}", "parent_asin": f"item{i}", "timestamp": i})
    path = tmp_path / "reviews.jsonl"
    _write_jsonl(path, rows)
    return path


@pytest.fixture
def sparse_jsonl(tmp_path, dense_jsonl):
    """dense_jsonl + a sparse user (1 interaction) and sparse item (1 interaction)."""
    extra = [
        {"user_id": "sparse_user", "parent_asin": "item0", "timestamp": 999},
        {"user_id": "u0", "parent_asin": "sparse_item", "timestamp": 999},
    ]
    path = tmp_path / "sparse.jsonl"
    # Re-read dense rows
    rows = []
    with open(dense_jsonl) as f:
        for line in f:
            rows.append(json.loads(line))
    _write_jsonl(path, rows + extra)
    return path


def test_filter_reviews_returns_dataframe_with_correct_columns(dense_jsonl):
    df = filter_reviews(str(dense_jsonl), min_interactions=5)
    assert list(df.columns) == ["user_id", "parent_asin", "timestamp"]


def test_filter_reviews_passes_dense_dataset_unchanged(dense_jsonl):
    df = filter_reviews(str(dense_jsonl), min_interactions=5)
    assert df["user_id"].nunique() == 5
    assert df["parent_asin"].nunique() == 7
    assert len(df) == 35


def test_filter_reviews_removes_sparse_users_and_items(sparse_jsonl):
    df = filter_reviews(str(sparse_jsonl), min_interactions=5)
    assert "sparse_user" not in df["user_id"].values
    assert "sparse_item" not in df["parent_asin"].values


def test_filter_reviews_converges_iteratively(tmp_path):
    """Removing a sparse user may drop an item below threshold, which in turn
    drops another user, etc. Verify iterative convergence."""
    rows = []
    # 5 users × 6 common items (survive filtering)
    for u in range(5):
        for i in range(6):
            rows.append({"user_id": f"u{u}", "parent_asin": f"item{i}", "timestamp": i})
    # 1 user with only 5 interactions, but 4 of them on unique items
    # that are only shared with this user -> cascades on removal
    rows += [
        {"user_id": "cascade_user", "parent_asin": "item0", "timestamp": 0},
        {"user_id": "cascade_user", "parent_asin": "item1", "timestamp": 1},
        {"user_id": "cascade_user", "parent_asin": "item2", "timestamp": 2},
        {"user_id": "cascade_user", "parent_asin": "item3", "timestamp": 3},
        {"user_id": "cascade_user", "parent_asin": "cascade_only_item", "timestamp": 4},
    ]
    path = tmp_path / "cascade.jsonl"
    _write_jsonl(path, rows)
    df = filter_reviews(str(path), min_interactions=5)
    assert "cascade_user" not in df["user_id"].values
    assert "cascade_only_item" not in df["parent_asin"].values


def test_filter_reviews_column_types(dense_jsonl):
    df = filter_reviews(str(dense_jsonl), min_interactions=5)
    assert df["user_id"].dtype == object   # str
    assert df["parent_asin"].dtype == object  # str
    assert pd.api.types.is_integer_dtype(df["timestamp"])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/preprocess/test_filter.py -v
```

Expected: `ImportError: cannot import name 'filter_reviews'`

- [ ] **Step 3: Implement filter.py**

```python
# src/hstu_rec/preprocess/filter.py
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def filter_reviews(jsonl_path: str, min_interactions: int) -> pd.DataFrame:
    """Load a .jsonl review file and apply iterative k-core filtering.

    Reads only user_id, parent_asin, and timestamp fields. Repeatedly removes
    users and items with fewer than min_interactions interactions until
    convergence.

    Args:
        jsonl_path: Path to the decompressed .jsonl file.
        min_interactions: Minimum number of interactions for users and items.

    Returns:
        Filtered DataFrame with columns [user_id, parent_asin, timestamp].
    """
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            obj = json.loads(line)
            rows.append({
                "user_id": str(obj["user_id"]),
                "parent_asin": str(obj["parent_asin"]),
                "timestamp": int(obj["timestamp"]),
            })
    df = pd.DataFrame(rows, columns=["user_id", "parent_asin", "timestamp"])

    while True:
        user_counts = df["user_id"].value_counts()
        item_counts = df["parent_asin"].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        valid_items = item_counts[item_counts >= min_interactions].index
        filtered = df[
            df["user_id"].isin(valid_users) & df["parent_asin"].isin(valid_items)
        ]
        if len(filtered) == len(df):
            break
        df = filtered.reset_index(drop=True)

    return df.reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/preprocess/test_filter.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hstu_rec/preprocess/filter.py tests/preprocess/test_filter.py
git commit -m "feat: iterative 5-core filter for review datasets"
```

---

## Task 5: TFRecords Module

**Files:**
- Create: `src/hstu_rec/preprocess/tfrecords.py`
- Create: `tests/preprocess/test_tfrecords.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/preprocess/test_tfrecords.py
import json
from pathlib import Path
import pandas as pd
import pytest
import tensorflow as tf
from hstu_rec.preprocess.tfrecords import write_tfrecords


@pytest.fixture
def five_users_df():
    """5 users, each with 7 items in order. Items shared across users."""
    rows = []
    for u in range(5):
        for i in range(7):
            rows.append({
                "user_id": f"u{u}",
                "parent_asin": f"ASIN{i}",
                "timestamp": i * 1000,
            })
    return pd.DataFrame(rows)


def test_write_tfrecords_creates_expected_files(five_users_df, tmp_path):
    write_tfrecords(five_users_df, max_seq_len=50, output_dir=str(tmp_path))
    assert (tmp_path / "train.tfrecord").exists()
    assert (tmp_path / "val.tfrecord").exists()
    assert (tmp_path / "test.tfrecord").exists()
    assert (tmp_path / "item_map.json").exists()
    assert (tmp_path / "vocab_size.txt").exists()


def test_write_tfrecords_vocab_size(five_users_df, tmp_path):
    write_tfrecords(five_users_df, max_seq_len=50, output_dir=str(tmp_path))
    vocab_size = int((tmp_path / "vocab_size.txt").read_text().strip())
    assert vocab_size == 7  # 7 unique items (ASIN0..ASIN6)


def test_write_tfrecords_item_map(five_users_df, tmp_path):
    write_tfrecords(five_users_df, max_seq_len=50, output_dir=str(tmp_path))
    item_map = json.loads((tmp_path / "item_map.json").read_text())
    assert set(item_map.keys()) == {f"ASIN{i}" for i in range(7)}
    # IDs are 1-indexed; 0 is padding
    assert min(item_map.values()) == 1
    assert max(item_map.values()) == 7
    assert len(set(item_map.values())) == 7  # no duplicates


def _read_tfrecord(path: Path, max_seq_len: int):
    """Read all examples from a TFRecord file."""
    ds = tf.data.TFRecordDataset(str(path))
    features = {
        "input_ids": tf.io.FixedLenFeature([max_seq_len], tf.int64),
        "target_id": tf.io.FixedLenFeature([1], tf.int64),
    }
    examples = []
    for raw in ds:
        parsed = tf.io.parse_single_example(raw, features)
        examples.append({
            "input_ids": parsed["input_ids"].numpy().tolist(),
            "target_id": int(parsed["target_id"].numpy()[0]),
        })
    return examples


def test_val_has_one_example_per_user(five_users_df, tmp_path):
    write_tfrecords(five_users_df, max_seq_len=50, output_dir=str(tmp_path))
    examples = _read_tfrecord(tmp_path / "val.tfrecord", max_seq_len=50)
    # Each of the 5 users contributes exactly one val example
    assert len(examples) == 5


def test_test_has_one_example_per_user(five_users_df, tmp_path):
    write_tfrecords(five_users_df, max_seq_len=50, output_dir=str(tmp_path))
    examples = _read_tfrecord(tmp_path / "test.tfrecord", max_seq_len=50)
    assert len(examples) == 5


def test_train_example_count(five_users_df, tmp_path):
    # Each user has L=7 interactions → L-3=4 training examples
    write_tfrecords(five_users_df, max_seq_len=50, output_dir=str(tmp_path))
    examples = _read_tfrecord(tmp_path / "train.tfrecord", max_seq_len=50)
    assert len(examples) == 5 * 4  # 5 users × 4 examples each


def test_input_ids_are_left_padded(five_users_df, tmp_path):
    write_tfrecords(five_users_df, max_seq_len=50, output_dir=str(tmp_path))
    examples = _read_tfrecord(tmp_path / "train.tfrecord", max_seq_len=50)
    # First training example per user has only 1 real item → 49 padding tokens
    first_examples = [e for e in examples if sum(1 for x in e["input_ids"] if x == 0) == 49]
    assert len(first_examples) == 5


def test_target_id_is_real_item(five_users_df, tmp_path):
    write_tfrecords(five_users_df, max_seq_len=50, output_dir=str(tmp_path))
    for split in ["train", "val", "test"]:
        examples = _read_tfrecord(tmp_path / f"{split}.tfrecord", max_seq_len=50)
        for ex in examples:
            assert ex["target_id"] >= 1  # never padding token
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/preprocess/test_tfrecords.py -v
```

Expected: `ImportError: cannot import name 'write_tfrecords'`

- [ ] **Step 3: Implement tfrecords.py**

```python
# src/hstu_rec/preprocess/tfrecords.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


def _make_example(input_ids: list[int], target_id: int) -> tf.train.Example:
    return tf.train.Example(features=tf.train.Features(feature={
        "input_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=input_ids)),
        "target_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[target_id])),
    }))


def _pad_left(seq: list[int], max_len: int) -> list[int]:
    """Left-pad sequence with zeros to max_len, or truncate from the left."""
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq


def write_tfrecords(df: pd.DataFrame, max_seq_len: int, output_dir: str) -> None:
    """Write train/val/test TFRecords and metadata from a filtered review DataFrame.

    Args:
        df: Filtered DataFrame with columns [user_id, parent_asin, timestamp].
        max_seq_len: Length of input_ids in each TFRecord example.
        output_dir: Directory to write all output files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Assign item IDs: 0 = padding, real items start at 1
    unique_asins = sorted(df["parent_asin"].unique())  # sorted for determinism
    item_map = {asin: idx + 1 for idx, asin in enumerate(unique_asins)}
    vocab_size = len(item_map)

    # Save metadata
    (out / "item_map.json").write_text(json.dumps(item_map))
    (out / "vocab_size.txt").write_text(str(vocab_size))

    train_writer = tf.io.TFRecordWriter(str(out / "train.tfrecord"))
    val_writer = tf.io.TFRecordWriter(str(out / "val.tfrecord"))
    test_writer = tf.io.TFRecordWriter(str(out / "test.tfrecord"))

    for user_id, user_df in df.groupby("user_id"):
        # Sort by timestamp and encode to integer IDs
        seq = [
            item_map[asin]
            for asin in user_df.sort_values("timestamp")["parent_asin"]
        ]
        L = len(seq)

        # Test: input = seq[:-1] truncated to max_seq_len, target = seq[-1]
        test_writer.write(
            _make_example(_pad_left(seq[:-1], max_seq_len), seq[-1]).SerializeToString()
        )

        # Val: input = seq[:-2] truncated to max_seq_len, target = seq[-2]
        val_writer.write(
            _make_example(_pad_left(seq[:-2], max_seq_len), seq[-2]).SerializeToString()
        )

        # Train: sliding window over positions 1..L-3 (inclusive)
        for j in range(1, L - 2):
            train_writer.write(
                _make_example(_pad_left(seq[:j], max_seq_len), seq[j]).SerializeToString()
            )

    train_writer.close()
    val_writer.close()
    test_writer.close()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/preprocess/test_tfrecords.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hstu_rec/preprocess/tfrecords.py tests/preprocess/test_tfrecords.py
git commit -m "feat: TFRecord writer with leave-last-out split"
```

---

## Task 6: Preprocess CLI

**Files:**
- Modify: `src/hstu_rec/preprocess/__init__.py`

- [ ] **Step 1: Write a failing integration test**

```python
# tests/preprocess/test_download.py  (append to existing file)

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import gzip


def test_preprocess_main_end_to_end(tmp_path):
    """Integration test: run main() with synthetic JSONL data (mocked download)."""
    import yaml
    from hstu_rec.preprocess import main

    # Write a minimal config pointing to a fake URL
    config = {
        "dataset": {
            "name": "test",
            "review_url": "https://fake.example.com/Test.jsonl.gz",
            "min_interactions": 2,
        },
        "model": {
            "vocab_size": None,
            "max_sequence_length": 4,
            "model_dim": 8,
            "num_heads": 2,
            "num_layers": 1,
            "dropout": 0.0,
            "learning_rate": 1e-3,
        },
        "training": {
            "batch_size": 2,
            "train_steps": 10,
            "steps_per_eval": 5,
            "steps_per_loop": 5,
            "model_dir": str(tmp_path / "runs"),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config))

    # Synthetic JSONL: 3 users × 4 items each (passes min=2 core)
    rows = []
    for u in range(3):
        for i in range(4):
            rows.append({"user_id": f"u{u}", "parent_asin": f"A{i}", "timestamp": i})
    jsonl_content = "\n".join(json.dumps(r) for r in rows)
    gz_bytes = gzip.compress(jsonl_content.encode())

    output_dir = tmp_path / "data"
    with patch("hstu_rec.preprocess.download.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = [gz_bytes]
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value.__enter__ = lambda s: mock_resp
        mock_get.return_value.__exit__ = MagicMock(return_value=False)

        main(str(config_path), str(output_dir))

    assert (output_dir / "train.tfrecord").exists()
    assert (output_dir / "val.tfrecord").exists()
    assert (output_dir / "test.tfrecord").exists()
    assert (output_dir / "vocab_size.txt").exists()
    vocab_size = int((output_dir / "vocab_size.txt").read_text().strip())
    assert vocab_size == 4
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/preprocess/test_download.py::test_preprocess_main_end_to_end -v
```

Expected: fails because `main` is not yet implemented in `preprocess/__init__.py`.

- [ ] **Step 3: Implement preprocess/__init__.py**

```python
# src/hstu_rec/preprocess/__init__.py
from __future__ import annotations

import argparse

from hstu_rec.dataset import load_config
from hstu_rec.preprocess import download as _download
from hstu_rec.preprocess import filter as _filter
from hstu_rec.preprocess import tfrecords as _tfrecords


def main(config_path: str | None = None, output_dir: str | None = None) -> None:
    """CLI entry point for the preprocessing stage."""
    if config_path is None or output_dir is None:
        parser = argparse.ArgumentParser(description="Preprocess Amazon Reviews data")
        parser.add_argument("--config", required=True, help="Path to YAML config file")
        parser.add_argument("--output", required=True, help="Output directory for TFRecords")
        args = parser.parse_args()
        config_path = args.config
        output_dir = args.output

    config = load_config(config_path)

    jsonl_path = _download.download(config.dataset.review_url, output_dir)
    df = _filter.filter_reviews(str(jsonl_path), config.dataset.min_interactions)
    _tfrecords.write_tfrecords(df, config.model.max_sequence_length, output_dir)

    print(f"Preprocessing complete. Output written to {output_dir}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/preprocess/test_download.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Verify CLI is wired up**

```bash
uv run preprocess --help
```

Expected: prints usage with `--config` and `--output` arguments.

- [ ] **Step 6: Commit**

```bash
git add src/hstu_rec/preprocess/__init__.py
git commit -m "feat: preprocess CLI main() entry point"
```

---

## Task 7: TFRecordDataFactory and parse_tfrecord_fn

**Files:**
- Modify: `src/hstu_rec/dataset.py`
- Modify: `tests/test_dataset.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dataset.py`:

```python
# tests/test_dataset.py (additions)
import json
import tempfile
from pathlib import Path
import tensorflow as tf
import pytest
from hstu_rec.dataset import (
    Config, load_config, make_data_factory, parse_tfrecord_fn, TFRecordDataFactory
)
from hstu_rec.preprocess.tfrecords import write_tfrecords


@pytest.fixture
def tiny_tfrecords(tmp_path):
    """3 users × 5 items → writes TFRecords to tmp_path. Returns (data_dir, config)."""
    import pandas as pd
    rows = []
    for u in range(3):
        for i in range(5):
            rows.append({"user_id": f"u{u}", "parent_asin": f"A{i}", "timestamp": i})
    df = pd.DataFrame(rows)
    write_tfrecords(df, max_seq_len=4, output_dir=str(tmp_path))
    return tmp_path


def test_make_data_factory_returns_factory(tmp_path, tiny_tfrecords, video_games_config_path):
    config = load_config(video_games_config_path)
    config.model.max_sequence_length = 4
    config.training.batch_size = 2
    factory = make_data_factory(config, str(tiny_tfrecords), "train")
    assert isinstance(factory, TFRecordDataFactory)


def test_factory_make_returns_tf_dataset(tmp_path, tiny_tfrecords, video_games_config_path):
    config = load_config(video_games_config_path)
    config.model.max_sequence_length = 4
    config.training.batch_size = 2
    factory = make_data_factory(config, str(tiny_tfrecords), "train")
    ds = factory.make()
    assert isinstance(ds, tf.data.Dataset)


def test_factory_yields_correct_shapes(tmp_path, tiny_tfrecords, video_games_config_path):
    config = load_config(video_games_config_path)
    config.model.max_sequence_length = 4
    config.training.batch_size = 2
    factory = make_data_factory(config, str(tiny_tfrecords), "val")
    ds = factory.make()
    for x_batch, y_batch in ds.take(1):
        assert x_batch["input_ids"].shape == (2, 4)
        assert y_batch.shape == (2,)
        assert x_batch["input_ids"].dtype == tf.int32
        assert y_batch.dtype == tf.int32


def test_parse_tfrecord_fn_casts_to_int32(tmp_path, tiny_tfrecords):
    ds = tf.data.TFRecordDataset(str(tiny_tfrecords / "val.tfrecord"))
    parse_fn = parse_tfrecord_fn(max_sequence_length=4)
    for raw in ds.take(1):
        x, y = parse_fn(raw)
        assert x["input_ids"].dtype == tf.int32
        assert y.dtype == tf.int32
        assert x["input_ids"].shape == (4,)
        assert y.shape == ()


def test_training_factory_is_training_flag(tmp_path, tiny_tfrecords, video_games_config_path):
    config = load_config(video_games_config_path)
    config.model.max_sequence_length = 4
    config.training.batch_size = 2
    train_factory = make_data_factory(config, str(tiny_tfrecords), "train")
    val_factory = make_data_factory(config, str(tiny_tfrecords), "val")
    assert train_factory.is_training is True
    assert val_factory.is_training is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_dataset.py -v -k "factory or parse"
```

Expected: `ImportError: cannot import name 'TFRecordDataFactory'`

- [ ] **Step 3: Add TFRecordDataFactory and related functions to dataset.py**

Append to `src/hstu_rec/dataset.py`:

```python
import dataclasses
from typing import Callable
import tensorflow as tf


@dataclasses.dataclass
class TFRecordDataFactory:
    """Builds a tf.data.Dataset from TFRecord files for a given split."""
    path: str
    batch_size: int
    max_sequence_length: int
    is_training: bool

    def make(self) -> tf.data.Dataset:
        ds = tf.data.TFRecordDataset(
            tf.data.Dataset.list_files(self.path, shuffle=self.is_training),
            num_parallel_reads=tf.data.AUTOTUNE,
        )
        parse_fn = parse_tfrecord_fn(self.max_sequence_length)
        ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
        if self.is_training:
            ds = ds.shuffle(buffer_size=10_000).repeat()
        return ds.batch(self.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


def parse_tfrecord_fn(max_sequence_length: int) -> Callable:
    """Returns a function that parses a serialized TFRecord example.

    Returns:
        A callable mapping raw bytes → ({"input_ids": int32[max_seq_len]}, int32[]).
        input_ids is zero-padded on the left; target is a scalar.
    """
    feature_spec = {
        "input_ids": tf.io.FixedLenFeature([max_sequence_length], tf.int64),
        "target_id": tf.io.FixedLenFeature([1], tf.int64),
    }

    def _parse(serialized: tf.Tensor):
        parsed = tf.io.parse_single_example(serialized, feature_spec)
        input_ids = tf.cast(parsed["input_ids"], tf.int32)
        target_id = tf.cast(tf.squeeze(parsed["target_id"], axis=0), tf.int32)
        return {"input_ids": input_ids}, target_id

    return _parse


def make_data_factory(config: "Config", data_dir: str, split: str) -> TFRecordDataFactory:
    """Construct a TFRecordDataFactory for a given split.

    Args:
        config: Loaded Config object.
        data_dir: Directory containing train/val/test.tfrecord files.
        split: One of "train", "val", or "test".

    Returns:
        A TFRecordDataFactory ready to call .make() on.
    """
    return TFRecordDataFactory(
        path=f"{data_dir}/{split}.tfrecord",
        batch_size=config.training.batch_size,
        max_sequence_length=config.model.max_sequence_length,
        is_training=(split == "train"),
    )
```

- [ ] **Step 4: Run all dataset tests to verify they pass**

```bash
uv run pytest tests/test_dataset.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hstu_rec/dataset.py tests/test_dataset.py
git commit -m "feat: TFRecordDataFactory, parse_tfrecord_fn, make_data_factory"
```

---

## Task 8: NDCGAtK Metric

**Files:**
- Create: `src/hstu_rec/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics.py
import numpy as np
import pytest
import keras
from hstu_rec.metrics import NDCGAtK


def _logits_with_target_at_rank(vocab_size: int, target_id: int, rank: int) -> np.ndarray:
    """Return a logit vector where target_id is ranked exactly `rank` (1-indexed)."""
    logits = np.zeros(vocab_size, dtype=np.float32)
    # Assign logit values so target is at position `rank` when sorted descending
    for i in range(vocab_size):
        if i == target_id:
            logits[i] = vocab_size - rank  # rank-th highest
        else:
            idx = i if i < target_id else i - 1  # shift around target
            logits[i] = vocab_size - (idx + 1 if idx < rank - 1 else idx + 2)
    return logits


def test_ndcg_at_k_perfect_rank():
    """Target at rank 1 → NDCG = 1/log2(2) = 1.0."""
    metric = NDCGAtK(k=10)
    vocab_size = 20
    # Put target_id=1 at rank 1: give it the highest logit
    logits = np.array([-float(i) for i in range(vocab_size)], dtype=np.float32)
    # logits[0] = 0 (rank 1), logits[1] = -1, ...
    y_true = np.array([0], dtype=np.int32)   # target is item 0
    y_pred = np.expand_dims(logits, 0)        # [1, vocab_size]
    metric.update_state(y_true, y_pred)
    result = float(metric.result())
    assert abs(result - 1.0) < 1e-5


def test_ndcg_at_k_target_beyond_k():
    """Target at rank > k → NDCG = 0."""
    metric = NDCGAtK(k=5)
    vocab_size = 20
    # Item 19 has highest logit, item 0 (our target) has lowest
    logits = np.array([float(i) for i in range(vocab_size)], dtype=np.float32)
    # logits[19]=19 (rank 1), ..., logits[0]=0 (rank 20)
    y_true = np.array([0], dtype=np.int32)
    y_pred = np.expand_dims(logits, 0)
    metric.update_state(y_true, y_pred)
    result = float(metric.result())
    assert result == 0.0


def test_ndcg_at_k_batch_average():
    """Average of rank-1 and rank-beyond-k should be 0.5."""
    metric = NDCGAtK(k=5)
    vocab_size = 10
    # Example 1: target=0 at rank 1 → NDCG = 1.0
    logits1 = np.array([float(vocab_size - i) for i in range(vocab_size)], dtype=np.float32)
    # Example 2: target=0 at rank 10 (beyond k=5) → NDCG = 0.0
    logits2 = np.array([float(i) for i in range(vocab_size)], dtype=np.float32)
    y_true = np.array([0, 0], dtype=np.int32)
    y_pred = np.stack([logits1, logits2], axis=0)
    metric.update_state(y_true, y_pred)
    result = float(metric.result())
    assert abs(result - 0.5) < 1e-5


def test_ndcg_at_k_reset_state():
    metric = NDCGAtK(k=10)
    vocab_size = 10
    logits = np.zeros((1, vocab_size), dtype=np.float32)
    y_true = np.array([0], dtype=np.int32)
    metric.update_state(y_true, logits)
    metric.reset_state()
    assert float(metric.result()) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_metrics.py -v
```

Expected: `ImportError: cannot import name 'NDCGAtK'`

- [ ] **Step 3: Implement metrics.py**

```python
# src/hstu_rec/metrics.py
from __future__ import annotations

import keras


class NDCGAtK(keras.metrics.Metric):
    """Normalized Discounted Cumulative Gain at K for next-item prediction.

    Inputs:
        y_true: int32[batch] — ground-truth item IDs (scalars, 0-indexed into vocab).
        y_pred: float32[batch, vocab_size] — raw logits over the full vocabulary.

    Ranking is performed over all vocab_size positions including the padding
    token (index 0), consistent with SparseTopKCategoricalAccuracy.
    """

    def __init__(self, k: int, name: str = "ndcg_at_k", **kwargs):
        super().__init__(name=name, **kwargs)
        self.k = k
        self._sum = self.add_weight(name="sum", initializer="zeros")
        self._count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        import keras.ops as ops

        # Rank of the ground-truth item: count items with higher logit + 1
        # y_true: [batch], y_pred: [batch, vocab]
        y_true_expanded = ops.cast(ops.expand_dims(y_true, axis=1), "int32")  # [batch, 1]
        true_logits = ops.take_along_axis(
            ops.cast(y_pred, "float32"),
            y_true_expanded,
            axis=1,
        )  # [batch, 1]

        # Number of items with strictly higher logit than the target
        higher = ops.cast(
            ops.sum(
                ops.cast(y_pred > true_logits, "int32"),
                axis=1,
            ),
            "float32",
        )  # [batch]
        ranks = higher + 1.0  # 1-indexed rank

        # NDCG contribution: 1/log2(rank+1) if rank <= k, else 0
        import keras.ops as ops
        import math

        within_k = ops.cast(ranks <= self.k, "float32")
        ndcg_per_example = within_k / ops.log(ranks + 1.0) * math.log(2.0)

        self._sum.assign_add(ops.sum(ndcg_per_example))
        self._count.assign_add(ops.cast(ops.shape(y_true)[0], "float32"))

    def result(self):
        return keras.ops.divide_no_nan(self._sum, self._count)

    def reset_state(self):
        self._sum.assign(0.0)
        self._count.assign(0.0)

    def get_config(self):
        return {**super().get_config(), "k": self.k}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_metrics.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hstu_rec/metrics.py tests/test_metrics.py
git commit -m "feat: NDCGAtK Keras metric"
```

---

## Task 9: LastNonPaddingToken Layer

**Files:**
- Create: `src/hstu_rec/train.py` (layer only — HSTUTask and main() added in Tasks 10-11)
- Create: `tests/test_train.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train.py
import numpy as np
import pytest
import keras
from hstu_rec.train import LastNonPaddingToken


def test_last_non_padding_token_basic():
    """Verify it extracts the logit vector at the last non-padding position."""
    layer = LastNonPaddingToken()
    batch, seq_len, vocab = 2, 5, 10

    # seq_logits: [batch=2, seq=5, vocab=10]
    seq_logits = np.arange(batch * seq_len * vocab, dtype=np.float32).reshape(batch, seq_len, vocab)

    # padding_mask: [batch=2, seq=5]
    # Example 0: positions 0-1 are padding (mask=0), positions 2-4 are real → last real = 4
    # Example 1: positions 0-3 are padding, position 4 is real → last real = 4
    padding_mask = np.array([
        [0, 0, 1, 1, 1],  # last non-padding index = 4
        [0, 0, 0, 0, 1],  # last non-padding index = 4
    ], dtype=np.int32)

    result = layer([seq_logits, padding_mask])
    assert result.shape == (2, vocab)

    # Example 0: last real position is index 4 → logits[0, 4, :]
    np.testing.assert_array_equal(result[0], seq_logits[0, 4, :])
    # Example 1: last real position is index 4 → logits[1, 4, :]
    np.testing.assert_array_equal(result[1], seq_logits[1, 4, :])


def test_last_non_padding_token_single_real_token():
    """Works when only the last token in the sequence is real."""
    layer = LastNonPaddingToken()
    seq_logits = np.ones((1, 4, 8), dtype=np.float32)
    seq_logits[0, 3, :] = 99.0  # last position is distinct
    padding_mask = np.array([[0, 0, 0, 1]], dtype=np.int32)

    result = layer([seq_logits, padding_mask])
    np.testing.assert_array_equal(result[0], seq_logits[0, 3, :])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_train.py::test_last_non_padding_token_basic -v
```

Expected: `ImportError: cannot import name 'LastNonPaddingToken'`

- [ ] **Step 3: Create train.py with LastNonPaddingToken**

```python
# src/hstu_rec/train.py
from __future__ import annotations

import keras
import keras.ops as ops


class LastNonPaddingToken(keras.layers.Layer):
    """Extract the logit vector at the last non-padding position in a sequence.

    Inputs: [seq_logits, padding_mask]
        seq_logits:   float32[batch, seq_len, vocab_size]
        padding_mask: int32[batch, seq_len]  — 1 = real token, 0 = padding

    Output: float32[batch, vocab_size]
    """

    def call(self, inputs):
        seq_logits, padding_mask = inputs
        # [batch]: number of real tokens per example
        lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=1)
        # [batch]: 0-indexed position of the last real token
        last_indices = lengths - 1  # [batch]

        # Expand for take_along_axis: [batch, 1, vocab_size]
        vocab_size = ops.shape(seq_logits)[2]
        indices = ops.reshape(last_indices, (-1, 1, 1))
        indices = ops.broadcast_to(indices, (ops.shape(seq_logits)[0], 1, vocab_size))

        # [batch, 1, vocab_size] → [batch, vocab_size]
        return ops.squeeze(
            ops.take_along_axis(seq_logits, indices, axis=1),
            axis=1,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_train.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hstu_rec/train.py tests/test_train.py
git commit -m "feat: LastNonPaddingToken Keras layer"
```

---

## Task 10: HSTUTask

**Files:**
- Modify: `src/hstu_rec/train.py`
- Modify: `tests/test_train.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_train.py`:

```python
# tests/test_train.py (additions)
import pandas as pd
import tempfile
from pathlib import Path
from hstu_rec.dataset import load_config, make_data_factory
from hstu_rec.preprocess.tfrecords import write_tfrecords
from hstu_rec.train import HSTUTask
import os


@pytest.fixture
def tiny_config_and_data(tmp_path):
    """Build a minimal Config and write TFRecords for 3 users × 5 items."""
    import yaml
    config_dict = {
        "dataset": {
            "name": "test",
            "review_url": "https://fake.example.com/Test.jsonl.gz",
            "min_interactions": 2,
        },
        "model": {
            "vocab_size": None,
            "max_sequence_length": 4,
            "model_dim": 8,
            "num_heads": 2,
            "num_layers": 1,
            "dropout": 0.0,
            "learning_rate": 1e-3,
        },
        "training": {
            "batch_size": 2,
            "train_steps": 10,
            "steps_per_eval": 5,
            "steps_per_loop": 5,
            "model_dir": str(tmp_path / "runs"),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_dict))
    config = load_config(str(config_path))

    rows = []
    for u in range(3):
        for i in range(5):
            rows.append({"user_id": f"u{u}", "parent_asin": f"A{i}", "timestamp": i})
    df = pd.DataFrame(rows)
    data_dir = tmp_path / "data"
    write_tfrecords(df, max_seq_len=4, output_dir=str(data_dir))
    vocab_size = int((data_dir / "vocab_size.txt").read_text())
    config.model.vocab_size = vocab_size
    return config, str(data_dir)


def test_hstu_task_create_dataset_returns_tf_dataset(tiny_config_and_data):
    import tensorflow as tf
    config, data_dir = tiny_config_and_data
    task = HSTUTask(config=config, data_dir=data_dir)
    ds = task.create_dataset(training=True)
    assert isinstance(ds, tf.data.Dataset)


def test_hstu_task_create_model_returns_keras_model(tiny_config_and_data):
    config, data_dir = tiny_config_and_data
    task = HSTUTask(config=config, data_dir=data_dir)
    model = task.create_model()
    assert isinstance(model, keras.Model)
    # Output shape: [batch, vocab_size+1]
    assert model.output_shape == (None, config.model.vocab_size + 1)


def test_hstu_task_model_forward_pass(tiny_config_and_data):
    import numpy as np
    config, data_dir = tiny_config_and_data
    task = HSTUTask(config=config, data_dir=data_dir)
    model = task.create_model()
    batch_size = 2
    seq_len = config.model.max_sequence_length
    x = np.ones((batch_size, seq_len), dtype=np.int32)
    out = model(x, training=False)
    assert out.shape == (batch_size, config.model.vocab_size + 1)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_train.py -v -k "hstu_task"
```

Expected: `ImportError: cannot import name 'HSTUTask'`

- [ ] **Step 3: Add HSTUTask to train.py**

```python
# src/hstu_rec/train.py (add after LastNonPaddingToken)

import tensorflow as tf
import recml
from recml.core.training import keras_trainer
from recml.layers.keras import hstu
from hstu_rec.dataset import Config, make_data_factory
from hstu_rec.metrics import NDCGAtK


class HSTUTask(keras_trainer.KerasTask):
    """RecML KerasTask wrapping the HSTU model for next-item prediction."""

    def __init__(self, config: Config, data_dir: str):
        self._config = config
        self._data_dir = data_dir

    def create_dataset(self, training: bool) -> tf.data.Dataset:
        split = "train" if training else "val"
        factory = make_data_factory(self._config, self._data_dir, split)
        return factory.make()

    def create_model(self, **kwargs) -> keras.Model:
        cfg = self._config.model
        vocab_size_with_pad = cfg.vocab_size + 1  # +1 for padding token (ID 0)

        inputs = keras.Input(
            shape=(cfg.max_sequence_length,), dtype="int32", name="input_ids"
        )
        padding_mask = keras.ops.cast(keras.ops.not_equal(inputs, 0), "int32")

        seq_logits = hstu.HSTU(
            vocab_size=vocab_size_with_pad,
            max_positions=cfg.max_sequence_length,
            model_dim=cfg.model_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )(inputs, padding_mask=padding_mask)

        output_logits = LastNonPaddingToken()([seq_logits, padding_mask])
        output_logits = keras.layers.Activation("linear", dtype="float32")(output_logits)

        model = keras.Model(inputs=inputs, outputs=output_logits)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseTopKCategoricalAccuracy(k=10, name="HR@10"),
                NDCGAtK(k=10, name="NDCG@10"),
            ],
        )
        return model
```

- [ ] **Step 4: Run all train tests to verify they pass**

```bash
uv run pytest tests/test_train.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/hstu_rec/train.py tests/test_train.py
git commit -m "feat: HSTUTask with HSTU model and training/eval datasets"
```

---

## Task 11: Train CLI and End-to-End Smoke Test

**Files:**
- Modify: `src/hstu_rec/train.py`
- Modify: `tests/test_train.py`

- [ ] **Step 1: Write the failing smoke test**

Append to `tests/test_train.py`:

```python
# tests/test_train.py (additions)
from unittest.mock import patch


def test_train_main_smoke(tiny_config_and_data, tmp_path):
    """Smoke test: run a few training steps end-to-end using synthetic data."""
    import yaml
    from hstu_rec.train import main as train_main

    config, data_dir = tiny_config_and_data

    # Write a config file where model_dir and train_steps are minimal
    config_dict = {
        "dataset": {
            "name": "test",
            "review_url": "https://fake.example.com/Test.jsonl.gz",
            "min_interactions": 2,
        },
        "model": {
            "vocab_size": None,
            "max_sequence_length": 4,
            "model_dim": 8,
            "num_heads": 2,
            "num_layers": 1,
            "dropout": 0.0,
            "learning_rate": 1e-3,
        },
        "training": {
            "batch_size": 2,
            "train_steps": 10,
            "steps_per_eval": 5,
            "steps_per_loop": 5,
            "model_dir": str(tmp_path / "model"),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_dict))

    # Should complete without error
    train_main(
        config_path=str(config_path),
        data_dir=data_dir,
        model_dir=None,
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_train.py::test_train_main_smoke -v
```

Expected: `ImportError: cannot import name 'main' from 'hstu_rec.train'`

- [ ] **Step 3: Add main() to train.py**

```python
# src/hstu_rec/train.py (add at end of file)

import argparse
import os
from pathlib import Path


def main(
    config_path: str | None = None,
    data_dir: str | None = None,
    model_dir: str | None = None,
) -> None:
    """CLI entry point for the training stage."""
    if config_path is None or data_dir is None:
        parser = argparse.ArgumentParser(description="Train HSTU recommender")
        parser.add_argument("--config", required=True, help="Path to YAML config file")
        parser.add_argument("--data", required=True, help="Directory with TFRecords")
        parser.add_argument("--model-dir", default=None, help="Override model output dir")
        args = parser.parse_args()
        config_path = args.config
        data_dir = args.data
        model_dir = args.model_dir

    from hstu_rec.dataset import load_config
    import recml
    from recml.core.training import keras_trainer

    config = load_config(config_path)

    if model_dir is not None:
        config.training.model_dir = model_dir

    vocab_size_path = Path(data_dir) / "vocab_size.txt"
    if not vocab_size_path.exists():
        raise FileNotFoundError(
            f"vocab_size.txt not found in {data_dir}. "
            "Run `uv run preprocess` first."
        )
    config.model.vocab_size = int(vocab_size_path.read_text().strip())

    task = HSTUTask(config=config, data_dir=data_dir)
    trainer = keras_trainer.KerasTrainer(
        model_dir=config.training.model_dir,
        train_steps=config.training.train_steps,
        steps_per_eval=config.training.steps_per_eval,
        steps_per_loop=config.training.steps_per_loop,
    )
    experiment = recml.Experiment(task=task, trainer=trainer)
    recml.run_experiment(experiment, recml.Experiment.Mode.TRAIN_AND_EVAL)
```

- [ ] **Step 4: Run the smoke test**

```bash
uv run pytest tests/test_train.py::test_train_main_smoke -v -s
```

Expected: test PASSES (a few training steps complete, loss is printed).

- [ ] **Step 5: Run the full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Verify the train CLI is wired up**

```bash
uv run train --help
```

Expected: prints usage with `--config`, `--data`, `--model-dir`.

- [ ] **Step 7: Final commit**

```bash
git add src/hstu_rec/train.py tests/test_train.py
git commit -m "feat: train CLI main() and end-to-end smoke test"
```

---

## Usage After Implementation

```bash
# One-time setup
uv sync
git submodule update --init

# Preprocess Video_Games
uv run preprocess --config configs/video_games.yaml --output data/video_games

# Train
uv run train --config configs/video_games.yaml --data data/video_games

# Switch to Books (future)
uv run preprocess --config configs/books.yaml --output data/books
uv run train --config configs/books.yaml --data data/books
```
