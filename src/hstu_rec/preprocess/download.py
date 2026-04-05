from __future__ import annotations

import gzip
import shutil
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm


def download(url: str, output_dir: str | Path) -> Path:
    """Download and decompress a .jsonl.gz file.

    Skips if the decompressed .jsonl already exists at output_dir/cache/<name>.jsonl.

    Args:
        url: URL to a .jsonl.gz file.
        output_dir: Root output directory.

    Returns:
        Path to the decompressed .jsonl file.
    """
    filename_gz = Path(urlparse(url).path).name
    filename_jsonl = filename_gz.replace(".jsonl.gz", ".jsonl")
    cache_dir = Path(output_dir) / "cache"
    jsonl_path = cache_dir / filename_jsonl

    if jsonl_path.exists():
        print(f"Cache hit: {jsonl_path}")
        return jsonl_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    gz_path = cache_dir / filename_gz

    print(f"Downloading {url} ...")
    try:
        with requests.get(url, stream=True) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(gz_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))
    except Exception:
        gz_path.unlink(missing_ok=True)
        raise

    print(f"Decompressing to {jsonl_path} ...")
    try:
        with gzip.open(gz_path, "rb") as f_in, open(jsonl_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    except Exception:
        jsonl_path.unlink(missing_ok=True)
        raise

    gz_path.unlink()
    return jsonl_path
