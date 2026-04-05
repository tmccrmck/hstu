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
