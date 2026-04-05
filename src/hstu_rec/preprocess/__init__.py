from __future__ import annotations

import argparse

from hstu_rec.dataset import load_config
from hstu_rec.preprocess import download as _download
from hstu_rec.preprocess import filter as _filter
from hstu_rec.preprocess import tfrecords as _tfrecords


def main(config_path: str | None = None, output_dir: str | None = None) -> None:
    """CLI entry point for the preprocessing stage.

    Downloads, filters, and writes TFRecords for the configured dataset.
    """
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
