"""Komut satırından aşamalı veya tam deney yürütme."""
import argparse
import hashlib
from importlib.metadata import version
from pathlib import Path
import platform

import numpy as np
from threadpoolctl import threadpool_limits

from .config import ROOT, RANDOM_STATE, FEATURE_COLUMNS, output_dirs
from .data import load_dataset, prepare_modeling_data, shared_split
from .eda import run_eda
from .evaluation import save_json


def prepare_experiment(data_path=None, output_dir=ROOT / "results"):
    raw = load_dataset(data_path)
    frame = prepare_modeling_data(raw)
    train_indices, test_indices = shared_split(frame)
    tables, _ = output_dirs(output_dir)
    manifest = {
        "random_state": RANDOM_STATE, "raw_rows": len(raw), "modeling_rows": len(frame),
        "train_rows": len(train_indices), "test_rows": len(test_indices), "features": FEATURE_COLUMNS,
        "duplicates_removed": len(raw) - len(frame),
        "train_index_sha256": hashlib.sha256(train_indices.astype(np.int64).tobytes()).hexdigest(),
        "test_index_sha256": hashlib.sha256(test_indices.astype(np.int64).tobytes()).hexdigest(),
        "data_sha256": hashlib.sha256(raw.to_csv(index=False).encode()).hexdigest(),
        "python": platform.python_version(),
        "versions": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "ucimlrepo"]},
        "cv_folds": 5, "cv_n_jobs": 1, "blas_threads": 1,
    }
    save_json(manifest, tables / "experiment_manifest.json")
    return raw, frame, train_indices, test_indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "eda", "classification", "regression", "clustering"], default="all")
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    raw, frame, train_indices, test_indices = prepare_experiment(args.data_path, args.output_dir)
    with threadpool_limits(limits=1):
        if args.stage in {"all", "eda"}:
            run_eda(raw, args.output_dir)
        if args.stage in {"all", "classification"}:
            from .classification import run_classification
            run_classification(frame, train_indices, test_indices, args.output_dir)
        if args.stage in {"all", "regression"}:
            from .regression import run_regression
            run_regression(frame, train_indices, test_indices, args.output_dir)
        if args.stage in {"all", "clustering"}:
            from .clustering import run_clustering
            run_clustering(frame, train_indices, test_indices, args.output_dir)


if __name__ == "__main__":
    main()
