import json
from pathlib import Path
import subprocess

import nbformat
import numpy as np
import pandas as pd
import pytest

from ml_coursework.config import ROOT
from ml_coursework.data import load_dataset


def test_real_dataset_if_cached():
    path = ROOT / "data" / "online_shoppers_intention.csv"
    if not path.exists():
        pytest.skip("Gerçek veri yerel önbellekte yok; test ağ bağlantısı gerektirmez.")
    frame = load_dataset(path)
    assert frame.shape == (12330, 18)
    assert frame.Revenue.sum() == 1908
    assert frame.isna().sum().sum() == 0


def test_committed_results_selection_and_metric_consistency():
    for task, metric, maximize in [("classification", "f1", True), ("regression", "rmse", False)]:
        frame = pd.read_csv(ROOT / "results" / "tables" / f"{task}_results.csv")
        selected = frame.loc[frame.selected_by_cv, "model"].item()
        best_index = frame[f"cv_{metric}_mean"].idxmax() if maximize else frame[f"cv_{metric}_mean"].idxmin()
        assert selected == frame.loc[best_index, "model"]
        assert np.isfinite(frame[f"cv_{metric}_mean"]).all()
    regression = pd.read_csv(ROOT / "results/tables/regression_results.csv")
    for split in ["train", "test"]:
        np.testing.assert_allclose(regression[f"{split}_rmse"] ** 2, regression[f"{split}_mse"])
    classification = pd.read_csv(ROOT / "results/tables/classification_results.csv")
    for split in ["train", "test"]:
        for metric in ["accuracy", "precision", "recall", "f1"]:
            assert classification[f"{split}_{metric}"].between(0, 1).all()
    manifest = json.loads((ROOT / "results/tables/experiment_manifest.json").read_text(encoding="utf-8"))
    matrices = json.loads((ROOT / "results/tables/classification_selection.json").read_text(encoding="utf-8"))["confusion_matrices"]
    for matrix in matrices.values():
        assert np.array(matrix).sum() == manifest["test_rows"]
    clusters = pd.read_csv(ROOT / "results/tables/clustering_results.csv")
    assert clusters.loc[clusters.selected, "k"].item() == clusters.loc[clusters.silhouette.idxmax(), "k"]
    assert clusters.k.tolist() == [2, 3, 4, 5, 6]
    assert clusters.silhouette.between(-1, 1).all()
    sizes = pd.read_csv(ROOT / "results/tables/cluster_sizes.csv")
    assert sizes.train_count.sum() == manifest["train_rows"]
    assert sizes.test_count.sum() == manifest["test_rows"]


def test_notebook_valid_and_outputs_clean():
    notebook = nbformat.read(ROOT / "notebooks/ML_Donem_Odevi.ipynb", as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code_cells) >= 5
    assert all(cell.execution_count is None and cell.outputs == [] for cell in code_cells)


def test_git_tracks_only_small_source_and_summaries():
    if not (ROOT / ".git").exists():
        pytest.skip("Git olmayan arşiv kopyasında indeks denetlenemez.")
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    forbidden = [".venv/", ".tools/", "results/figures/", "results/models/"]
    for name in filter(None, names):
        assert not any(name.startswith(prefix) for prefix in forbidden)
        assert not (name.startswith("data/") and name.endswith((".csv", ".zip", ".parquet")))
        assert (ROOT / name).stat().st_size <= 2_000_000
    assert sum((ROOT / name).stat().st_size for name in names if name) < 15_000_000
