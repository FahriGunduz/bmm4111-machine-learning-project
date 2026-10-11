import numpy as np
import pytest

from ml_coursework.data import validate_dataset, shared_split, prepare_modeling_data, load_dataset


def test_schema_validation(sample_frame):
    frame = validate_dataset(sample_frame, expected_rows=120)
    assert frame.shape == (120, 18)
    assert frame.Revenue.dtype == bool
    with pytest.raises(ValueError):
        validate_dataset(sample_frame)
    with pytest.raises(ValueError):
        validate_dataset(sample_frame.drop(columns="Revenue"), expected_rows=None)


def test_invalid_boolean_and_target(sample_frame):
    frame = sample_frame.copy()
    frame["Revenue"] = "unknown"
    with pytest.raises(ValueError):
        validate_dataset(frame, expected_rows=None)
    frame = sample_frame.copy()
    frame.loc[0, "ProductRelated_Duration"] = -1
    with pytest.raises(ValueError):
        validate_dataset(frame, expected_rows=None)


def test_shared_stratified_split(sample_frame):
    train, test = shared_split(sample_frame)
    assert len(train) == 96 and len(test) == 24
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(sample_frame.index)
    assert abs(sample_frame.loc[train].Revenue.mean() - sample_frame.loc[test].Revenue.mean()) < .04
    again = shared_split(sample_frame)
    assert all(np.array_equal(a, b) for a, b in zip((train, test), again))


def test_deduplication_before_split(sample_frame):
    import pandas as pd
    duplicated = pd.concat([sample_frame, sample_frame.iloc[:5]], ignore_index=True)
    clean = prepare_modeling_data(duplicated)
    assert len(clean) == 120
    train, test = shared_split(clean)
    hashes = pd.util.hash_pandas_object(clean, index=False)
    assert set(hashes.loc[train]).isdisjoint(hashes.loc[test])


def test_offline_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.csv")
