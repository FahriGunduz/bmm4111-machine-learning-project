import numpy as np
import pytest

from ml_coursework.config import EXCLUDED_COLUMNS
from ml_coursework.preprocessing import task_features, build_preprocessor


@pytest.mark.parametrize("task", ["classification", "regression", "clustering"])
def test_targets_and_derivatives_excluded(sample_frame, task):
    sample_frame["Revenue_copy"] = sample_frame.Revenue
    sample_frame["duration_per_page"] = sample_frame.ProductRelated_Duration / sample_frame.ProductRelated
    features = task_features(sample_frame, task)
    assert not EXCLUDED_COLUMNS.intersection(features.columns)
    assert "Revenue_copy" not in features and "duration_per_page" not in features
    assert features.shape[1] == 15


def test_preprocessor_learns_only_training(sample_frame):
    train = task_features(sample_frame.iloc[:90], "classification")
    test = task_features(sample_frame.iloc[90:], "classification")
    test.loc[:, "Administrative"] = 1_000_000
    test.loc[:, "Month"] = "unseen_month"
    transformer = build_preprocessor()
    transformed_train = transformer.fit_transform(train)
    numeric = transformer.named_transformers_["numeric"]
    numeric_columns = transformer.transformers_[0][2]
    np.testing.assert_allclose(numeric.named_steps["imputer"].statistics_, train[numeric_columns].median())
    before = numeric.named_steps["scaler"].mean_.copy()
    np.testing.assert_allclose(before, train[numeric_columns].mean())
    with pytest.warns(UserWarning, match="unknown categories"):
        transformed_test = transformer.transform(test)
    np.testing.assert_array_equal(before, numeric.named_steps["scaler"].mean_)
    assert transformed_test.shape[1] == transformed_train.shape[1]
    categories = transformer.named_transformers_["categorical"].named_steps["encoder"].categories_
    assert "unseen_month" not in categories[0]


def test_numeric_codes_are_categorical(sample_frame):
    transformer = build_preprocessor().fit(task_features(sample_frame, "clustering"))
    columns = transformer.transformers_[1][2]
    assert {"OperatingSystems", "Browser", "Region", "TrafficType"} <= set(columns)
