import importlib
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, cross_validate
from threadpoolctl import threadpool_limits

from ml_coursework.classification import classification_candidates
from ml_coursework.regression import regression_candidates
from ml_coursework.data import shared_split
from ml_coursework.evaluation import classification_metrics, regression_metrics
from ml_coursework.preprocessing import task_features, build_pipeline


@pytest.mark.parametrize("name", ["Dummy", "LogisticRegression", "ANN_32", "ANN_64_32"])
def test_classification_predictions_and_repeatability(sample_frame, name):
    train, test = shared_split(sample_frame)
    x = task_features(sample_frame, "classification")
    pipeline = classification_candidates()[name]
    with threadpool_limits(limits=1):
        fitted = clone(pipeline).fit(x.loc[train], sample_frame.loc[train, "Revenue"])
        prediction = fitted.predict(x.loc[test])
        repeated = clone(pipeline).fit(x.loc[train], sample_frame.loc[train, "Revenue"]).predict(x.loc[test])
    np.testing.assert_array_equal(prediction, repeated)
    assert prediction.shape == (len(test),)
    assert set(prediction) <= {True, False}
    assert all(0 <= v <= 1 for v in classification_metrics(sample_frame.loc[test, "Revenue"], prediction).values())


@pytest.mark.parametrize("name", ["Dummy", "LinearRegression", "RandomForest_leaf2", "RandomForest_leaf5", "RandomForest_log1p_leaf2"])
def test_regression_predictions_and_repeatability(sample_frame, name):
    train, test = shared_split(sample_frame)
    x = task_features(sample_frame, "regression")
    y = sample_frame.ProductRelated_Duration
    pipeline = regression_candidates()[name]
    with threadpool_limits(limits=1):
        prediction = clone(pipeline).fit(x.loc[train], y.loc[train]).predict(x.loc[test])
        repeated = clone(pipeline).fit(x.loc[train], y.loc[train]).predict(x.loc[test])
    np.testing.assert_allclose(prediction, repeated, rtol=0, atol=1e-10)
    assert prediction.shape == (len(test),) and np.isfinite(prediction).all()
    metrics = regression_metrics(y.loc[test], prediction)
    assert metrics["mae"] >= 0 and metrics["mse"] >= 0 and metrics["r2"] <= 1
    assert metrics["rmse"] ** 2 == pytest.approx(metrics["mse"])


def test_cv_preprocessor_fits_each_fold_training_only(sample_frame):
    train, _ = shared_split(sample_frame)
    x = task_features(sample_frame.loc[train], "classification")
    y = sample_frame.loc[train, "Revenue"]
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    folds = list(cv.split(x, y))
    with threadpool_limits(limits=1):
        scores = cross_validate(classification_candidates()["LogisticRegression"], x, y,
                                cv=folds, return_estimator=True, error_score="raise")
    for (fold_train, _), estimator in zip(folds, scores["estimator"]):
        transformer = estimator.named_steps["preprocessor"]
        numeric_columns = transformer.transformers_[0][2]
        scaler = transformer.named_transformers_["numeric"].named_steps["scaler"]
        np.testing.assert_allclose(scaler.mean_, x.iloc[fold_train][numeric_columns].mean())
        assert scaler.n_samples_seen_ == len(fold_train)


def test_clustering_reproducibility_and_no_test_fit(sample_frame):
    train, test = shared_split(sample_frame)
    x = task_features(sample_frame, "clustering")
    pipeline = build_pipeline(KMeans(n_clusters=3, n_init=10, random_state=42), drop_first=False)
    with threadpool_limits(limits=1):
        fitted = clone(pipeline).fit(x.loc[train])
        before = fitted.named_steps["model"].cluster_centers_.copy()
        prediction = fitted.predict(x.loc[test])
        repeated = clone(pipeline).fit(x.loc[train]).predict(x.loc[test])
    np.testing.assert_array_equal(prediction, repeated)
    np.testing.assert_array_equal(before, fitted.named_steps["model"].cluster_centers_)
    assert prediction.shape == (len(test),)


@pytest.mark.parametrize("module", ["config", "data", "eda", "preprocessing", "classification", "regression", "clustering", "evaluation", "run_all"])
def test_all_modules_import(module):
    importlib.import_module(f"ml_coursework.{module}")
