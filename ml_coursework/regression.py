"""Orijinal süre biriminde, eğitim CV RMSE'siyle regresyon karşılaştırması."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate

from .config import CV_FOLDS, RANDOM_STATE, output_dirs
from .evaluation import regression_metrics, cv_summary, save_figure, save_json
from .preprocessing import build_pipeline, task_features


def regression_candidates(include_log: bool = True):
    models = {"Dummy": DummyRegressor(strategy="mean"), "LinearRegression": LinearRegression()}
    for leaf in [2, 5]:
        models[f"RandomForest_leaf{leaf}"] = RandomForestRegressor(
            n_estimators=80, max_depth=16, min_samples_leaf=leaf, random_state=RANDOM_STATE, n_jobs=1,
        )
    if include_log:
        models["RandomForest_log1p_leaf2"] = TransformedTargetRegressor(
            regressor=clone(models["RandomForest_leaf2"]), func=np.log1p, inverse_func=np.expm1,
        )
    return {name: build_pipeline(model, scale=name in {"Dummy", "LinearRegression"}) for name, model in models.items()}


def run_regression(frame, train_indices, test_indices, output_dir: Path):
    tables, figures = output_dirs(output_dir)
    features = task_features(frame, "regression")
    x_train, y_train = features.loc[train_indices], frame.loc[train_indices, "ProductRelated_Duration"]
    train_skew = float(y_train.skew())
    candidates = regression_candidates(include_log=train_skew > 1)
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"mae": "neg_mean_absolute_error", "mse": "neg_mean_squared_error",
               "rmse": "neg_root_mean_squared_error", "r2": "r2"}
    cv_rows, fold_rows = [], []
    for name, pipeline in candidates.items():
        print(f"Regression CV: {name}", flush=True)
        scores = cross_validate(pipeline, x_train, y_train, cv=cv, scoring=scoring, n_jobs=1, error_score="raise")
        cv_rows.append({"model": name, **cv_summary(scores, scoring, negative=["mae", "mse", "rmse"])})
        for fold in range(CV_FOLDS):
            fold_rows.append({"model": name, "fold": fold + 1,
                              **{m: scores[f"test_{m}"][fold] * (-1 if m != "r2" else 1) for m in scoring}})
    cv_table = pd.DataFrame(cv_rows).set_index("model")
    forest_names = [n for n in candidates if n.startswith("RandomForest")]
    selected_forest = cv_table.loc[forest_names, "cv_rmse_mean"].idxmin()
    final_names = ["Dummy", "LinearRegression", selected_forest]
    selected_model = cv_table.loc[final_names, "cv_rmse_mean"].idxmin()
    # Test bu noktaya kadar model/ayar seçimine katılmadı.
    x_test, y_test = features.loc[test_indices], frame.loc[test_indices, "ProductRelated_Duration"]
    rows, fitted = [], {}
    for name in final_names:
        pipeline = clone(candidates[name]).fit(x_train, y_train)
        fitted[name] = pipeline
        row = {"model": name, **cv_table.loc[name].to_dict(), "selected_by_cv": name == selected_model}
        for split, x, y in [("train", x_train, y_train), ("test", x_test, y_test)]:
            prediction = pipeline.predict(x)
            row.update({f"{split}_{metric}": value for metric, value in regression_metrics(y, prediction).items()})
            # Hedefte sıfırlar olduğundan MAPE bilinçli olarak boş bırakılır.
            row[f"{split}_mape"] = None
        rows.append(row)
        prediction = pipeline.predict(x_test)
        plt.figure(figsize=(6, 5))
        plt.scatter(y_test, prediction, s=8, alpha=.35)
        limits = [min(y_test.min(), prediction.min()), max(y_test.max(), prediction.max())]
        plt.plot(limits, limits, "r--")
        plt.xlabel("Gerçek ürün sayfası süresi (saniye)")
        plt.ylabel("Tahmin (saniye)")
        plt.title(name)
        save_figure(figures / f"actual_predicted_{name}.png")
        plt.figure(figsize=(6, 4))
        plt.scatter(prediction, y_test - prediction, s=8, alpha=.35)
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel("Tahmin (saniye)")
        plt.ylabel("Hata: gerçek − tahmin (saniye)")
        plt.title(name)
        save_figure(figures / f"residuals_{name}.png")
    results = pd.DataFrame(rows)
    results.to_csv(tables / "regression_results.csv", index=False)
    cv_table.reset_index().to_csv(tables / "regression_candidates.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(tables / "regression_cv_folds.csv", index=False)
    parameters = {}
    for name, pipeline in candidates.items():
        model = pipeline.named_steps["model"]
        params = model.regressor.get_params() if isinstance(model, TransformedTargetRegressor) else model.get_params()
        parameters[name] = {"parameters": params, "target_transform": "log1p" if "log1p" in name else "none"}
    save_json({"selected_model": selected_model, "selected_forest": selected_forest,
               "selection_metric": "training_5_fold_mean_RMSE", "training_target_skew": train_skew,
               "training_zero_targets": int(y_train.eq(0).sum()), "unit": "seconds",
               "mape_omitted_reason": "Hedef sıfır ve sıfıra yakın süreler içeriyor; yüzde hata kararsızdır.",
               "parameters": parameters}, tables / "regression_selection.json")
    results.set_index("model")[["train_rmse", "test_rmse", "cv_rmse_mean"]].plot.bar(figsize=(9, 4))
    plt.ylabel("RMSE (saniye)")
    plt.xticks(rotation=15)
    save_figure(figures / "regression_comparison.png")
    print(results.to_string(index=False), flush=True)
    return results, fitted
