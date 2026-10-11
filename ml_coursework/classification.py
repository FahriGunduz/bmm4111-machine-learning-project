"""Yalnız eğitim CV'siyle ANN mimarisi ve sınıflandırma modeli seçimi."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier

from .config import CV_FOLDS, RANDOM_STATE, output_dirs
from .evaluation import classification_metrics, cv_summary, save_figure, save_json
from .preprocessing import build_pipeline, task_features


def classification_candidates():
    models = {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
    }
    for name, architecture in [("ANN_32", (32,)), ("ANN_64_32", (64, 32))]:
        models[name] = MLPClassifier(
            hidden_layer_sizes=architecture, alpha=.001, learning_rate_init=.001,
            max_iter=200, early_stopping=True, validation_fraction=.15,
            n_iter_no_change=15, random_state=RANDOM_STATE,
        )
    return {name: build_pipeline(model) for name, model in models.items()}


def run_classification(frame, train_indices, test_indices, output_dir: Path):
    tables, figures = output_dirs(output_dir)
    features = task_features(frame, "classification")
    x_train, y_train = features.loc[train_indices], frame.loc[train_indices, "Revenue"]
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"accuracy": "accuracy", "precision": make_scorer(precision_score, zero_division=0),
               "recall": make_scorer(recall_score, zero_division=0), "f1": make_scorer(f1_score, zero_division=0)}
    candidates = classification_candidates()
    cv_rows, fold_rows = [], []
    for name, pipeline in candidates.items():
        print(f"Classification CV: {name}", flush=True)
        scores = cross_validate(pipeline, x_train, y_train, cv=cv, scoring=scoring, n_jobs=1, error_score="raise")
        cv_rows.append({"model": name, **cv_summary(scores, scoring)})
        for fold in range(CV_FOLDS):
            fold_rows.append({"model": name, "fold": fold + 1, **{m: scores[f"test_{m}"][fold] for m in scoring}})
    cv_table = pd.DataFrame(cv_rows).set_index("model")
    ann_names = [name for name in candidates if name.startswith("ANN_")]
    selected_ann = cv_table.loc[ann_names, "cv_f1_mean"].idxmax()
    final_names = ["Dummy", "LogisticRegression", selected_ann]
    selected_model = cv_table.loc[final_names, "cv_f1_mean"].idxmax()
    # Seçim burada tamamlandı; bundan sonra test yalnız nihai karşılaştırma içindir.
    x_test, y_test = features.loc[test_indices], frame.loc[test_indices, "Revenue"]
    rows, matrices, fitted = [], {}, {}
    for name in final_names:
        pipeline = clone(candidates[name]).fit(x_train, y_train)
        fitted[name] = pipeline
        row = {"model": name, **cv_table.loc[name].to_dict(), "selected_by_cv": name == selected_model}
        for split, x, y in [("train", x_train, y_train), ("test", x_test, y_test)]:
            prediction = pipeline.predict(x)
            row.update({f"{split}_{metric}": value for metric, value in classification_metrics(y, prediction).items()})
        row["fit_iterations"] = getattr(pipeline.named_steps["model"], "n_iter_", None)
        rows.append(row)
        matrix = confusion_matrix(y_test, pipeline.predict(x_test), labels=[False, True])
        matrices[name] = matrix.tolist()
        ConfusionMatrixDisplay(matrix, display_labels=["Satın almadı", "Satın aldı"]).plot(cmap="Blues")
        plt.title(name)
        save_figure(figures / f"confusion_matrix_{name}.png")
    results = pd.DataFrame(rows)
    results.to_csv(tables / "classification_results.csv", index=False)
    ann_table = cv_table.loc[ann_names].reset_index()
    ann_table["selected"] = ann_table.model.eq(selected_ann)
    ann_table.to_csv(tables / "ann_architecture_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(tables / "classification_cv_folds.csv", index=False)
    save_json({"selected_model": selected_model, "selected_ann": selected_ann,
               "selection_metric": "training_5_fold_mean_F1", "decision_threshold": .5,
               "confusion_matrices": matrices,
               "parameters": {n: p.named_steps["model"].get_params() for n, p in candidates.items()}},
              tables / "classification_selection.json")
    results.set_index("model")[["train_f1", "test_f1", "cv_f1_mean"]].plot.bar(figsize=(8, 4))
    plt.ylabel("F1 (pozitif sınıf)")
    plt.xticks(rotation=0)
    save_figure(figures / "classification_comparison.png")
    plt.figure(figsize=(6, 4))
    plt.bar(ann_table.model, ann_table.cv_f1_mean, yerr=ann_table.cv_f1_std, capsize=5)
    plt.ylabel("CV F1 ortalama ± örnek standart sapması")
    save_figure(figures / "ann_architectures.png")
    print(results.to_string(index=False), flush=True)
    return results, fitted
