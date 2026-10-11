"""Metrikler ve küçük, yerel grafiklerin ortak yardımcıları."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def save_json(value, path: Path):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_figure(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()


def classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"mae": mean_absolute_error(y_true, y_pred), "mse": mse,
            "rmse": float(np.sqrt(mse)), "r2": r2_score(y_true, y_pred)}


def cv_summary(scores, metrics, negative=()):
    result = {}
    for metric in metrics:
        values = scores[f"test_{metric}"] * (-1 if metric in negative else 1)
        result[f"cv_{metric}_mean"] = float(np.mean(values))
        result[f"cv_{metric}_std"] = float(np.std(values, ddof=1))
    return result
